import math
import os
import sys

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.nn import voxel_grid, max_pool, GMMConv, global_mean_pool
from torchmetrics import Accuracy

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.left_conv1 = GMMConv(in_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn1 = torch.nn.BatchNorm1d(out_channel)

        self.left_conv2 = GMMConv(out_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn2 = torch.nn.BatchNorm1d(out_channel)

        self.shortcut_conv = GMMConv(in_channel, out_channel, dim=3, kernel_size=1)
        self.shortcut_bn = torch.nn.BatchNorm1d(out_channel)

    def forward(self, data):
        data.x = F.elu(
            self.left_bn2(
                self.left_conv2(F.elu(self.left_bn1(self.left_conv1(data.x, data.edge_index, data.edge_attr))),
                                data.edge_index, data.edge_attr)) + self.shortcut_bn(
                self.shortcut_conv(data.x, data.edge_index, data.edge_attr)))
        return data


class AllGraphBlock(torch.nn.Module):
    def __init__(self, out_dim, Config):
        super(AllGraphBlock, self).__init__()
        self.Config = Config
        self.conv1 = GMMConv(3, out_dim, dim=3, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(out_dim)
        self.block1 = ResidualBlock(out_dim, out_dim * 2)
        self.block2 = ResidualBlock(out_dim * 2, out_dim * 4)
        self.block3 = ResidualBlock(out_dim * 4, out_dim * 8)

    def step(self, graph):
        graph.x = F.elu(self.bn1(self.conv1(graph.x, graph.edge_index, graph.edge_attr)))
        cluster = voxel_grid(graph.pos, batch=graph.batch, size=4)
        graph = max_pool(cluster, graph, transform=T.Cartesian(cat=False))

        graph = self.block1(graph)
        cluster = voxel_grid(graph.pos, batch=graph.batch, size=6)
        graph = max_pool(cluster, graph, transform=T.Cartesian(cat=False))

        graph = self.block2(graph)
        cluster = voxel_grid(graph.pos, batch=graph.batch, size=24)
        graph = max_pool(cluster, graph, transform=T.Cartesian(cat=False))
        graph = self.block3(graph)
        cluster = voxel_grid(graph.pos, batch=graph.batch, size=64)
        graph = max_pool(cluster, graph, transform=T.Cartesian(cat=False))
        x = global_mean_pool(graph.x, batch=graph.batch)
        return x.unsqueeze(1)

    def forward(self, graphs):
        x = self.step(graphs)
        x = x.view(-1, 5, 192)
        return x

    def batch_to_tensor(self, data_batch, batch_size, num_graphs_per_batch, max_nodes=None):

        num_features = data_batch.num_features

        # 如果没有提供最大节点数，从batch数据中推断
        if max_nodes is None:
            max_nodes = 0
            for i in range(data_batch.num_graphs):
                node_mask = data_batch.batch == i
                max_nodes = max(max_nodes, node_mask.sum().item())

        # 初始化填充后的tensor
        tensor = torch.zeros(batch_size, num_graphs_per_batch, max_nodes, num_features)

        # 从Batch对象中提取数据
        current_batch = 0
        graph_idx = 0
        start_idx = 0
        for i in range(data_batch.num_graphs):
            # 找出当前图的节点范围
            node_mask = data_batch.batch == i
            num_nodes = node_mask.sum().item()
            end_idx = start_idx + num_nodes

            # 提取节点特征
            features = data_batch.x[start_idx:end_idx]

            # 填充到tensor中
            tensor[current_batch, graph_idx, :num_nodes] = features

            # 更新索引
            start_idx = end_idx
            graph_idx += 1
            if graph_idx == num_graphs_per_batch:
                graph_idx = 0
                current_batch += 1

        return tensor



class AttentionBlock(nn.Module):
    def __init__(self, opt_dim, heads, dropout, att_dropout, **args):
        super(AttentionBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            opt_dim,  # embed_dim
            heads,  # num_heads
            dropout=att_dropout,
            bias=True,
            add_bias_kv=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(opt_dim, opt_dim)
        self.linear2 = nn.Linear(opt_dim, opt_dim)
        self.linear3 = nn.Linear(opt_dim, opt_dim)
        self.ln_x = nn.LayerNorm(opt_dim)
        self.ln_z = nn.LayerNorm(opt_dim)
        self.ln_att = nn.LayerNorm(opt_dim)
        self.ln1 = nn.LayerNorm(opt_dim)
        self.ln2 = nn.LayerNorm(opt_dim)

    def forward(self, x, z, mask=None, q_mask=None, **args):
        x = self.ln_x(x)
        z = self.ln_z(z)
        z_att, _ = self.attention(z, x, x, key_padding_mask=mask, attn_mask=q_mask)  # Q, K, V
        z_att = z_att + z
        z = self.ln_att(z)
        z = self.dropout(z)
        z = self.linear1(z)
        z = self.ln1(z)
        z = F.elu(z)
        z = self.dropout(z)
        z = self.linear2(z)
        z = self.ln2(z)
        z = F.elu(z)
        z = self.dropout(z)
        z = self.linear3(z)
        return z + z_att


class StrModule(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(StrModule, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[: -2] + (self.all_head_size,)
        context = context.view(*new_size)

        last_query = query[:, -1, :].unsqueeze(1)
        last_query_r = torch.repeat_interleave(last_query, repeats=key.shape[1], dim=1)
        out = torch.sum(torch.mul((last_query_r - key), value), dim=1)
        op = torch.cat((out, context[:, -1, :]), dim=1)
        return op


class ClfBlok(nn.Module):
    def __init__(self, int_dim, hid_dim, out_dim=20):
        super().__init__()
        self.linear1 = nn.Linear(int_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GraphAttention(nn.Module):
    def __init__(self, num_attention, out_dim, heads, num_class, Config):
        super(GraphAttention, self).__init__()
        self.graph_model = AllGraphBlock(out_dim=out_dim, Config=Config)
        # self.attentions = nn.ModuleList([
        #     AttentionBlock(opt_dim=out_dim * 8, heads=heads, dropout=0.5, att_dropout=0) for _ in range(num_attention)])
        self.cls = ClfBlok(int_dim=out_dim * 8 * 2, hid_dim=64, out_dim=num_class)
        self.str = StrModule(num_attention_heads=heads, input_size=out_dim * 8, hidden_size=out_dim * 8)
        # self.accuracy = Accuracy(task="multiclass", num_classes=20)

    def forward(self, graphs):
        x = self.graph_model(graphs)
        x = self.str.forward(x)
        # for attention in self.attentions:
        #     x = attention(x, x)
        # x = x.mean(dim=1)
        x = self.cls(x)
        return F.log_softmax(x, dim=1)


class AttModel(nn.Module):
    def __init__(self, modulelist):
        super(AttModel, self).__init__()
        self.model = modulelist

    def forward(self, x):
        for module in self.model:
            x = module(x, x)
        return x.mean(dim=1)


class GraphAttention2(nn.Module):
    def __init__(self, num_attention, out_dim, heads, num_class):
        super(GraphAttention2, self).__init__()
        self.graph_model = AllGraphBlock(out_dim=out_dim)
        self.attentions = AttModel(nn.ModuleList([
            AttentionBlock(opt_dim=out_dim * 8, heads=heads, dropout=0.5, att_dropout=0) for _ in
            range(num_attention)]))
        self.cls = ClfBlok(int_dim=out_dim * 8, hid_dim=64, out_dim=num_class)
        # self.accuracy = Accuracy(task="multiclass", num_classes=20)

    def forward(self, graphs):
        x = self.graph_model(graphs)
        x = self.attentions(x)
        x = self.cls(x)
        return F.log_softmax(x, dim=1)


class PlGraphAttention(LightningModule):
    def __init__(self, Config, num_attention, learning_rate, f_graph_dim, heads):
        super(PlGraphAttention, self).__init__()
        self.save_hyperparameters()
        self.model = GraphAttention(num_attention=num_attention, out_dim=f_graph_dim, heads=heads,
                                    num_class=Config.num_class, Config=Config)
        self.accuracy = Accuracy(num_classes=Config.num_class)
        self.Config = Config

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[20, 40, 60, 80, 100, 110, 120, 130, 140],
                                                      gamma=0.8)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        y = batch.y[0::self.Config.split_graph_num]
        loss = F.nll_loss(preds, y)
        return {"loss": loss, "preds": preds, "y": y}

    def training_step_end(self, outputs):
        train_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("train_acc", train_acc, prog_bar=True, batch_size=self.Config.batch_size)
        self.log("train_loss", outputs['loss'], prog_bar=True, batch_size=self.Config.batch_size)
        return {"loss": outputs["loss"].mean()}

    def validation_step(self, batch, batch_idex):
        preds = self(batch)
        y = batch.y[0::self.Config.split_graph_num]
        loss = F.nll_loss(preds, y)
        return {"loss": loss, "preds": preds, "y": y}

    def validation_step_end(self, outputs):
        val_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("val_loss", outputs["loss"].mean(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.batch_size)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.Config.batch_size)

    def test_step(self, batch, batch_idx):
        preds = self(batch)
        y = batch.y[0::self.Config.split_graph_num]
        loss = F.nll_loss(preds, y)
        return {"loss": loss, "preds": preds, "y": y}

    def test_step_end(self, outputs):
        val_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("val_loss", outputs["loss"].mean(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.batch_size)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.Config.batch_size)


class PlGraphAttention_casia(LightningModule):
    def __init__(self, Config, num_attention, learning_rate, f_graph_dim, heads):
        super(PlGraphAttention_casia, self).__init__()
        self.save_hyperparameters()
        self.model = GraphAttention2(num_attention=num_attention, out_dim=f_graph_dim, heads=heads,
                                     num_class=Config.step1_num_class)
        self.accuracy = Accuracy(num_classes=Config.step1_num_class)
        self.Config = Config

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[20, 40, 60, 80, 100, 110, 120, 130, 140],
                                                      gamma=0.8)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        y = batch[0].y - 1
        loss = F.nll_loss(preds, y)
        return {"loss": loss, "preds": preds, "y": y}

    def training_step_end(self, outputs):
        train_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("train_acc", train_acc, prog_bar=True, batch_size=self.Config.step1_batch_size)
        self.log("train_loss", outputs['loss'], prog_bar=True, batch_size=self.Config.step1_batch_size)
        return {"loss": outputs["loss"].mean()}

    def validation_step(self, batch, batch_idex):
        preds = self(batch)
        y = batch[0].y - 1
        loss = F.nll_loss(preds, y)
        return {"loss": loss, "preds": preds, "y": y}

    def validation_step_end(self, outputs):
        val_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("val_loss", outputs["loss"].mean(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step1_batch_size)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step1_batch_size)

    def test_step(self, batch, batch_idx):
        preds = self(batch)
        y = batch[0].y - 1
        loss = F.nll_loss(preds, y)
        return {"loss": loss, "preds": preds, "y": y}

    def test_step_end(self, outputs):
        val_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("val_loss", outputs["loss"].mean(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step1_batch_size)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step1_batch_size)


class PlGraphAttention_casia_2(LightningModule):
    def __init__(self, Config, pretrained_model, learning_rate, Gout_dim):
        super(PlGraphAttention_casia_2, self).__init__()
        self.save_hyperparameters(ignore=['pretrained_model'])
        self.cls = ClfBlok(int_dim=Gout_dim * 8, hid_dim=64, out_dim=Config.step2_num_class)
        self.pretrained_model = list(list(pretrained_model.children())[0].children())
        self.pretrained_model[2] = self.cls
        self.model = nn.Sequential(*self.pretrained_model)
        self.accuracy = Accuracy(num_classes=Config.step2_num_class)
        self.Config = Config

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[20, 40, 60, 80, 100, 110, 120, 130, 140],
                                                      gamma=0.7)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        y = batch[0].y - 75
        loss = F.nll_loss(preds, y)
        return {"loss": loss, "preds": preds, "y": y}

    def training_step_end(self, outputs):
        train_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("train_acc", train_acc, prog_bar=True, batch_size=self.Config.step2_batch_size)
        self.log("train_loss", outputs['loss'], prog_bar=True, batch_size=self.Config.step2_batch_size)
        return {"loss": outputs["loss"].mean()}

    def validation_step(self, batch, batch_idex):
        preds = self(batch)
        y = batch[0].y - 75
        loss = F.nll_loss(preds, y)
        return {"loss": loss, "preds": preds, "y": y}

    def validation_step_end(self, outputs):
        val_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("val_loss", outputs["loss"].mean(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step2_batch_size)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step2_batch_size)

    def test_step(self, batch, batch_idx):
        preds = self(batch)
        y = batch[0].y - 75
        loss = F.nll_loss(preds, y)
        return {"loss": loss, "preds": preds, "y": y}

    def test_step_end(self, outputs):
        val_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("val_loss", outputs["loss"].mean(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step2_batch_size)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step2_batch_size)



if __name__ == '__main__':
    # model = PlGraphAttention(num_attention=1, learning_rate=0.001, f_graph_dim=16, heads=2)
    # model_size = pl.utilities.memory.get_model_size_mb(model)
    # print("model_size = {} M \n".format(model_size))
    x = torch.rand((32, 20, 20))
    attention = StrModule(2, 20, 20)
    result = attention.forward(x)
