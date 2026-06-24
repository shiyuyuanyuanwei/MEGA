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


class EdgeFeatureGenerator(nn.Module):

    def __init__(self, norm_params=None, level=0):
        super().__init__()
        self.norm_params = norm_params or {'W': 128.0, 'H': 128.0}
        self.level = level

    def forward(self, pos, edge_index):

        d = pos[edge_index[1]] - pos[edge_index[0]]

        dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)


        W, H = self.norm_params['W'], self.norm_params['H']


        if self.level >= 1:

            t_span = (pos[:, 0].max() - pos[:, 0].min()).clamp_min(1e-6)
            dt_norm = d[..., 0] / t_span
        else:

            h = 128.0
            dt_norm = d[..., 0] / h


        d_norm = torch.stack([
            d[..., 1] / W,
            d[..., 2] / H,
            dt_norm
        ], dim=-1)


        dist_norm_for_dir = d_norm.norm(dim=-1, keepdim=True).clamp_min(1e-6)


        dir_norm = d_norm / dist_norm_for_dir
        dir_norm = torch.nan_to_num(dir_norm, nan=0.0, posinf=1.0, neginf=-1.0)


        if dist.numel() >= 16:
            q = torch.quantile(dist.squeeze(), 0.95).clamp_min(1e-6)
        else:
            q = dist.mean().clamp_min(1e-6)
        radius_norm = dist / q
        radius_norm = torch.nan_to_num(radius_norm, nan=1e-6, posinf=1.0, neginf=1e-6)


        dt_abs = torch.abs(d[..., 0]).clamp_min(1e-6)
        ds = d[..., 1:3].norm(dim=-1)


        velocity = ds / dt_abs


        motion_strength = torch.log1p(velocity)


        if motion_strength.numel() >= 16:
            motion_q = torch.quantile(motion_strength, 0.95).clamp_min(1e-6)
        else:
            motion_q = motion_strength.mean().clamp_min(1e-6)
        motion_norm = motion_strength / motion_q
        motion_norm = torch.clamp(motion_norm, 0, 2.0)
        motion_norm = torch.nan_to_num(motion_norm, nan=0.0, posinf=2.0, neginf=0.0)


        edge_attr = torch.cat([dir_norm, radius_norm, motion_norm.unsqueeze(-1)], dim=-1)

        return edge_attr



class AMSoftmaxHead(nn.Module):

    def __init__(self, feat_dim, num_classes, s=30, m=0.35):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m


        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):

        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)


        cos_theta = F.linear(features, weight)

        if self.training and labels is not None:

            cos_theta_m = cos_theta.clone()
            cos_theta_m[range(len(labels)), labels] -= self.m
            logits = self.s * cos_theta_m
        else:

            logits = self.s * cos_theta

        return logits


class CenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, alpha=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha


        self.register_buffer('centers', torch.zeros(num_classes, feat_dim))

    def forward(self, features, labels):

        centers_batch = self.centers[labels]
        loss = F.mse_loss(features, centers_batch)


        with torch.no_grad():
            for i in range(self.num_classes):
                mask = (labels == i)
                if mask.sum() > 0:
                    self.centers[i] = self.alpha * self.centers[i] + (1 - self.alpha) * features[mask].mean(0)

        return loss



class LossCollector:

    _losses = []

    @classmethod
    def clear(cls):
        cls._losses = []

    @classmethod
    def add(cls, loss):
        cls._losses.append(loss)

    @classmethod
    def get_total(cls):
        if not cls._losses:
            return None
        return sum(cls._losses)


class MotionEdgeNoiseFilter(nn.Module):

    def __init__(self, edge_dim=5, hidden_dim=32):
        super().__init__()




        self.edge_confidence = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )




    def forward(self, edge_attr, edge_index, pos, return_loss=False):





        edge_conf = self.edge_confidence(edge_attr)
        edge_conf = edge_conf.clamp(1e-3, 1.0)



        edge_attr_4d = edge_attr[:, :4]
        filtered_edge_attr = edge_attr_4d * edge_conf


        if return_loss and self.training:
            consistency_loss = self._compute_consistency_loss(
                edge_attr, edge_conf, edge_index, pos
            )
            LossCollector.add(consistency_loss)

        return filtered_edge_attr

    def _compute_consistency_loss(self, edge_attr, edge_conf, edge_index, pos):

        if edge_attr.shape[1] < 5:
            return torch.tensor(0.0, device=edge_attr.device)

        edge_conf_squeezed = edge_conf.squeeze(1)



        motion_strength = edge_attr[:, 4]
        motion_norm = motion_strength / (motion_strength.max() + 1e-6)
        motion_conf_alignment = torch.mean((edge_conf_squeezed - motion_norm.clamp(0, 1))**2)



        radius = edge_attr[:, 3]


        radius_reasonable = ((radius > 0.1) & (radius < 1.5)).float()
        radius_conf_alignment = torch.mean((edge_conf_squeezed - radius_reasonable)**2)




        direction = edge_attr[:, :3]
        direction_norm = direction.norm(dim=-1)

        direction_valid = (direction_norm > 0.9).float()
        direction_conf_alignment = torch.mean((edge_conf_squeezed - direction_valid)**2)


        conf_entropy = -(edge_conf_squeezed * torch.log(edge_conf_squeezed + 1e-8) +
                        (1 - edge_conf_squeezed) * torch.log(1 - edge_conf_squeezed + 1e-8))
        target_entropy = 0.5
        entropy_loss = torch.abs(conf_entropy.mean() - target_entropy)


        total_loss = (0.4 * motion_conf_alignment +
                     0.2 * radius_conf_alignment +
                     0.2 * direction_conf_alignment +
                     0.2 * entropy_loss)

        return total_loss


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.2):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
            return x * mask / keep_prob
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, edge_dim=4, drop_prob=0.2, norm_params=None, level=0):

        super(ResidualBlock, self).__init__()

        self.edge_feature_generator = EdgeFeatureGenerator(norm_params, level)


        self.noise_filter = MotionEdgeNoiseFilter(edge_dim=5, hidden_dim=32)


        self.left_conv1 = GMMConv(in_channel, out_channel, dim=edge_dim, kernel_size=5)
        self.left_bn1 = torch.nn.BatchNorm1d(out_channel)

        self.left_conv2 = GMMConv(out_channel, out_channel, dim=edge_dim, kernel_size=5)
        self.left_bn2 = torch.nn.BatchNorm1d(out_channel)

        self.shortcut_conv = GMMConv(in_channel, out_channel, dim=edge_dim, kernel_size=1)
        self.shortcut_bn = torch.nn.BatchNorm1d(out_channel)
        self.drop_path = DropPath(drop_prob)

    def forward(self, data):
        x = data.x
        pos = data.pos
        edge_index = data.edge_index


        edge_attr_5d = self.edge_feature_generator(pos, edge_index)




        edge_attr = self.noise_filter(edge_attr_5d, edge_index, pos, return_loss=self.training)


        out = F.elu(self.left_bn1(self.left_conv1(x, edge_index, edge_attr)))
        out = self.left_bn2(self.left_conv2(out, edge_index, edge_attr))
        out = F.elu(out)
        shortcut = self.shortcut_bn(self.shortcut_conv(x, edge_index, edge_attr))
        shortcut = self.drop_path(shortcut)
        data.x = F.elu(out + shortcut)
        return data


class AllGraphBlock(torch.nn.Module):
    def __init__(self, out_dim, edge_dim=4, norm_params=None):

        super(AllGraphBlock, self).__init__()

        self.edge_feature_generator = EdgeFeatureGenerator(norm_params, level=0)


        self.noise_filter = MotionEdgeNoiseFilter(edge_dim=5, hidden_dim=32)


        self.conv1 = GMMConv(3, out_dim, dim=edge_dim, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(out_dim)

        self.block1 = ResidualBlock(out_dim, out_dim * 2, edge_dim=edge_dim, drop_prob=0.0, norm_params=norm_params, level=1)
        self.block2 = ResidualBlock(out_dim * 2, out_dim * 4, edge_dim=edge_dim, drop_prob=0.0, norm_params=norm_params, level=2)
        self.block3 = ResidualBlock(out_dim * 4, out_dim * 8, edge_dim=edge_dim, drop_prob=0.0, norm_params=norm_params, level=3)

    def step(self, graph):

        edge_attr_5d = self.edge_feature_generator(graph.pos, graph.edge_index)



        edge_attr = self.noise_filter(edge_attr_5d, graph.edge_index, graph.pos, return_loss=self.training)


        graph.x = F.elu(self.bn1(self.conv1(graph.x, graph.edge_index, edge_attr)))
        cluster = voxel_grid(graph.pos, batch=graph.batch, size=4)
        graph = max_pool(cluster, graph, transform=None)

        graph = self.block1(graph)
        cluster = voxel_grid(graph.pos, batch=graph.batch, size=6)
        graph = max_pool(cluster, graph, transform=None)

        graph = self.block2(graph)
        cluster = voxel_grid(graph.pos, batch=graph.batch, size=24)
        graph = max_pool(cluster, graph, transform=None)

        graph = self.block3(graph)
        cluster = voxel_grid(graph.pos, batch=graph.batch, size=64)
        graph = max_pool(cluster, graph, transform=None)
        x = global_mean_pool(graph.x, batch=graph.batch)
        return x.unsqueeze(1)

    def forward(self, graphs):
        x = self.step(graphs)
        x = x.view(-1, 5, 256)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, opt_dim, heads, dropout, att_dropout, **args):
        super(AttentionBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            opt_dim,
            heads,
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
        z_att, _ = self.attention(z, x, x, key_padding_mask=mask, attn_mask=q_mask)
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
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.dropout = nn.Dropout(p=0.3)



        self.am_softmax = AMSoftmaxHead(hid_dim, out_dim, s=18, m=0.27)
        self.center_loss = CenterLoss(out_dim, hid_dim, alpha=0.35)

    def forward(self, x):

        x = self.linear1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        features = self.linear2(x)
        return features

    def loss(self, features, labels):


        am_logits = self.am_softmax(features, labels)
        am_loss = F.cross_entropy(am_logits, labels)


        center_loss = self.center_loss(features, labels)


        total_loss = am_loss + 0.008 * center_loss
        return total_loss, am_loss, center_loss


class GraphAttention(nn.Module):
    def __init__(self, num_attention, out_dim, heads, num_class, norm_params=None):
        super(GraphAttention, self).__init__()
        self.graph_model = AllGraphBlock(out_dim=out_dim, norm_params=norm_params)



        self.cls = ClfBlok(int_dim=out_dim * 8 * 2, hid_dim=64, out_dim=num_class)
        self.str = StrModule(num_attention_heads=heads, input_size=out_dim * 8, hidden_size=out_dim * 8)


    def forward(self, graphs):
        x = self.graph_model(graphs)
        x = self.str.forward(x)



        features = self.cls(x)
        return features


class AttModel(nn.Module):
    def __init__(self, modulelist):
        super(AttModel, self).__init__()
        self.model = modulelist

    def forward(self, x):
        for module in self.model:
            x = module(x, x)
        return x.mean(dim=1)


class GraphAttention2(nn.Module):
    def __init__(self, num_attention, out_dim, heads, num_class, norm_params=None):
        super(GraphAttention2, self).__init__()
        self.graph_model = AllGraphBlock(out_dim=out_dim, norm_params=norm_params)
        self.attentions = AttModel(nn.ModuleList([
            AttentionBlock(opt_dim=out_dim * 8, heads=heads, dropout=0.5, att_dropout=0) for _ in
            range(num_attention)]))
        self.cls = ClfBlok(int_dim=out_dim * 8, hid_dim=64, out_dim=num_class)


    def forward(self, graphs):
        x = self.graph_model(graphs)
        x = self.attentions(x)
        features = self.cls(x)
        return features


class PlGraphAttention(LightningModule):
    def __init__(self, Config, num_attention, learning_rate, f_graph_dim, heads):
        super(PlGraphAttention, self).__init__()
        self.save_hyperparameters()

        norm_params = {'W': 128.0, 'H': 128.0}
        self.model = GraphAttention(num_attention=num_attention, out_dim=f_graph_dim, heads=heads,
                                    num_class=Config.num_class, norm_params=norm_params)
        self.accuracy = Accuracy(num_classes=Config.num_class, task='multiclass')
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

        LossCollector.clear()


        features = self(batch)
        y = batch.y[0::self.Config.split_graph_num]


        main_loss, am_loss, center_loss = self.model.cls.loss(features, y)


        consistency_loss_total = LossCollector.get_total()
        if consistency_loss_total is None:
            consistency_loss = torch.tensor(0.0, device=features.device)
        else:
            consistency_loss = consistency_loss_total


        total_loss = main_loss + 0.1 * consistency_loss


        with torch.no_grad():
            am_logits = self.model.cls.am_softmax(features, y)
            preds = F.softmax(am_logits, dim=1)

        return {
            "loss": total_loss,
            "preds": preds,
            "y": y,
            "main_loss": main_loss,
            "consistency_loss": consistency_loss,
            "am_loss": am_loss,
            "center_loss": center_loss,
        }

    def training_step_end(self, outputs):
        train_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("train_acc", train_acc, prog_bar=True, batch_size=self.Config.batch_size)
        self.log("train_loss", outputs['loss'], prog_bar=True, batch_size=self.Config.batch_size)

        if 'main_loss' in outputs:
            self.log("train_main_loss", outputs['main_loss'].mean(), prog_bar=False, batch_size=self.Config.batch_size)
        if 'consistency_loss' in outputs:
            self.log("train_consistency_loss", outputs['consistency_loss'].mean(), prog_bar=False, batch_size=self.Config.batch_size)
        if 'am_loss' in outputs:
            self.log("train_am_loss", outputs['am_loss'].mean(), prog_bar=False, batch_size=self.Config.batch_size)
        if 'center_loss' in outputs:
            self.log("train_center_loss", outputs['center_loss'].mean(), prog_bar=False, batch_size=self.Config.batch_size)
        return {"loss": outputs["loss"].mean()}

    def validation_step(self, batch, batch_idex):

        features = self(batch)
        y = batch.y[0::self.Config.split_graph_num]
        loss, am_loss, center_loss = self.model.cls.loss(features, y)

        with torch.no_grad():
            am_logits = self.model.cls.am_softmax(features, y)
            preds = F.softmax(am_logits, dim=1)

        return {"loss": loss, "preds": preds, "y": y,
                "am_loss": am_loss, "center_loss": center_loss}

    def validation_step_end(self, outputs):
        val_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("val_loss", outputs["loss"].mean(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.batch_size)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.Config.batch_size)

        if 'am_loss' in outputs:
            self.log("val_am_loss", outputs["am_loss"].mean(), prog_bar=False, on_epoch=True, on_step=False,
                     batch_size=self.Config.batch_size)
        if 'center_loss' in outputs:
            self.log("val_center_loss", outputs["center_loss"].mean(), prog_bar=False, on_epoch=True, on_step=False,
                     batch_size=self.Config.batch_size)

    def test_step(self, batch, batch_idx):
        features = self(batch)
        y = batch.y[0::self.Config.split_graph_num]
        loss, am_loss, center_loss = self.model.cls.loss(features, y)

        with torch.no_grad():
            am_logits = self.model.cls.am_softmax(features, y)
            preds = F.softmax(am_logits, dim=1)

        return {"loss": loss, "preds": preds, "y": y,
                "am_loss": am_loss, "center_loss": center_loss}

    def test_step_end(self, outputs):
        val_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("val_loss", outputs["loss"].mean(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.batch_size)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.Config.batch_size)
        if 'am_loss' in outputs:
            self.log("test_am_loss", outputs["am_loss"].mean(), prog_bar=False, on_epoch=True, on_step=False,
                     batch_size=self.Config.batch_size)
        if 'center_loss' in outputs:
            self.log("test_center_loss", outputs["center_loss"].mean(), prog_bar=False, on_epoch=True, on_step=False,
                     batch_size=self.Config.batch_size)


class PlGraphAttention_casia(LightningModule):
    def __init__(self, Config, num_attention, learning_rate, f_graph_dim, heads):
        super(PlGraphAttention_casia, self).__init__()
        self.save_hyperparameters()

        norm_params = {'W': 128.0, 'H': 128.0}
        self.model = GraphAttention2(num_attention=num_attention, out_dim=f_graph_dim, heads=heads,
                                     num_class=Config.step1_num_class, norm_params=norm_params)
        self.accuracy = Accuracy(num_classes=Config.step1_num_class, task='multiclass')
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

        LossCollector.clear()


        features = self(batch)

        y = batch.y[0::self.Config.max_num] - 1


        main_loss, am_loss, center_loss = self.model.cls.loss(features, y)


        consistency_loss_total = LossCollector.get_total()
        LossCollector.clear()
        if consistency_loss_total is None:
            consistency_loss = torch.tensor(0.0, device=features.device)
        else:
            consistency_loss = consistency_loss_total


        total_loss = main_loss + 0.1 * consistency_loss


        with torch.no_grad():
            am_logits = self.model.cls.am_softmax(features, labels=None)
            preds = F.softmax(am_logits, dim=1)

        return {
            "loss": total_loss,
            "preds": preds,
            "y": y,
            "main_loss": main_loss,
            "consistency_loss": consistency_loss,
            "am_loss": am_loss,
            "center_loss": center_loss,
        }

    def training_step_end(self, outputs):
        train_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("train_acc", train_acc, prog_bar=True, batch_size=self.Config.step1_batch_size)
        self.log("train_loss", outputs['loss'], prog_bar=True, batch_size=self.Config.step1_batch_size)

        if 'main_loss' in outputs:
            self.log("train_main_loss", outputs['main_loss'].mean(), prog_bar=False, batch_size=self.Config.step1_batch_size)
        if 'consistency_loss' in outputs:
            self.log("train_consistency_loss", outputs['consistency_loss'].mean(), prog_bar=False, batch_size=self.Config.step1_batch_size)
        if 'am_loss' in outputs:
            self.log("train_am_loss", outputs['am_loss'].mean(), prog_bar=False, batch_size=self.Config.step1_batch_size)
        if 'center_loss' in outputs:
            self.log("train_center_loss", outputs['center_loss'].mean(), prog_bar=False, batch_size=self.Config.step1_batch_size)
        return {"loss": outputs["loss"].mean()}

    def validation_step(self, batch, batch_idex):

        features = self(batch)

        y = batch.y[0::self.Config.max_num] - 1
        loss, am_loss, center_loss = self.model.cls.loss(features, y)

        with torch.no_grad():
            am_logits = self.model.cls.am_softmax(features, labels=None)
            preds = F.softmax(am_logits, dim=1)
        torch.cuda.empty_cache()

        return {"loss": loss, "preds": preds, "y": y,
                "am_loss": am_loss, "center_loss": center_loss}

    def validation_step_end(self, outputs):
        val_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("val_loss", outputs["loss"].mean(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step1_batch_size)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step1_batch_size)

        if 'am_loss' in outputs:
            self.log("val_am_loss", outputs["am_loss"].mean(), prog_bar=False, on_epoch=True, on_step=False,
                     batch_size=self.Config.step1_batch_size)
        if 'center_loss' in outputs:
            self.log("val_center_loss", outputs["center_loss"].mean(), prog_bar=False, on_epoch=True, on_step=False,
                     batch_size=self.Config.step1_batch_size)

    def test_step(self, batch, batch_idx):
        features = self(batch)

        y = batch.y[0::self.Config.max_num] - 1
        loss, am_loss, center_loss = self.model.cls.loss(features, y)

        with torch.no_grad():
            am_logits = self.model.cls.am_softmax(features, labels=None)
            preds = F.softmax(am_logits, dim=1)

        return {"loss": loss, "preds": preds, "y": y,
                "am_loss": am_loss, "center_loss": center_loss}

    def test_step_end(self, outputs):
        val_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("val_loss", outputs["loss"].mean(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step1_batch_size)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step1_batch_size)
        if 'am_loss' in outputs:
            self.log("test_am_loss", outputs["am_loss"].mean(), prog_bar=False, on_epoch=True, on_step=False,
                     batch_size=self.Config.step1_batch_size)
        if 'center_loss' in outputs:
            self.log("test_center_loss", outputs["center_loss"].mean(), prog_bar=False, on_epoch=True, on_step=False,
                     batch_size=self.Config.step1_batch_size)


class PlGraphAttention_casia_2(LightningModule):
    def __init__(self, Config, pretrained_model, learning_rate, Gout_dim):
        super(PlGraphAttention_casia_2, self).__init__()
        self.save_hyperparameters(ignore=['pretrained_model'])
        self.cls = ClfBlok(int_dim=Gout_dim * 8, hid_dim=64, out_dim=Config.step2_num_class)
        self.pretrained_model = list(list(pretrained_model.children())[0].children())
        self.pretrained_model[2] = self.cls
        self.model = nn.Sequential(*self.pretrained_model)
        self.accuracy = Accuracy(num_classes=Config.step2_num_class, task='multiclass')
        self.Config = Config

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[20, 40, 60, 80, 100, 110, 120, 130, 140],
                                                      gamma=0.7)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict

    def forward(self, x):



        features = self.model(x)
        return features

    def training_step(self, batch, batch_idx):

        LossCollector.clear()


        features = self(batch)

        y = batch.y[0::self.Config.max_num] - 75


        main_loss, am_loss, center_loss = self.cls.loss(features, y)


        consistency_loss_total = LossCollector.get_total()
        LossCollector.clear()
        if consistency_loss_total is None:
            consistency_loss = torch.tensor(0.0, device=features.device)
        else:
            consistency_loss = consistency_loss_total


        total_loss = main_loss + 0.1 * consistency_loss


        with torch.no_grad():
            am_logits = self.cls.am_softmax(features, labels=None)
            preds = F.softmax(am_logits, dim=1)

        return {
            "loss": total_loss,
            "preds": preds,
            "y": y,
            "main_loss": main_loss,
            "consistency_loss": consistency_loss,
            "am_loss": am_loss,
            "center_loss": center_loss,
        }

    def training_step_end(self, outputs):
        train_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("train_acc", train_acc, prog_bar=True, batch_size=self.Config.step2_batch_size)
        self.log("train_loss", outputs['loss'], prog_bar=True, batch_size=self.Config.step2_batch_size)

        if 'main_loss' in outputs:
            self.log("train_main_loss", outputs['main_loss'].mean(), prog_bar=False, batch_size=self.Config.step2_batch_size)
        if 'consistency_loss' in outputs:
            self.log("train_consistency_loss", outputs['consistency_loss'].mean(), prog_bar=False, batch_size=self.Config.step2_batch_size)
        if 'am_loss' in outputs:
            self.log("train_am_loss", outputs['am_loss'].mean(), prog_bar=False, batch_size=self.Config.step2_batch_size)
        if 'center_loss' in outputs:
            self.log("train_center_loss", outputs['center_loss'].mean(), prog_bar=False, batch_size=self.Config.step2_batch_size)
        return {"loss": outputs["loss"].mean()}

    def validation_step(self, batch, batch_idex):

        features = self(batch)

        y = batch.y[0::self.Config.max_num] - 75
        loss, am_loss, center_loss = self.cls.loss(features, y)

        with torch.no_grad():
            am_logits = self.cls.am_softmax(features, labels=None)
            preds = F.softmax(am_logits, dim=1)
        torch.cuda.empty_cache()

        return {"loss": loss, "preds": preds, "y": y,
                "am_loss": am_loss, "center_loss": center_loss}

    def validation_step_end(self, outputs):
        val_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("val_loss", outputs["loss"].mean(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step2_batch_size)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step2_batch_size)

        if 'am_loss' in outputs:
            self.log("val_am_loss", outputs["am_loss"].mean(), prog_bar=False, on_epoch=True, on_step=False,
                     batch_size=self.Config.step2_batch_size)
        if 'center_loss' in outputs:
            self.log("val_center_loss", outputs["center_loss"].mean(), prog_bar=False, on_epoch=True, on_step=False,
                     batch_size=self.Config.step2_batch_size)

    def test_step(self, batch, batch_idx):
        features = self(batch)

        y = batch.y[0::self.Config.max_num] - 75
        loss, am_loss, center_loss = self.cls.loss(features, y)

        with torch.no_grad():
            am_logits = self.cls.am_softmax(features, labels=None)
            preds = F.softmax(am_logits, dim=1)

        return {"loss": loss, "preds": preds, "y": y,
                "am_loss": am_loss, "center_loss": center_loss}

    def test_step_end(self, outputs):
        val_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("val_loss", outputs["loss"].mean(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step2_batch_size)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=self.Config.step2_batch_size)
        if 'am_loss' in outputs:
            self.log("test_am_loss", outputs["am_loss"].mean(), prog_bar=False, on_epoch=True, on_step=False,
                     batch_size=self.Config.step2_batch_size)
        if 'center_loss' in outputs:
            self.log("test_center_loss", outputs["center_loss"].mean(), prog_bar=False, on_epoch=True, on_step=False,
                     batch_size=self.Config.step2_batch_size)



if __name__ == '__main__':



    x = torch.rand((32, 20, 20))
    attention = StrModule(2, 20, 20)
    result = attention.forward(x)
