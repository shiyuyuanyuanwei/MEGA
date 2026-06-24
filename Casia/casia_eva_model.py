
import pytorch_lightning as pl
import torch

from casia_config import Config
from utils.graphtransformer_dataset import PlData_casia
from model.net import PlGraphAttention_casia_2, PlGraphAttention_casia

model = PlGraphAttention_casia(Config, num_attention=1, learning_rate=0.001, f_graph_dim=32, heads=2)
clone = model.load_from_checkpoint(
    '../pretrained/CASIA_B_108_1/2_5_5/0111_1554_model_0/weights/epoch=113-val_loss=0.01407-val_acc=1.00000.ckpt')
step2_train_dataset = PlData_casia(Config, step='2')
trainer = pl.Trainer(gpus=1)
checkpoint = torch.load(
    "../pretrained/CASIA_B_108_2/2_5_5/0111_1856_model_1/weights/epoch=129-val_loss=0.16119-val_acc=0.98000.ckpt")
hyper_parameters = checkpoint["hyper_parameters"]
t = PlGraphAttention_casia_2(pretrained_model=clone, **hyper_parameters)
model_weights = checkpoint["state_dict"]
t.load_state_dict(model_weights)
result = trainer.test(t, step2_train_dataset.val_dataloader())
