
import pytorch_lightning as pl
import torch
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))  
rootPath = os.path.split(curPath)[0]                       
sys.path.append(rootPath)

from day_config import Config
from utils.graphtransformer_dataset import PlData
from model.net import PlGraphAttention

train_dataset = PlData(Config)
trainer = pl.Trainer()
checkpoint = torch.load(
    "../pretrained/Day/6_5_5/1228_0110_model_29/weights/epoch=234-val_loss=0.02302-val_acc=0.99550.ckpt",map_location=torch.device('cpu'))
hyper_parameters = checkpoint["hyper_parameters"]
t = PlGraphAttention(**hyper_parameters)
model_weights = checkpoint["state_dict"]
t.load_state_dict(model_weights)
result = trainer.test(t, train_dataset.val_dataloader())
