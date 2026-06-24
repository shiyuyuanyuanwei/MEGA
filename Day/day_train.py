import os
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from utils import training_utils
from day_config import Config
from dataset_scripts.day_downsample import Downsample
from dataset_scripts.tograph import ToGraph
from utils.graphtransformer_dataset import PlData
from model.net import PlGraphAttention
# %%
downsample = Downsample(Config)
downsample.exe()
tograph = ToGraph(Config)
tograph.exe()
path_model = training_utils.create_model_folder(f'../pretrained/{Config.datasetname}', Config.data_format)
train_dataset = PlData(Config)
model = PlGraphAttention(Config, num_attention=1, learning_rate=0.001, f_graph_dim=32, heads=1)
datas = next(iter(train_dataset.train_dataloader()))
out = model(datas)

pl.seed_everything(1234)
callback = [
    EarlyStopping(
        **{'monitor': 'val_acc', 'min_delta': 0.000001, 'patience': 60, 'check_finite': True, 'mode': 'max'}),
    ModelCheckpoint(**{'monitor': 'val_acc', 'mode': 'max', 'filename': '{epoch}-{val_loss:.5f}-{val_acc:.5f}',
                       'dirpath': os.path.join(path_model, 'weights'),
                       'save_last': False,
                       'save_weights_only': False, 'save_top_k': 1, 'every_n_epochs': 1}),
    pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2, swa_epoch_start=0.95),
    LearningRateMonitor(**{'logging_interval': 'epoch'})]
logger = TensorBoardLogger('../log_dir', name='gt')
trainer = pl.Trainer(accelerator='gpu', max_epochs=330, callbacks=callback, logger=logger,
                     check_val_every_n_epoch=1,
                     accumulate_grad_batches=2)
trainer.fit(model, train_dataset)
