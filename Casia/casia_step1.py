
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import training_utils
from casia_config import Config
from dataset_scripts.casia_downsample import Downsample
from dataset_scripts.casia_tograph import ToGraph
from utils.graphtransformer_dataset import PlData_casia
from model.net import PlGraphAttention_casia

# %%
downsample = Downsample(Config)
downsample.exe()
tograph = ToGraph(Config)
tograph.exe()
path_model = training_utils.create_model_folder(f'../pretrained/{Config.step1_datasetname}', Config.data_format)
step1_train_dataset = PlData_casia(Config, step='1')
model = PlGraphAttention_casia(Config, num_attention=1, learning_rate=0.001, f_graph_dim=32, heads=2)
datas = next(iter(step1_train_dataset.train_dataloader()))
out = model(datas)
pl.seed_everything(1234)
callback = [
    EarlyStopping(
        **{'monitor': 'val_acc', 'min_delta': 0.000001, 'patience': 50, 'check_finite': True, 'mode': 'max'}),
    ModelCheckpoint(**{'monitor': 'val_acc', 'mode': 'max', 'filename': '{epoch}-{val_loss:.5f}-{val_acc:.5f}',
                       'dirpath': os.path.join(path_model, 'weights'),
                       'save_last': False,
                       'save_weights_only': False, 'save_top_k': 3, 'every_n_epochs': 1}),
    pl.callbacks.StochasticWeightAveraging(1e-2),
    LearningRateMonitor(**{'logging_interval': 'epoch'})]
logger = TensorBoardLogger('../log_dir', name='gt')
trainer = pl.Trainer(gpus=1, max_epochs=200, callbacks=callback, logger=logger,
                     check_val_every_n_epoch=2,
                     accumulate_grad_batches=2)
trainer.fit(model, step1_train_dataset)
