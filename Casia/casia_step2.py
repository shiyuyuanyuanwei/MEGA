
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import training_utils
from casia_config import Config
from utils.graphtransformer_dataset import PlData_casia
from model.net import PlGraphAttention_casia, PlGraphAttention_casia_2

# %%
model = PlGraphAttention_casia(Config, num_attention=1, learning_rate=0.001, f_graph_dim=32, heads=2)
clone = model.load_from_checkpoint(
    '../pretrained/CASIA_B_036_1/2_5_5/0114_2151_model_2/weights/epoch=159-val_loss=0.00145-val_acc=1.00000.ckpt')
path_model = training_utils.create_model_folder(f'../pretrained/{Config.step2_datasetname}', Config.data_format)
step2_train_dataset = PlData_casia(Config, step='2')
model2 = PlGraphAttention_casia_2(Config, pretrained_model=clone, learning_rate=0.001, Gout_dim=32)
pl.seed_everything(1)
datas = next(iter(step2_train_dataset.train_dataloader()))
out = model2(datas)
callback = [
    EarlyStopping(
        **{'monitor': 'val_acc', 'min_delta': 0.000001, 'patience': 50, 'check_finite': True, 'mode': 'max'}),
    ModelCheckpoint(**{'monitor': 'val_acc', 'mode': 'max', 'filename': '{epoch}-{val_loss:.5f}-{val_acc:.5f}',
                       'dirpath': os.path.join(path_model, 'weights'),
                       'save_last': False,
                       'save_weights_only': False, 'save_top_k': 1, 'every_n_epochs': 1}),
    pl.callbacks.StochasticWeightAveraging(1e-2),
    LearningRateMonitor(**{'logging_interval': 'epoch'})]
logger = TensorBoardLogger('../log_dir', name='gt')
trainer = pl.Trainer(gpus=1, max_epochs=200, callbacks=callback, logger=logger,
                     check_val_every_n_epoch=2,
                     accumulate_grad_batches=1)
trainer.fit(model2, step2_train_dataset)
