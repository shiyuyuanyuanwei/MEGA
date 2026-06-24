# author:c19h
# datetime:2022/12/7 13:35
# author:c19h
# datetime:2022/12/1 18:34
# -- coding: utf-8 --**
# the dataset class for EV-Gait-3DGraph model
import glob
import os
import os.path as osp
import pickle
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data
import torch_geometric.transforms as T
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch_geometric.data import Batch

class MyDataset(Dataset):
    def __init__(self, root, split_graph_num, max_num, validation, process_folder='processed', transform=None,
                 pre_transform=None):
        self.split_graph_num = split_graph_num
        self.max_num = max_num
        self.validation = validation
        self.process_folder = process_folder
        super(MyDataset, self).__init__(root, transform, pre_transform)

    # return file list of self.raw_dir
    @property
    def processed_dir(self):
        return osp.join(self.root, self.process_folder)

    @property
    def raw_file_names(self):
        all_filenames = glob.glob(os.path.join(self.raw_dir, "*.pkl"))
        # get all file names
        file_names = [f.split(os.sep)[-1] for f in all_filenames]
        return file_names

    # get all file names in  self.processed_dir
    @property
    def processed_file_names(self):
        filenames = glob.glob(os.path.join(self.raw_dir, "*.pkl"))
        file = [f.split(os.sep)[-1] for f in filenames]
        saved_file = [f.replace(".pkl", ".pt") for f in file]
        return saved_file

    def __len__(self):
        return len(self.processed_file_names)

    def len(self):
        return len(self.processed_file_names)

    def download(self):
        pass

    # convert the mat files of self.raw_dir to torch_geometric.Data format, save the result files in self.processed_dir
    # this method will only execute one time at the first running.
    def process(self):
        for raw_path in tqdm(self.raw_paths):
            with open(raw_path, 'rb') as f:
                file = pickle.load(f)
            graphs = file['graphs']
            datas = []
            for graph in graphs:
                feature, position, edge_index, label = graph
                feature = torch.tensor(feature, dtype=torch.float)
                position = torch.tensor(position, dtype=torch.float32)
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                label = torch.tensor(label, dtype=torch.long)
                data = Data(x=feature, edge_index=edge_index, pos=position, y=label.item())
                datas.append(data)
            saved_name = raw_path.split(os.sep)[-1].replace(".pkl", ".pt")
            torch.save(datas, osp.join(self.processed_dir, saved_name))

    def __getitem__(self, idx):
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            datas = self.get(self.indices()[idx])
            if self.transform is None:
                datas = datas
            else:
                events = datas
                for event in events:
                    self.transform(event)
            return datas

        else:
            return self.index_select(idx)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_paths[idx]))
        if self.split_graph_num != self.max_num:
            if not self.validation:
                init = np.random.randint(self.split_graph_num - self.max_num)
                output = data[init:init + self.max_num]
            else:
                init = (self.split_graph_num - self.max_num) // 2
                output = data[init:init + self.max_num]
        else:
            output = data
        return output


class PlData(LightningDataModule):
    def __init__(self, Config):
        super().__init__()
        self.Config = Config
        self.train_data_aug = T.Compose(
            [T.Cartesian(cat=False), T.RandomScale([0.95, 0.999]), T.RandomJitter(0.01)])
        self.test_data_aug = T.Compose(
            [T.Cartesian(cat=False), T.RandomScale([0.95, 0.999])])

    def train_dataloader(self):
        dt = MyDataset(self.Config.graph_train_dir, validation=False, split_graph_num=self.Config.split_graph_num,
                       max_num=self.Config.max_num,
                       process_folder='processed_all',
                       transform=self.train_data_aug)
        dl = MyDataLoader(dt, batch_size=self.Config.batch_size, collate_fn=collate_sequences, shuffle=True, num_workers=self.Config.num_workers,
                        pin_memory=True)
        return dl

    def val_dataloader(self):
        dt = MyDataset(self.Config.graph_test_dir, validation=True, split_graph_num=self.Config.split_graph_num,
                       max_num=self.Config.max_num,
                       process_folder='processed_all',
                       transform=self.test_data_aug)
        dl = MyDataLoader(dt, batch_size=self.Config.batch_size, collate_fn=collate_sequences, shuffle=False, num_workers=self.Config.num_workers,
                        pin_memory=True)
        return dl


class PlData_casia(LightningDataModule):
    def __init__(self, Config, step):
        super().__init__()
        self.Config = Config
        self.step = step
        self.train_data_aug = T.Compose(
            [T.Cartesian(cat=False), T.RandomScale([0.95, 0.999]), T.RandomJitter(0.01)])
        self.test_data_aug = T.Compose(
            [T.Cartesian(cat=False), T.RandomScale([0.95, 0.999])])

    def train_dataloader(self):
        if self.step == '1':
            dt = MyDataset(self.Config.step1_graph_train_dir, validation=False,
                           split_graph_num=self.Config.split_graph_num,
                           max_num=self.Config.max_num,
                           process_folder='processed_all',
                           transform=self.train_data_aug)
            dl = MyDataLoader(dt, batch_size=self.Config.step1_batch_size, shuffle=True, collate_fn=collate_sequences,
                            num_workers=self.Config.num_workers,
                            pin_memory=True)
        else:
            dt = MyDataset(self.Config.step2_graph_train_dir, validation=False,
                           split_graph_num=self.Config.split_graph_num,
                           max_num=self.Config.max_num,
                           process_folder='processed_all',
                           transform=self.train_data_aug)
            dl = MyDataLoader(dt, batch_size=self.Config.step2_batch_size, shuffle=True,
                            num_workers=self.Config.num_workers,collate_fn=collate_sequences,
                            pin_memory=True)
        return dl

    def val_dataloader(self):
        if self.step == '1':
            dt = MyDataset(self.Config.step1_graph_test_dir, validation=True,
                           split_graph_num=self.Config.split_graph_num,
                           max_num=self.Config.max_num,
                           process_folder='processed_all',
                           transform=self.test_data_aug)
            dl = MyDataLoader(dt, batch_size=self.Config.step1_batch_size,collate_fn=collate_sequences, shuffle=False, num_workers=self.Config.num_workers,
                            pin_memory=True)
        else:
            dt = MyDataset(self.Config.step2_graph_test_dir, validation=True,
                           split_graph_num=self.Config.split_graph_num,
                           max_num=self.Config.max_num,
                           process_folder='processed_all',
                           transform=self.test_data_aug)
            dl = MyDataLoader(dt, batch_size=self.Config.step2_batch_size,collate_fn=collate_sequences, shuffle=False, num_workers=self.Config.num_workers,
                        pin_memory=True)
        return dl


def collate_sequences(batch):
    data_list = [data for sequence in batch for data in sequence]
    out = Batch.from_data_list(data_list)
    return out

class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=None, exclude_keys=None, collate_fn=None, **kwargs):
        # 使用自定义的 collate_fn，如果提供的话
        if collate_fn is not None:
            kwargs['collate_fn'] = collate_fn
        else:
            kwargs['collate_fn'] = collate_fn(follow_batch, exclude_keys)
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, **kwargs)

if __name__ == '__main__':
    pass
