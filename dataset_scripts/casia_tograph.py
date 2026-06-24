# author:c19h
# datetime:2022/12/27 12:58
# author:c19h
# datetime:2022/12/16 15:42
# author:c19h
# datetime:2022/12/7 11:20
# -- coding: utf-8 --**
# convert downsample file to graph
import os

import numpy as np
import scipy.io as sio
from joblib import Parallel, delayed
from tqdm import tqdm


def calculate_edges(data, r=5):
    # threshold of radius
    d = 64
    # scaling factor to tune the difference between temporal and spatial resolution
    alpha = 1
    beta = 1
    data_size = data.shape[0]
    # max number of edges is 1000000,
    edges = np.zeros([2000000, 2])
    # get t, x,y
    points = data[:, 0:3]
    row_num = 0
    for i in range(data_size - 1):
        count = 0
        distance_matrix = points[i + 1: data_size + 1, 0:3]
        distance_matrix[:, 1:3] = distance_matrix[:, 1:3] - points[i, 1:3]
        distance_matrix[:, 0] = distance_matrix[:, 0] - points[i, 0]
        distance_matrix = np.square(distance_matrix)
        distance_matrix[:, 0] *= alpha
        distance_matrix[:, 1:3] *= beta
        # calculate the distance of each pair of events
        distance = np.sqrt(np.sum(distance_matrix, axis=1))
        index = np.where(distance <= r)
        # save the edges
        if index:
            index = index[0].tolist()
            for id in index:
                edges[row_num, 0] = i
                edges[row_num + 1, 1] = i
                edges[row_num, 1] = int(id) + i + 1
                edges[row_num + 1, 0] = int(id) + i + 1
                row_num = row_num + 2
                count = count + 1
                if count > d:
                    break
    edges = edges[~np.all(edges == 0, axis=1)]
    edges = np.transpose(edges)
    return edges


# get polarity as the feature of the node
def extract_feature(data):
    data_size = data.shape[0]
    feature = np.zeros([data_size, 2])
    for i in range(data_size):
        if data[i, 3] == 1:
            feature[i, 0] = +1
        else:
            feature[i, 0] = -1
    feature[:, 1] = data[:, 0] / 128
    return feature


def extract_position(data):
    data_size = data.shape[0]
    position = np.zeros([data_size, 3])
    for i in range(data_size):
        position[i, :] = data[i, 0:3]
    return position


def generate_graph(origin_path, target_path, label, split_graph_num):
    file = sio.loadmat(origin_path)
    total_events = file["points"]
    index = len(total_events) // split_graph_num
    events_split = np.split(total_events[:split_graph_num * index, :], split_graph_num, axis=0)
    datas = []
    for event in events_split:
        feature = extract_feature(event)
        position = extract_position(event)
        edges = calculate_edges(event, 5)
        # if the number of edges is 0 or less than 10, skip this sample
        if edges.shape[1] < 10:
            # view this file
            print(origin_path + " : " + str(edges.shape[1]))
            return

        save_data = [feature, position, edges, int(label)]
        datas.append(save_data)
    all = {'graphs': datas}
    sio.savemat(target_path, all)


class ToGraph():
    def __init__(self, Config):
        self.Config = Config

    def exe(self):
        if not os.path.exists(self.Config.step1_graph_dir) or not os.path.exists(self.Config.step2_graph_dir):
            step1_origin_path_list = []
            step1_label_list = []
            step1_target_path_list = []
            # iterate train and test
            for train_test in os.listdir(self.Config.step1_downsample_dir):
                # iterate each person
                for person in os.listdir(os.path.join(self.Config.step1_downsample_dir, train_test)):
                    # make corresponding person graph dir
                    if not os.path.exists(os.path.join(self.Config.step1_graph_dir, train_test, "raw")):
                        os.makedirs(os.path.join(self.Config.step1_graph_dir, train_test, "raw"))
                    for file in os.listdir(os.path.join(self.Config.step1_downsample_dir, train_test, person)):
                        step1_origin_path_list.append(
                            os.path.join(self.Config.step1_downsample_dir, train_test, person, file))
                        step1_label_list.append(person)
                        # faltten the graph files in one directory
                        step1_target_path_list.append(
                            os.path.join(self.Config.step1_graph_dir, train_test, "raw", person + "_" + file))

            Parallel(n_jobs=-1)(
                delayed(generate_graph)(step1_origin_path_list[i], step1_target_path_list[i], step1_label_list[i],
                                        split_graph_num=self.Config.split_graph_num) for i in
                tqdm(range(len(step1_origin_path_list))))

            step2_origin_path_list = []
            step2_label_list = []
            step2_target_path_list = []
            # iterate train and test
            for train_test in os.listdir(self.Config.step2_downsample_dir):
                # iterate each person
                for person in os.listdir(os.path.join(self.Config.step2_downsample_dir, train_test)):
                    # make corresponding person graph dir
                    if not os.path.exists(os.path.join(self.Config.step2_graph_dir, train_test, "raw")):
                        os.makedirs(os.path.join(self.Config.step2_graph_dir, train_test, "raw"))
                    for file in os.listdir(os.path.join(self.Config.step2_downsample_dir, train_test, person)):
                        step2_origin_path_list.append(
                            os.path.join(self.Config.step2_downsample_dir, train_test, person, file))
                        step2_label_list.append(person)
                        # faltten the graph files in one directory
                        step2_target_path_list.append(
                            os.path.join(self.Config.step2_graph_dir, train_test, "raw", person + "_" + file))

            Parallel(n_jobs=-1)(
                delayed(generate_graph)(step2_origin_path_list[i], step2_target_path_list[i], step2_label_list[i],
                                        split_graph_num=self.Config.split_graph_num) for i in
                tqdm(range(len(step2_origin_path_list))))

        else:
            print()


if __name__ == "__main__":
    pass
