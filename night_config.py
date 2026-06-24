
import os


class Config():
    datasetname = 'Night'
    curPath = os.path.abspath(__file__)
    rootPath = os.path.split(curPath)[0]
    every_k_points = 10
    nb_neighbors = 500
    std = 0.001
    split_graph_num = 10
    max_num = 5
    num_class = 20
    data_format = '/{}_{}_{}'.format(every_k_points, split_graph_num, max_num)
    downsample_dir = os.path.join(rootPath, 'data/DVS128-Gait-Night/downsample_{}'.format(every_k_points))
    origin_data_dir = os.path.join(rootPath, 'data/DVS128-Gait-Night/origin')
    data_dir = os.path.join(rootPath, 'data/DVS128-Gait-Night')
    graph_dir = os.path.join(data_dir, 'graph_{}_{}_{}'.format(every_k_points, split_graph_num, max_num))
    graph_train_dir = os.path.join(graph_dir, 'train')
    graph_test_dir = os.path.join(graph_dir, 'test')
    batch_size = 6
    num_workers = 12
