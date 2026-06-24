
import os


class Config():
    datasetname = 'Day'
    curPath = os.path.abspath(__file__)
    rootPath = os.path.split(curPath)[0]
    every_k_points = 6
    nb_neighbors = 80
    std = 0.1
    split_graph_num = 5
    max_num = 5
    num_class = 20
    data_format = '/{}_{}_{}'.format(every_k_points, split_graph_num, max_num)
    downsample_dir = os.path.join(rootPath, 'data/DVS128-Gait-Day/downsample_{}'.format(every_k_points))
    origin_data_dir = os.path.join(rootPath, 'data/DVS128-Gait-Day/origin')
    data_dir = os.path.join(rootPath, 'data/DVS128-Gait-Day')
    graph_dir = os.path.join(data_dir, 'graph_{}_{}_{}'.format(every_k_points, split_graph_num, max_num))
    graph_train_dir = os.path.join(graph_dir, 'train')
    graph_test_dir = os.path.join(graph_dir, 'test')
    batch_size = 3
    num_workers = 0
