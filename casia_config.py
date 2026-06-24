
import os


class Config():
    angle = '000'
    step1_datasetname = 'CASIA_B_' + angle + '_1'
    step2_datasetname = 'CASIA_B_' + angle + '_2'
    curPath = os.path.abspath(__file__)
    rootPath = os.path.split(curPath)[0]
    every_k_points = 2
    nb_neighbors = 10
    std = 0.1
    split_graph_num = 5
    max_num = 5
    step1_num_class = 74
    step2_num_class = 50
    data_format = '/{}_{}_{}'.format(every_k_points, split_graph_num, max_num)
    data_dir = os.path.join(rootPath, 'data/CASIA_B_2020/' + angle)
    stpe1_dir = os.path.join(data_dir, 'step1')
    stpe2_dir = os.path.join(data_dir, 'step2')
    step1_downsample_dir = os.path.join(stpe1_dir, 'downsample_{}'.format(every_k_points))
    step2_downsample_dir = os.path.join(stpe2_dir, 'downsample_{}'.format(every_k_points))
    step1_origin_data_dir = os.path.join(stpe1_dir, 'origin')
    step2_origin_data_dir = os.path.join(stpe2_dir, 'origin')
    step1_graph_dir = os.path.join(stpe1_dir, 'graph_{}_{}_{}'.format(every_k_points, split_graph_num, max_num))
    step2_graph_dir = os.path.join(stpe2_dir, 'graph_{}_{}_{}'.format(every_k_points, split_graph_num, max_num))
    step1_graph_train_dir = os.path.join(step1_graph_dir, 'train')
    step2_graph_train_dir = os.path.join(step2_graph_dir, 'train')
    step1_graph_test_dir = os.path.join(step1_graph_dir, 'train')
    step2_graph_test_dir = os.path.join(step2_graph_dir, 'test')
    step1_batch_size = 8
    step2_batch_size = 8
    num_workers = 3