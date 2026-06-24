import datetime
import os
import pytorch_lightning as pl

def create_folder_if_not_exists(folder_path):
    if not os.path.isdir(folder_path): os.makedirs(folder_path)


def get_model_name(model_dir):
    folder_num = len(os.listdir(model_dir))
    return '/{}_model_{}/'.format(datetime.datetime.today().strftime('%m%d_%H%M'), folder_num)


def create_model_folder(path_results, folder_name):
    path_model = path_results + folder_name
    # Create logs and model main folder
    create_folder_if_not_exists(path_model)
    model_name = get_model_name(path_results + folder_name)
    path_model += model_name
    create_folder_if_not_exists(path_model)
    create_folder_if_not_exists(path_model + 'weights/')

    return path_model


def summary_model(model, datas):
    model_size = pl.utilities.memory.get_model_size_mb(model)
    print("model_size = {} M \n".format(model_size))
    model.example_input_array = [datas[0]]
    summary = pl.utilities.model_summary.ModelSummary(model, max_depth=-1)
    print(summary)
