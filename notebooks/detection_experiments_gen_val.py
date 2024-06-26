import os
import random
from ultralytics import YOLO
import torch
import pandas as pd
import numpy as np
import gc
import sys
#import multiprocessing
import json

# create a function to save the results structured as a dict to a json file
def save_dict_to_json(data, save_dir, file_name):
    '''
    saves a dictionary to a json file
    param:data: the dictionary to be saved
    param:save_dir: path to the directory that the json file will be saved to
    param:file_name: name of the json file that you want to save (MUST NOT INCLUDE EXTENSION)
    '''
    # convert dict to dataframe
    df = pd.DataFrame(data)

    # create save dir if doesn't already exist
    os.makedirs(save_dir, exist_ok=True)

    # create output file path
    json_file_path = os.path.join(save_dir, f'{file_name}.json')

    # save df to json
    df.to_json(json_file_path, orient='records', indent=4)

def Val(model_path, run_name):
    data_path = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/data/processed/YoloV8_dataset_OI7_parent_K5/YoloV8_dataset_OI7_K5.yaml'
    model = YOLO(model_path)
    # the medium is running out of memory so I am dropping the batch size to 4 (from 16)
    val_results = model.val(data=data_path, device=0, name=run_name)

    # save the results
    data = {
        'results': [val_results],
    }

    save_dir = os.path.join('/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/runs/detect', run_name)

    file_name = run_name.split('.')[0]+'_val_results'

    save_dict_to_json(data, save_dir, file_name)


def Train(model_name, cfg_path, run_name):
    #with torch.no_grad():           
    data_path = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/data/processed/YoloV8_dataset_OI7_parent_K5/YoloV8_dataset_OI7_K5.yaml'
    model = YOLO(model_name)
    if 'yolov8m' in model_name:
        model.train(data=data_path, cfg=cfg_path, name=run_name, batch=6)
    else:
        results = model.train(data=data_path, cfg=cfg_path, name=run_name)

# Retrieve command-line arguments (excluding the script name)
#args = ['yolov8n.yaml', '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/args.yaml', 'k5_data_split_list_1_yolov8n.yaml']
args = sys.argv[1:]  # sys.argv[0] is the script name
if args:
    print(f"Received arguments: {args}")
    # Perform operations with the arguments
else:
    print("No arguments received")

model_name = args[0]
cfg_path = args[1]
run_name = args[2]
print(model_name, cfg_path, run_name)

#args = ['yolov8n.yaml', '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/args.yaml', 'k5_data_split_list_1_yolov8n.yaml']

Val(model_name, run_name)

#Train(model_name, cfg_path, run_name)



