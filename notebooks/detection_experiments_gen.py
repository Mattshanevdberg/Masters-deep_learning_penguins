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

def Val():
    data_path = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/data/processed/YoloV8_dataset_OI7_parent_K5/YoloV8_dataset_OI7_K5.yaml'
    model = YOLO('/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/runs/detect/train34/weights/best.pt')
    # the medium is running out of memory so I am dropping the batch size to 4 (from 16)
    val_results = model.val(data=data_path, device=0)
    return val_results

def Train(model_name, cfg_path, run_name):
    #with torch.no_grad():           
    data_path = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/data/processed/YoloV8_dataset_OI7_parent_K5/YoloV8_dataset_OI7_K5.yaml'
    model = YOLO(model_name)
    if 'yolov8m' in model_name:
        model.train(data=data_path, cfg=cfg_path, name=run_name, batch=2)
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
#print(model_name, cfg_path, run_name)

#args = ['yolov8n.yaml', '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/args.yaml', 'k5_data_split_list_1_yolov8n.yaml']

Train(model_name, cfg_path, run_name)



