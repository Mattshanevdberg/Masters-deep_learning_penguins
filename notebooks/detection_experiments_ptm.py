import os
import random
from ultralytics import YOLO
import torch
import pandas as pd
import numpy as np
import gc
#import multiprocessing

def Train(model_name, cfg_path):
    #with torch.no_grad():           
    data_path = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/data/processed/YoloV8_dataset_OI7_parent_NG/YoloV8_dataset_OI7_NG.yaml'
    model = YOLO(model_name)
    # the medium is running out of memory so I am dropping the batch size to 4 (from 16)
    results = model.train(data=data_path, cfg=cfg_path, batch=7, epochs=2)

model1 = 'yolov8m.pt'
cfg_path = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/args.yaml'


Train(model1, cfg_path)

def Val():
    data_path = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/data/processed/YoloV8_dataset_OI7_parent_NG/YoloV8_dataset_OI7_NG.yaml'
    model = YOLO('/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/runs/detect/train53/weights/best.pt')
    # the medium is running out of memory so I am dropping the batch size to 4 (from 16)
    val_results = model.val(data=data_path, device=0)
    return val_results
print('MPT')
#results = Val()