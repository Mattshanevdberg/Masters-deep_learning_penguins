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
    results = model.train(data=data_path, cfg=cfg_path)
    
model1 = 'yolov8n.yaml'
cfg_path = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/args.yaml'


Train(model1, cfg_path)

'''
def Train(model_name, cfg_path):
    #with torch.no_grad():           
    data_path = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/data/processed/YoloV8_dataset_OI7_parent_NG/YoloV8_dataset_OI7_NG.yaml'
    model = YOLO(model_name)
    results = model.train(data=data_path, cfg=cfg_path, epochs=2)
    #if model.grad:
    #    model.detach_()
    model = None
    del model
    #if results.grad:
    #    results.detach_()
    results = None
    del results
    print('thing 2')
    torch.cuda.empty_cache()
    print('thing 3')
    gc.collect()

if __name__ == "__main__":
    model1 = 'yolov8n.yaml'
    cfg_path = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/args.yaml'
    p = multiprocessing.Process(target=Train, args=(model1, cfg_path))
    print('start process 1')
    p.start()
    p.join()  # Wait for the process to finish
    print('finish process 1')
    p.terminate()
    print('terminated p1')

    # When the process completes, all its memory should be freed.
    # You can start another process for another training session
    model2 = 'yolov8s.yaml'
    p2 = multiprocessing.Process(target=Train, args=(model2, cfg_path))
    print('start process 2')
    p2.start()
    p2.join()  # Wait for the process to finish
    print('finish process 2')
    p2.terminate()
    print('terminated p2')

'''
'''
model1 = 'yolov8n.yaml'
cfg_path = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/notebooks/args.yaml'


Train(model1, cfg_path)

print(torch.cuda.memory_summary())
print('thing 0.1')
torch.cuda.reset_max_memory_allocated()
print('thing 0.2')
torch.cuda.reset_accumulated_memory_stats()
print('thing 0.3')
torch.cuda.device_reset()
print('thing 1')
torch.cuda.reset_peak_memory_stats()

print('thing 2')
torch.cuda.empty_cache()
print('thing 3')
gc.collect()

model2 = 'yolov8s.yaml'
Train(model2, cfg_path)
'''