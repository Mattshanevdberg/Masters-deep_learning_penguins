import torch
import ultralytics
ckpt_file = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/runs/detect/train6/weights/best.pt'
print(ckpt_file)
metrics = torch.load(ckpt_file)["train_metrics"]
print(metrics)
fitness = metrics.get("fitness", 0.0)
print(fitness)