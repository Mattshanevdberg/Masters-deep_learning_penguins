import torch
import ultralytics


ckpt_file = '/home/matthew/Desktop/Master_Dev/masters_penguin_pose_estimation/runs/detect/train_testRun_nanoPt_2_savejson/weights/best.pt'
print(ckpt_file)
all = torch.load(ckpt_file)
#print(all)
#metrics = torch.load(ckpt_file)["train_metrics"]
#print(metrics)
#fitness = metrics.get("fitness", 0.0)
#print(fitness)

m = ultralytics.YOLO(ckpt_file)
m.train()
print(m.metrics)
m