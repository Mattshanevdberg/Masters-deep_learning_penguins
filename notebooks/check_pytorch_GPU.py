# check out this article:
# https://saturncloud.io/blog/how-to-check-if-pytorch-is-using-the-gpu/
import torch

print(torch.cuda.is_available())
