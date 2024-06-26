# check out this article:
# https://saturncloud.io/blog/how-to-check-if-pytorch-is-using-the-gpu/
import torch
print(torch.__version__)
print(torch.version.cuda)
#print(torch.cuda.current_device())
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
