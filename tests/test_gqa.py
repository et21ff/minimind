import torch
import sys


sys.path.append("/root/minimind")
from model.model import repeat_kv

t = torch.ones(1, 2, 3, 4)
print(repeat_kv(t, 2).shape)
