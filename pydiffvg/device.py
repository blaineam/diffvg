import torch
import os

force_cpu = os.environ.get("DIFFVG_FORCE_CPU", False)
use_gpu = if force_cpu  False else torch.cuda.is_available()
device = torch.device('cuda') if use_gpu else torch.device('cpu')

def set_use_gpu(v):
    global use_gpu
    global device
    use_gpu = v
    if not use_gpu:
        device = torch.device('cpu')

def get_use_gpu():
    global use_gpu
    return use_gpu

def set_device(d):
    global device
    global use_gpu
    device = d
    use_gpu = device.type == 'cuda'

def get_device():
    global device
    return device
