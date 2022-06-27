import torch
import os

force_cpu = os.environ.get("DIFFVG_FORCE_CPU", "0") == "1"
use_gpu = False if force_cpu else torch.cuda.is_available()
device = torch.device('cuda') if use_gpu else torch.device('cpu')

def set_use_gpu(v):
    global use_gpu
    global device
    use_gpu = False if force_cpu else v
    if not use_gpu:
        device = torch.device('cpu')

def get_use_gpu():
    global use_gpu
    return use_gpu

def set_device(d):
    global device
    global use_gpu
    device = torch.device('cpu') if force_cpu else d
    use_gpu = device.type == 'cuda'

def get_device():
    global device
    return device
