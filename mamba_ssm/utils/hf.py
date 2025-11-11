import os
import json
import torch

# 可手动设置这两个常量
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

def load_config_hf(model_path):
    config_path = os.path.join(model_path, CONFIG_NAME)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)

def load_state_dict_hf(model_path, device=None, dtype=None):
    model_path = os.path.join(model_path, WEIGHTS_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weights file not found at: {model_path}")

    # 加载时映射到 CPU，稍后再转换 dtype 和 device
    state_dict = torch.load(model_path, map_location="cpu")

    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    if device is not None:
        state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict
