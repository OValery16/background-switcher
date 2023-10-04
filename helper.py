import json
import os
import cv2
import numpy as np
import torch
from torch.hub import download_url_to_file


def norm_img(np_image: np.ndarray) -> np.ndarray:
    if len(np_image.shape) == 2:
        np_image = np_image[:, :, np.newaxis]
    np_image = np.transpose(np_image, (2, 0, 1))
    np_image = np_image.astype("float32") / 255
    return np_image


def load_jit_model(model_path: str, device: str = "cpu"
                   ) -> torch.jit.ScriptModule:
    try:
        model = torch.jit.load(model_path, map_location="cpu").to(device)
    except Exception as e:
        print(f"Error while loading the model: {e}")
        exit()
    model.eval()
    return model


def resize_max_size(
        np_image: np.ndarray, size_limit: int, interpolation=cv2.INTER_CUBIC
) -> np.ndarray:
    height, width = np_image.shape[:2]
    if max(height, width) > size_limit:
        ratio = size_limit / max(height, width)
        new_width = int(width * ratio + 0.5)
        new_height = int(height * ratio + 0.5)
        resized_image = cv2.resize(
            np_image, dsize=(new_width, new_height),
            interpolation=interpolation)
        return resized_image, (new_height, new_width)
    else:
        return np_image, np_image.shape[:2]

def resize_min_size(
        np_image: np.ndarray, size_limit: int, interpolation=cv2.INTER_CUBIC
) -> np.ndarray:
    height, width = np_image.shape[:2]

    ratio = size_limit / min(height, width)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    resized_image = cv2.resize(
        np_image, dsize=(new_width, new_height),
        interpolation=interpolation)
    return resized_image, (new_height, new_width)

def get_model_weight(name: str, models_info: str = "models.json") -> str:
    with open(models_info, "r") as f:
        models = json.load(f)
    model_url = models["models"][name]["url"]
    print(f"Downloading {model_url}")
    filename = model_url.split("/")[-1]
    if not os.path.exists(filename):
        download_url_to_file(model_url, filename, progress=True)
    return filename
