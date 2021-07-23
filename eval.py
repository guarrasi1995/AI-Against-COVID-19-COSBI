import torch
import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image


def get_box(img, box):
    l_h = box[2] - box[0]
    l_w = box[3] - box[1]
    c_h = box[0] + l_h // 2
    c_w = box[1] + l_w // 2
    if l_h > l_w:
        img = img[box[0]:box[2], max(0, c_w - l_h // 2):c_w + l_h // 2]
    elif l_w > l_h:
        img = img[max(0, c_h - l_w // 2):c_h + l_w // 2, box[1]:box[3]]
    else:
        img = img[box[0]:box[2], box[1]:box[3]]
    return img


def normalize(img):
    img -= img.mean()
    img /= img.std()
    return img


def loader(path, img_dim, box=None):
    img = Image.open(path)
    # to numpy
    img = np.array(img).astype(float)
    # to grayscale
    if img.ndim > 2:
        img = img.mean(axis=2)
    # select box area
    if box:
        img = get_box(img, box)
    # resize
    img = cv2.resize(img, (img_dim, img_dim))
    # normalize
    img = normalize(img)
    # to 3 channels
    img = np.stack((img, img, img), axis=-1)
    # to Tensor
    img = torch.Tensor(img)
    img = img.transpose(0, 2)
    return img


def evaluate(img_path_list, model_path="./models/cnn/single_model/model.pt"):

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Box
    box_data = pd.read_excel("./data/processed/data_submission.xlsx", index_col="img", dtype=list, usecols=["img", "all"])

    # Model
    model = torch.load(model_path, map_location=device)
    model = model.to(device)

    # Predict loop
    predictions = []
    model.eval()
    with torch.no_grad():
        for img_path in img_path_list:
            img_file = os.path.basename(img_path)
            box = box_data.loc[img_file].item()
            inputs = loader(path=img_path, img_dim=224, box=eval(box))
            inputs = inputs.unsqueeze(0).to(device)
            outputs = model(inputs.float())
            predictions.append((outputs[0][1].item() >= -0.830745)*1)

    return predictions

