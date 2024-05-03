#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Утилиты
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################

import io
import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Персональные
from openav.modules.nn.models import rearrange

# ######################################################################################################################
# Функции
# ######################################################################################################################


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(dataloader, optimizer, criterion, model, device):
    running_loss = 0.0
    processed_size = 0.0

    for i, data in enumerate(tqdm(dataloader)):
        audio, video, labels = data
        optimizer.zero_grad()
        audio = rearrange(audio, "b g n l c -> b g c n l")
        video = rearrange(video, "b g1 g2 h w c -> b g1 g2 c h w")
        pred = model(audio.to(device), video.to(device))
        labels = labels.type(torch.LongTensor)
        loss = criterion(pred, labels.to(device))
        loss.backward()
        optimizer.step()

        processing_size = len(labels)
        processed_size += processing_size

        running_loss += loss.item() * processing_size

    avg_loss = running_loss / processed_size

    return avg_loss


def val_one_epoch(dataloader, criterion, model, device):
    running_loss = 0.0
    processed_size = 0.0
    predictions, targets = list(), list()
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            audio, video, labels = data
            audio = rearrange(audio, "b g n l c -> b g c n l")
            video = rearrange(video, "b g1 g2 h w c -> b g1 g2 c h w")
            pred = model(audio.to(device), video.to(device))
            loss = criterion(pred, labels.to(device))
            processing_size = len(labels)
            processed_size += processing_size
            running_loss += loss.item() * processing_size
            pred = torch.argmax(pred, dim=1).cpu().numpy()
            true = labels.cpu().numpy()
            predictions.extend(pred)
            targets.extend(true)
    avg_vloss = running_loss / processed_size
    acc = accuracy_score(targets, predictions)
    return acc, avg_vloss


def save_conf_matrix(
    y_true,
    y_pred,
    name_labels,
    filename,
    figsize_w=2600,
    figsize_h=2600,
    font_size=14,
    dpi=600,
    pad_inches=0,
    font_scale=1,
):
    c_m = confusion_matrix(y_true, y_pred)
    conf_matrix = pd.DataFrame(c_m, name_labels, name_labels)

    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.values.flatten()]
    group_percentages = [
        "{0:.2%}".format(value) for value in conf_matrix.div(np.sum(conf_matrix, axis=1), axis=0).values.flatten()
    ]

    labels = [f" {v1} \n {v2} " for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(c_m.shape)

    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(figsize_w * px, figsize_h * px))
    sns.set_theme(font_scale=font_scale)
    sns.heatmap(
        conf_matrix,
        cbar=False,
        annot=labels,
        square=True,
        fmt="",
        annot_kws={"size": font_size * font_scale},
        cmap="Blues",
        ax=ax,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    buf.seek(0)

    with open(filename, "wb") as f:
        f.write(buf.read())

    plt.close(fig)
