#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Утилиты
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################

import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

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
