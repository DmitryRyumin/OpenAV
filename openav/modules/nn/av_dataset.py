#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Формирование набора данных
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################

import os
import subprocess
from torch.utils.data import Dataset
import torchaudio
import mediapipe as mp
import math
import numpy as np
import cv2
import librosa
import torch
from einops import rearrange

# ######################################################################################################################
# Константы
# ######################################################################################################################


# ######################################################################################################################
# Сообщения
# ######################################################################################################################


# ######################################################################################################################
# Набор данных
# ######################################################################################################################


class AVDataset(Dataset):
    def __init__(
        self,
        path_files=[],
        labels=[],
        subset="",
        shape=88,
        sampling_rate=16000,
        len_audio=19495,
        len_video=29,
        max_segment=1,
    ):
        self.path_files = path_files  # пути к видео
        self.labels = labels  # метки
        self.subset = subset  # имя выборки для сохранения данных
        self.shape_frame = shape  # фиксированный размер кадра
        self.sampling_rate = sampling_rate  # частота дискретизаци
        self.len_audio = len_audio  # длина аудио сегмента в отчетах
        self.len_video = len_video  # длина видео сегмента в кадрах
        self.max_segment = max_segment  # количество аудио и видео сегментов

        self.face_detector = mp.solutions.face_mesh  # инициализация детектора лиц

        self.lips = [
            61,
            146,
            91,
            181,
            84,
            17,
            314,
            405,
            321,
            375,
            291,
            409,
            270,
            269,
            267,
            0,
            37,
            39,
            40,
            185,
        ]  # координаты губ

        self.pl = [2048, 64, None, True, "reflect", 2.0, 64, "slaney", True, None]

    def norm_coordinates(self, normalized_x, normalized_y, image_width, image_height):

        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)

        return x_px, y_px

    def pad_sequence(self, lips):
        current_length = len(lips)

        if current_length < self.len_video:
            lips.extend([lips[-1]] * (self.len_video - current_length))

        return lips

    def get_box(self, fl, w, h):
        idx_to_coors = {}
        for idx, landmark in enumerate(fl.landmark):
            landmark_px = self.norm_coordinates(landmark.x, landmark.y, w, h)
            if landmark_px:
                idx_to_coors[idx] = landmark_px

        ys = [v[1] for _, v in idx_to_coors.items()]
        xs = [v[0] for _, v in idx_to_coors.items()]

        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xs), max(ys)

        (x_min, y_min) = max(x_min, 0), max(y_min, 0)
        (x_max, y_max) = min(x_max, w - 1), min(y_max, h - 1)

        return x_min, y_min, x_max, y_max

    def convert_video_to_audio(self, path, sampling_rate=16000):
        path_save = path

        if path[-3:] != "wav":
            path_save = path[:-3] + "wav"
            if not os.path.exists(path_save):
                ff_audio = "ffmpeg -i {} -vn -acodec pcm_s16le -ar 44100 -ac 2 {}".format(path, path_save)
                subprocess.call(ff_audio, shell=True)

        wav, sr = torchaudio.load(path_save)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != sampling_rate:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
            wav = transform(wav)
            sr = sampling_rate

        assert sr == sampling_rate
        return wav.squeeze(0)

    def make_padding(self, img):
        if img.shape[0] > 0 and img.shape[1] > 0:
            factor_0 = self.shape_frame / img.shape[0]
            factor_1 = self.shape_frame / img.shape[1]
            factor = min(factor_0, factor_1)
            dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
            img = cv2.resize(img, dsize)
            diff_0 = self.shape_frame - img.shape[0]
            diff_1 = self.shape_frame - img.shape[1]
            img = np.pad(
                img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), "mean"
            )
        if img.shape[0:2] != (self.shape_frame, self.shape_frame):
            img = cv2.resize(img, (self.shape_frame, self.shape_frame))
        return img

    def get_metadata(self, name_file):
        a_fss = []
        v_fss = []
        lip_areas = []

        video = cv2.VideoCapture(name_file)

        len_audio = self.len_audio

        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        with self.face_detector.FaceMesh(
            max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as face_mesh:

            while video.isOpened():
                success, image = video.read()

                if image is None:
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                image.flags.writeable = True

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for id_face, fl in enumerate(results.multi_face_landmarks):
                        x_norm_list = []
                        y_norm_list = []
                        for xxxx in self.lips:
                            x_px, y_px = self.norm_coordinates(fl.landmark[xxxx].x, fl.landmark[xxxx].y, w, h)
                            x_norm_list.append(x_px)
                            y_norm_list.append(y_px)
                        startX = int(min(x_norm_list))
                        startY = int(min(y_norm_list))
                        endX = int(max(x_norm_list))
                        endY = int(max(y_norm_list))
                        (startX, startY) = (max(0, startX), max(0, startY))
                        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                        cur_fr = image[startY:endY, startX:endX]
                        cur_fr = self.make_padding(cur_fr)
                        lip_areas.append(cur_fr / 255.0)
                else:
                    lip_areas.append(np.zeros((self.shape_frame, self.shape_frame, 3)))

        for i in range(0, total_frames, self.len_video):
            curr_frames = lip_areas[i : min(i + self.len_video, total_frames)]
            curr_frames = self.pad_sequence(curr_frames)
            v_fss.append(curr_frames)
            if i + self.len_video >= total_frames:
                break

        len_spec = len_audio / self.pl[1]
        if math.modf(len_spec)[0] == 0:
            len_spec += 1
        len_spec = math.ceil(len_spec) + 1

        try:
            wav = self.convert_video_to_audio(name_file)
            for i in range(0, len(wav) + 1, len_audio):
                curr_wav = wav[i : min(i + len_audio, len(wav))]
                if len(curr_wav) > self.pl[0]:
                    m_s = librosa.feature.melspectrogram(
                        y=np.asarray(curr_wav),
                        sr=self.sampling_rate,
                        n_fft=self.pl[0],
                        hop_length=self.pl[1],
                        win_length=self.pl[2],
                        center=self.pl[3],
                        pad_mode=self.pl[4],
                        power=self.pl[5],
                        n_mels=self.pl[6],
                        norm=self.pl[7],
                        htk=self.pl[8],
                        fmax=self.pl[9],
                    )

                    db_m_s = librosa.power_to_db(m_s, top_db=80)

                    if db_m_s.shape[1] < len_spec:
                        db_m_s = np.pad(db_m_s, ((0, 0), (0, len_spec - db_m_s.shape[1])), "mean")

                    db_m_s = db_m_s / 255.0 / 255.0
                    db_m_s = np.expand_dims(db_m_s, axis=-1)
                    a_fss.append(db_m_s)

                if i + len_audio >= len(wav):
                    break
        except Exception:
            pass

        if len(a_fss) > self.max_segment:
            a_fss = a_fss[: self.max_segment]
        elif len(a_fss) < self.max_segment:
            a_fss.extend([a_fss[-1]] * (self.max_segment - len(a_fss)))

        if len(v_fss) > self.max_segment:
            v_fss = v_fss[: self.max_segment]
        elif len(v_fss) < self.max_segment:
            v_fss.extend([v_fss[-1]] * (self.max_segment - len(v_fss)))

        return np.array(a_fss), np.array(v_fss)

    def __getitem__(self, index):
        a_fss, v_fss = self.get_metadata(str(self.path_files[index]))
        label = self.labels[index]  # метки
        return torch.FloatTensor(a_fss), torch.FloatTensor(v_fss), label

    def __len__(self):
        return len(self.path_files)


class AVTest:
    def __init__(
        self,
        shape=88,
        sampling_rate=16000,
        len_audio=19495,
        len_video=29,
        max_segment=1,
    ):
        self.shape_frame = shape  # фиксированный размер кадра
        self.sampling_rate = sampling_rate  # частота дискретизаци
        self.len_audio = len_audio  # длина аудио сегмента в отчетах
        self.len_video = len_video  # длина видео сегмента в кадрах
        self.max_segment = max_segment  # количество аудио и видео сегментов

        self.face_detector = mp.solutions.face_mesh  # инициализация детектора лиц

        self.lips = [
            61,
            146,
            91,
            181,
            84,
            17,
            314,
            405,
            321,
            375,
            291,
            409,
            270,
            269,
            267,
            0,
            37,
            39,
            40,
            185,
        ]  # координаты губ

        self.pl = [2048, 64, None, True, "reflect", 2.0, 64, "slaney", True, None]

    def norm_coordinates(self, normalized_x, normalized_y, image_width, image_height):

        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)

        return x_px, y_px

    def pad_sequence(self, lips):
        current_length = len(lips)

        if current_length < self.len_video:
            lips.extend([lips[-1]] * (self.len_video - current_length))

        return lips

    def get_box(self, fl, w, h):
        idx_to_coors = {}
        for idx, landmark in enumerate(fl.landmark):
            landmark_px = self.norm_coordinates(landmark.x, landmark.y, w, h)
            if landmark_px:
                idx_to_coors[idx] = landmark_px

        ys = [v[1] for _, v in idx_to_coors.items()]
        xs = [v[0] for _, v in idx_to_coors.items()]

        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xs), max(ys)

        (x_min, y_min) = max(x_min, 0), max(y_min, 0)
        (x_max, y_max) = min(x_max, w - 1), min(y_max, h - 1)

        return x_min, y_min, x_max, y_max

    def convert_video_to_audio(self, path, sampling_rate=16000):
        path_save = path

        if path[-3:] != "wav":
            path_save = path[:-3] + "wav"
            if not os.path.exists(path_save):
                ff_audio = "ffmpeg -i {} -vn -acodec pcm_s16le -ar 44100 -ac 2 {}".format(path, path_save)
                subprocess.call(ff_audio, shell=True)

        wav, sr = torchaudio.load(path_save)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != sampling_rate:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
            wav = transform(wav)
            sr = sampling_rate

        assert sr == sampling_rate
        return wav.squeeze(0)

    def make_padding(self, img):
        if img.shape[0] > 0 and img.shape[1] > 0:
            factor_0 = self.shape_frame / img.shape[0]
            factor_1 = self.shape_frame / img.shape[1]
            factor = min(factor_0, factor_1)
            dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
            img = cv2.resize(img, dsize)
            diff_0 = self.shape_frame - img.shape[0]
            diff_1 = self.shape_frame - img.shape[1]
            img = np.pad(
                img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), "mean"
            )
        if img.shape[0:2] != (self.shape_frame, self.shape_frame):
            img = cv2.resize(img, (self.shape_frame, self.shape_frame))
        return img

    def get_metadata(self, name_file):
        a_fss = []
        v_fss = []
        lip_areas = []

        # работа с видео сигналом
        video = cv2.VideoCapture(name_file)

        len_audio = self.len_audio

        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        with self.face_detector.FaceMesh(
            max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as face_mesh:

            while video.isOpened():
                success, image = video.read()

                if image is None:
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                image.flags.writeable = True

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for id_face, fl in enumerate(results.multi_face_landmarks):
                        x_norm_list = []
                        y_norm_list = []
                        for xxxx in self.lips:
                            x_px, y_px = self.norm_coordinates(fl.landmark[xxxx].x, fl.landmark[xxxx].y, w, h)
                            x_norm_list.append(x_px)
                            y_norm_list.append(y_px)
                        startX = int(min(x_norm_list))
                        startY = int(min(y_norm_list))
                        endX = int(max(x_norm_list))
                        endY = int(max(y_norm_list))
                        (startX, startY) = (max(0, startX), max(0, startY))
                        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                        cur_fr = image[startY:endY, startX:endX]
                        cur_fr = self.make_padding(cur_fr)
                        lip_areas.append(cur_fr / 255.0)
                else:
                    lip_areas.append(np.zeros((self.shape_frame, self.shape_frame, 3)))

        for i in range(0, total_frames, self.len_video):
            curr_frames = lip_areas[i : min(i + self.len_video, total_frames)]
            curr_frames = self.pad_sequence(curr_frames)
            v_fss.append(curr_frames)
            if i + self.len_video >= total_frames:
                break

        len_spec = len_audio / self.pl[1]
        if math.modf(len_spec)[0] == 0:
            len_spec += 1
        len_spec = math.ceil(len_spec) + 1

        wav = self.convert_video_to_audio(name_file)
        for i in range(0, len(wav) + 1, len_audio):
            curr_wav = wav[i : min(i + len_audio, len(wav))]
            if len(curr_wav) > self.pl[0]:
                m_s = librosa.feature.melspectrogram(
                    y=np.asarray(curr_wav),
                    sr=self.sampling_rate,
                    n_fft=self.pl[0],
                    hop_length=self.pl[1],
                    win_length=self.pl[2],
                    center=self.pl[3],
                    pad_mode=self.pl[4],
                    power=self.pl[5],
                    n_mels=self.pl[6],
                    norm=self.pl[7],
                    htk=self.pl[8],
                    fmax=self.pl[9],
                )

                db_m_s = librosa.power_to_db(m_s, top_db=80)

                if db_m_s.shape[1] < len_spec:
                    db_m_s = np.pad(db_m_s, ((0, 0), (0, len_spec - db_m_s.shape[1])), "mean")

                db_m_s = db_m_s / 255.0 / 255.0
                db_m_s = np.expand_dims(db_m_s, axis=-1)
                a_fss.append(db_m_s)

            if i + len_audio >= len(wav):
                break

        if len(a_fss) > self.max_segment:
            a_fss = a_fss[: self.max_segment]
        elif len(a_fss) < self.max_segment:
            a_fss.extend([a_fss[-1]] * (self.max_segment - len(a_fss)))

        if len(v_fss) > self.max_segment:
            v_fss = v_fss[: self.max_segment]
        elif len(v_fss) < self.max_segment:
            v_fss.extend([v_fss[-1]] * (self.max_segment - len(v_fss)))

        audio_data = torch.FloatTensor(np.array(a_fss))
        video_data = torch.FloatTensor(np.array(v_fss))

        audio_data = audio_data.unsqueeze(0)
        audio_data = rearrange(audio_data, "b g n l c -> b g c n l")
        video_data = video_data.unsqueeze(0)
        video_data = rearrange(video_data, "b g1 g2 h w c -> b g1 g2 c h w")

        return name_file, audio_data, video_data

    def __getitem__(self, index):
        a_fss, v_fss = self.get_metadata(str(self.path_files[index]))
        label = self.labels[index]
        return torch.FloatTensor(a_fss), torch.FloatTensor(v_fss), label

    def __len__(self):
        return len(self.path_files)
