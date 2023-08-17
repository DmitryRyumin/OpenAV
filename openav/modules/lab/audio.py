#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Аудиомодальность
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
# Подавление Warning
import warnings

for warn in [UserWarning, FutureWarning]:
    warnings.filterwarnings("ignore", category=warn)

from dataclasses import dataclass  # Класс данных

import os  # Взаимодействие с файловой системой
import re  # Регулярные выражения
import subprocess  # Работа с процессами
import numpy as np  # Научные вычисления
import torch  # Машинное обучение от Facebook
import torchvision  # Работа с видео от Facebook
import torchaudio  # Работа с аудио от Facebook
import filetype  # Определение типа файла и типа MIME
import json  # Кодирование и декодирование данных в удобном формате

# Парсинг URL
import urllib.parse
import urllib.error

from IPython.utils import io  # Подавление вывода
from pathlib import Path, PosixPath  # Работа с путями в файловой системе
from datetime import datetime, timedelta  # Работа со временем
from pymediainfo import MediaInfo  # Получение meta данных из медиафайлов

from vosk import Model, KaldiRecognizer, SetLogLevel  # Распознавание речи

import augly.audio as audags # Аугментация аудиофайлов

# Типы данных
from typing import List, Dict, Union, Optional

from types import FunctionType

# Персональные
from openav.modules.core.exceptions import (
    TypeEncodeVideoError,
    PresetCFREncodeVideoError,
    SRInputTypeError,
    IsNestedCatalogsNotFoundError,
    IsNestedDirectoryVNotFoundError,
    IsNestedDirectoryANotFoundError,
    SamplingRateError,
    WindowSizeSamplesError,
    VolumeDbError,
    CutoffHzError,
)
from openav.modules.file_manager.yaml_manager import Yaml  # Класс для работы с YAML

# ######################################################################################################################
# Константы
# ######################################################################################################################
TYPES_ENCODE: List[str] = ["qscale", "crf"]  # Типы кодирования
CRF_VALUE: int = 23  # Качество кодирования (от 0 до 51)
# Скорость кодирования и сжатия
PRESETS_CRF_ENCODE: List[str] = [
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
]
SR_INPUT_TYPES: List[str] = ["audio", "video"]  # Типы файлов для распознавания речи
SAMPLING_RATE_VAD: List[str] = [8000, 16000]  # Частота дискретизации
THRESHOLD_VAD: float = 0.56  # Порог вероятности речи (от 0.0 до 1.0)
MIN_SPEECH_DURATION_MS_VAD: int = 250  # Минимальная длительность речевого фрагмента в миллисекундах
# Минимальная длительность тишины в выборках между отдельными речевыми фрагментами
MIN_SILENCE_DURATION_MS_VAD: int = 50

# Количество выборок в каждом окне
# (512, 1024, 1536 для частоты дискретизации 16000 или 256, 512, 768 для частоты дискретизации 8000)
WINDOW_SIZE_SAMPLES_VAD: Dict[int, List[int]] = {8000: [256, 512, 768], 16000: [512, 1024, 1536]}
SPEECH_PAD_MS: int = 150  # Внутренние отступы для итоговых речевых фрагментов
# Суффиксы каналов аудиофрагментов
FRONT: Dict[str, List[str]] = {"mono": ["_mono"], "stereo": ["_left", "_right"]}
EXT_AUDIO_AUG: str = "flac"  # Расширение для сохраняемого аудио
EXT_AUDIO: str = "wav"  # Расширение для сохраняемого аудио
VOSK_SUPPORTED_LANGUAGES: List[str] = ["ru", "en"]  # Поддерживаемые языки (Vosk)
VOSK_SUPPORTED_DICTS: List[str] = ["small", "big"]  # Размеры словарей (Vosk)
VOSK_SPEECH_LEFT_PAD_MS: int = 0  # Внутренний левый отступ для итоговых речевых фрагментов
VOSK_SPEECH_RIGHT_PAD_MS: int = 0  # Внутренний правый отступ для итоговых речевых фрагментов

VOLUME_DB_AUGMENTATION: List[str] = [0, 1000]  # TODO: fix description
CUTOFF_HZ_AUGMENTATION: List[str] = [0, 1000]  # TODO: fix description


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class AudioMessages(Yaml):
    """Класс для сообщений

    Args:
        path_to_logs (str): Смотреть :attr:`~openav.modules.core.logging.Logging.path_to_logs`
        lang (str): Смотреть :attr:`~openav.modules.core.language.Language.lang`
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self._wrong_type_encode: str = self._('Тип кодирования видео должен быть одним из "{}"') + self._em
        self._wrong_preset_crf_encode: str = (
            self._("Скорость кодирования и сжатия видео должна быть " 'одной из "{}"') + self._em
        )
        self._wrong_sr_input_type: str = self._('Тип файлов для распознавания должен быть одним из "{}"') + self._em
        self._wrong_sampling_rate_vad: str = (
            self._('Частота дискретизации речевого сигнала должна быть одной из "{}"') + self._em
        )
        self._wrong_window_size_samples_type: str = (
            self._('Для частоты дискретизации "{}" количество выборок в каждом окне должно быть одним из "{}"')
            + self._em
        )

        self._download_model_from_repo: str = self._('Загрузка VAD модели "{}" из репозитория {}') + self._em

        # Переопределение
        self._automatic_download: str = self._("Загрузка Vosk модели") + ' "{}"' + self._em
        self._automatic_download_progress: str = self._automatic_download + " {}%" + self._em

        self._vosk_model_activation: str = self._("Активация Vosk модели") + ' "{}"' + self._em
        self._sr_not_recognized: str = self._("Речь не найдена") + self._em

        self._wrong_volume_db_aug: str = (
                self._('Громкость обработки речевого сигнала должна быть в пределах "{}"') + self._em
        )
        self._wrong_cutoff_hz_aug: str = (
                self._('Частота обработки речевого сигнала должна быть в пределах "{}"') + self._em
        )

        self._subfolders_search: str = (
            self._('Поиск вложенных директорий в директории "{}" (глубина вложенности: {})') + self._em
        )
        self._subfolders_not_found: str = self._("В указанной директории вложенные директории не найдены") + self._em

        self._files_av_find: str = (
            self._('Поиск файлов с расширениями "{}" в директории "{}" (глубина вложенности: {})') + self._em
        )

        self._files_analysis: str = self._("Анализ файлов") + self._em

        self._url_error: str = self._("Не удалось скачать модель{}") + self._em
        self._url_error_code: str = self._(" (ошибка {})")

        self._vad_true: str = self._("Все файлы успешно проанализированы") + self._em

        self._aug_true: str = self._("Все файлы успешно обработаны") + self._em


# ######################################################################################################################
# Аудио
# ######################################################################################################################
@dataclass
class Audio(AudioMessages):
    """Класс для обработки аудиомодальности

    Args:
        path_to_logs (str): Смотреть :attr:`~openav.modules.core.logging.Logging.path_to_logs`
        lang (str): Смотреть :attr:`~openav.modules.core.language.Language.lang`
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self._github_repo_vad: str = "snakers4/silero-vad"  # Репозиторий для загрузки VAD
        self._vad_model: str = "silero_vad"  # VAD модель

        # Пути к моделям распознавания речи
        self._vosk_models_url: Dict = {"vosk": "https://alphacephei.com/vosk/models/"}

        # Модели для распознавания речи
        self._vosk_models_for_sr: Dict = {
            "vosk": {
                "languages": VOSK_SUPPORTED_LANGUAGES,  # Поддерживаемые языки
                "dicts": VOSK_SUPPORTED_DICTS,  # Размеры словарей
                # Русский язык
                "ru": {"big": "vosk-model-ru-0.42.zip", "small": "vosk-model-small-ru-0.22.zip"},
                # Английский язык
                "en": {"big": "vosk-model-en-us-0.22.zip", "small": "vosk-model-small-en-us-0.15.zip"},
            },
        }

        self.vosk_language_sr: str = VOSK_SUPPORTED_LANGUAGES[0]  # Язык для распознавания речи (Vosk)
        self.vosk_dict_language_sr: str = VOSK_SUPPORTED_DICTS[1]  # Размер словаря для распознавания речи (Vosk)

        # ----------------------- Только для внутреннего использования внутри класса

        self.__model_vad: Optional[torch.jit._script.RecursiveScriptModule] = None  # VAD модель
        self.__get_speech_ts: Optional[FunctionType] = None  # Временные метки VAD

        self.__len_paths: int = 0  # Количество аудиовизуальных файлов
        self.__curr_path: PosixPath = ""  # Текущий аудиовизуальный файл
        self.__splitted_path: str = ""  # Локальный путь до директории
        self.__i: int = 0  # Счетчик
        self.__local_path: List[Union[List[PosixPath], str]] = ""  # Локальный путь
        self.__aframes: torch.Tensor = torch.empty((), dtype=torch.float32)  # Аудиокадры

        self.__dataset_video_vad: List[str] = []  # Пути до директорий с разделенными видеофрагментами
        self.__dataset_audio_vad: List[str] = []  # Пути до директорий с разделенными аудиофрагментами
        self.__unprocessed_files: List[str] = []  # Пути к файлам на которых VAD не отработал
        self.__not_saved_files: List[str] = []  # Пути к файлам которые не сохранились при обработке VAD

        # Результат дочернего процесс распознавания речи
        self.__subprocess_vosk_sr: Union[List[str], Dict[str, List[str]]] = []

        self.__type_encode: str = ""  # Тип кодирования
        self.__crf_value: int = 0  # Качество кодирования (от 0 до 51)
        self.__presets_crf_encode: str = ""  # Скорость кодирования и сжатия
        self.__sr_input_type: str = ""  # Тип файлов для распознавания речи
        self.__sampling_rate_vad: int = 0  # Частота дискретизации
        self.__threshold_vad: float = 0.0  # Порог вероятности речи
        self.__min_speech_duration_ms_vad: int = 0  # Минимальная длительность речевого фрагмента в миллисекундах
        # Минимальная длительность тишины в выборках между отдельными речевыми фрагментами
        self.__min_silence_duration_ms_vad: int = 0
        self.__window_size_samples_vad: int = 0  # Количество выборок в каждом окне
        self.__speech_pad_ms_vad: int = 0  # Внутренние отступы для итоговых речевых фрагментов

        # Метаданные для видео и аудио
        self.__file_metadata: Dict[
            str,
            Union[int, float],
        ] = {"video_fps": 0.0, "audio_fps": 0}

        self.__front: List[str] = []  # Суффиксы каналов аудиофрагментов
        self.__part_video_path: str = ""  # Путь до видеофрагмента
        self.__part_audio_path: str = ""  # Путь до аудиофрагмента

        self.__curr_ts: str = ""  # Текущее время (TimeStamp)

        self.__freq_sr: int = 16000  # Частота дискретизации

        self.__speech_model: Optional[Model] = None  # Модель распознавания речи
        self.__speech_rec: Optional[KaldiRecognizer] = None  # Активация распознавания речи
        self.__keys_speech_rec: List[str] = ["result", "text"]  # Ключи из результата распознавания речи

        self.__vosk_speech_left_pad_ms: int = 0  # Внутренний левый отступ для итоговых речевых фрагментов
        self.__vosk_speech_right_pad_ms: int = 0  # Внутренний правый отступ для итоговых речевых фрагментов

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def vosk_language_sr(self) -> str:
        """Получение/установка языка для распознавания речи

        Args:
            (str): Язык

        Returns:
            str: Язык
        """

        return self.__language_sr

    @vosk_language_sr.setter
    def vosk_language_sr(self, lang: str):
        """Установка языка для распознавания речи"""

        try:
            # Проверка аргументов
            if type(lang) is not str or (lang in VOSK_SUPPORTED_LANGUAGES) is False:
                raise TypeError
        except TypeError:
            pass
        else:
            self.__language_sr = lang

    @property
    def vosk_dict_language_sr(self) -> str:
        """Получение/установка размера словаря для распознавания речи

        Args:
            (str): Размер словаря

        Returns:
            str: Размер словаря
        """

        return self.__dict_language_sr

    @vosk_dict_language_sr.setter
    def vosk_dict_language_sr(self, dict_size: str):
        """Установка размера словаря для распознавания речи"""

        try:
            # Проверка аргументов
            if type(dict_size) is not str or (dict_size in VOSK_SUPPORTED_DICTS) is False:
                raise TypeError
        except TypeError:
            pass
        else:
            self.__dict_language_sr = dict_size

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (приватные)s
    # ------------------------------------------------------------------------------------------------------------------

    # Детальная информация о текущем процессе распознавания речи (Vosk)
    @staticmethod
    def __speech_rec_result(
        keys: List[str], speech_rec_res: Dict[str, Union[List[Dict[str, Union[float, str]]], str]]
    ) -> List[Union[str, float]]:
        """Детальная информация о текущем процессе распознавания речи (Vosk)

        Args:
            keys (List[str]): Ключи из результата распознавания
            speech_rec_res (Dict[str, Union[List[Dict[str, Union[float, str]]], str]]): Текущий результат

        Returns:
            List[Union[str, float]]: Список из текстового представления речи, начала и конца речи
        """

        # Детальная информация распознавания
        if keys[0] in speech_rec_res.keys():
            start = speech_rec_res[keys[0]][0]["start"]  # Начало речи

            if len(speech_rec_res[keys[0]]) == 1:
                idx = 0  # Индекс
            else:
                idx = -1  # Индекс

            end = speech_rec_res[keys[0]][idx]["end"]  # Конец речи
            curr_text = speech_rec_res[keys[1]]  # Распознанный текст

            return [curr_text, round(start, 2), round(end, 2)]  # Текущий результат

        return []

    def __subprocess_vosk_sr_video(self, out: bool) -> List[str]:
        """Дочерний процесс распознавания речи (Vosk) - видео

        Args:
            out (bool) Отображение

        Returns:
            List[str]: Список с текстовыми представлениями речи, начала и конца речи
        """

        try:
            # https://trac.ffmpeg.org/wiki/audio%20types
            # Выполнение в новом процессе
            with subprocess.Popen(
                ["ffmpeg", "-loglevel", "quiet", "-i", self.__curr_path]
                + ["-ar", str(self.__freq_sr), "-ac", str(1), "-f", "s16le", "-"],
                stdout=subprocess.PIPE,
            ) as process:
                results_recognized = []  # Результаты распознавания

                while True:
                    data = process.stdout.read(4000)
                    if len(data) == 0:
                        break

                    curr_res = []  # Текущий результат

                    # Распознанная речь
                    if self.__speech_rec.AcceptWaveform(data):
                        speech_rec_res = json.loads(self.__speech_rec.Result())  # Текущий результат

                        # Детальная информация распознавания
                        curr_res = self.__speech_rec_result(self.__keys_speech_rec, speech_rec_res)
                    else:
                        self.__speech_rec.PartialResult()

                    if len(curr_res) == 3:
                        results_recognized.append(curr_res)

                speech_rec_fin_res = json.loads(self.__speech_rec.FinalResult())  # Итоговый результат распознавания
                # Детальная информация распознавания
                speech_rec_fin_res = self.__speech_rec_result(self.__keys_speech_rec, speech_rec_fin_res)

                # Результат распознавания
                if len(speech_rec_fin_res) == 3:
                    results_recognized.append(speech_rec_fin_res)

                if len(results_recognized) == 0:
                    self.message_error(self._sr_not_recognized, out=out)
                    return []

                return results_recognized
        except OSError:
            self.message_error(self._sr_not_recognized, out=out)
            return []
        except Exception:
            self.message_error(self._unknown_err, out=out)
            return []

    # Дочерний процесс распознавания речи (Vosk) - аудио
    def __subprocess_vosk_sr_audio(self, out: bool) -> Dict[str, List[str]]:
        """Дочерний процесс распознавания речи (Vosk)  - аудио

        Args:
            out (bool) Отображение

        Returns:
            Dict[str, List[str]]: Словарь со вложенными списками из текстового представления речи, начала и конца речи
        """

        # Количество каналов в аудиодорожке
        channels_audio = MediaInfo.parse(self.__curr_path).to_data()["tracks"][1]["channel_s"]

        # Количество каналов больше 2
        if channels_audio > 2:
            self.__unprocessed_files.append(self.__curr_path)
            return {}

        map_channels = {"Mono": "0.0.0"}  # Извлечение моно
        if channels_audio == 2:
            map_channels = {"Left": "0.0.0", "Right": "0.0.1"}  # Стерео

        try:
            results_recognized = {}  # Результаты распознавания

            # Проход по всем каналам
            for front, channel in map_channels.items():
                results_recognized[front] = []  # Словарь для результатов определенного канала
                # https://trac.ffmpeg.org/wiki/audio%20types
                # Выполнение в новом процессе
                with subprocess.Popen(
                    ["ffmpeg", "-loglevel", "quiet", "-i", self.__curr_path]
                    + [
                        "-ar",
                        str(self.__freq_sr),
                        "-map_channel",
                        channel,
                        "-acodec",
                        "pcm_s16le",
                        "-ac",
                        str(1),
                        "-f",
                        "s16le",
                        "-",
                    ],
                    stdout=subprocess.PIPE,
                ) as process:
                    while True:
                        data = process.stdout.read(4000)
                        if len(data) == 0:
                            break

                        curr_res = []  # Текущий результат

                        # Распознанная речь
                        if self.__speech_rec.AcceptWaveform(data):
                            speech_rec_res = json.loads(self.__speech_rec.Result())  # Текущий результат

                            # Детальная информация распознавания
                            curr_res = self.__speech_rec_result(self.__keys_speech_rec, speech_rec_res)
                        else:
                            self.__speech_rec.PartialResult()

                        if len(curr_res) == 3:
                            results_recognized[front].append(curr_res)

                    speech_rec_fin_res = json.loads(self.__speech_rec.FinalResult())  # Итоговый результат распознавания
                    # Детальная информация распознавания
                    speech_rec_fin_res = self.__speech_rec_result(self.__keys_speech_rec, speech_rec_fin_res)

                    # Результат распознавания
                    if len(speech_rec_fin_res) == 3:
                        results_recognized[front].append(speech_rec_fin_res)

            if bool([l for l in results_recognized.values() if l != []]) is False:
                self.message_error(self._sr_not_recognized, out=out)
                return {}

            return results_recognized
        except OSError:
            self.message_error(self._sr_not_recognized, out=out)
            return {}
        except Exception:
            self.message_error(self._unknown_err, out=out)
            return {}

    def __audio_analysis(self) -> bool:
        """Анализ аудиодорожки (VAD)

        Returns:
            bool: **True** если анализ аудиодорожки произведен, в обратном случае **False**
        """

        # Количество каналов в аудиодорожке
        channels_audio = self.__aframes.shape[0]
        if channels_audio > 2:
            self.__unprocessed_files.append(self.__curr_path)
            return False

        if channels_audio == 1:
            self.__front = FRONT["mono"]  # Моно канал
        elif channels_audio == 2:
            self.__front = FRONT["stereo"]  # Стерео канал

        # Тип файла
        kind = filetype.guess(self.__curr_path)

        # Текущее время (TimeStamp)
        # см. datetime.fromtimestamp()
        self.__curr_ts = str(datetime.now().timestamp()).replace(".", "_")

        # Проход по всем каналам
        for channel in range(0, channels_audio):
            try:
                # Получение временных меток
                speech_timestamps = self.__get_speech_ts(
                    audio=self.__aframes[channel],
                    model=self.__model_vad,
                    threshold=self.__threshold_vad,
                    sampling_rate=self.__sampling_rate_vad,
                    min_speech_duration_ms=self.__min_speech_duration_ms_vad,
                    max_speech_duration_s=float("inf"),
                    min_silence_duration_ms=self.__min_silence_duration_ms_vad,
                    window_size_samples=self.__window_size_samples_vad,
                    speech_pad_ms=self.__speech_pad_ms_vad,
                    return_seconds=False,
                    visualize_probs=False,
                    progress_tracking_callback=None,
                )
            except Exception:
                self.__unprocessed_files.append(self.__curr_path)
                return False
            else:

                def join_path(dir_va):
                    return os.path.join(self.path_to_dataset_vad, dir_va, self.__splitted_path)

                # Временные метки найдены
                if len(speech_timestamps) > 0:
                    try:
                        # Видео
                        if kind.mime.startswith("video/") is True:
                            if join_path(self.dir_va_names[0]) not in self.__dataset_video_vad:
                                # Директория с разделенными видеофрагментами
                                self.__dataset_video_vad.append(join_path(self.dir_va_names[0]))
                                if not os.path.exists(self.__dataset_video_vad[-1]):
                                    # Директория не создана
                                    if self.create_folder(self.__dataset_video_vad[-1], out=False) is False:
                                        raise IsNestedDirectoryVNotFoundError
                        # Аудио
                        if join_path(self.dir_va_names[1]) not in self.__dataset_audio_vad:
                            # Директория с разделенными аудиофрагментами
                            self.__dataset_audio_vad.append(join_path(self.dir_va_names[1]))
                            if not os.path.exists(self.__dataset_audio_vad[-1]):
                                # Директория не создана
                                if self.create_folder(self.__dataset_audio_vad[-1], out=False) is False:
                                    raise IsNestedDirectoryANotFoundError
                    except (IsNestedDirectoryVNotFoundError, IsNestedDirectoryANotFoundError):
                        self.__unprocessed_files.append(self.__curr_path)
                        return False
                    except Exception:
                        self.__unprocessed_files.append(self.__curr_path)
                        return False

                # Проход по всем найденным меткам
                for cnt, curr_timestamps in enumerate(speech_timestamps):
                    # Начальное время
                    start_time = timedelta(seconds=curr_timestamps["start"] / self.__file_metadata["audio_fps"])
                    # Конечное время
                    end_time = timedelta(seconds=curr_timestamps["end"] / self.__file_metadata["audio_fps"])

                    diff_time = end_time - start_time  # Разница между начальным и конечным временем

                    # Путь до аудиофрагмента
                    self.__part_audio_path = os.path.join(
                        self.__dataset_audio_vad[-1],
                        Path(self.__curr_path).stem
                        + "_"
                        + str(cnt)
                        + self.__front[channel]
                        + "_"
                        + self.__curr_ts
                        + "."
                        + EXT_AUDIO,
                    )

                    # Видео
                    if kind.mime.startswith("video/") is True:
                        # Путь до видеофрагмента
                        self.__part_video_path = os.path.join(
                            self.__dataset_video_vad[-1],
                            Path(self.__curr_path).stem
                            + "_"
                            + str(cnt)
                            + "_"
                            + self.__curr_ts
                            + Path(self.__curr_path).suffix.lower(),
                        )

                    def not_saved_files():
                        return self.__not_saved_files.append([self.__curr_path, start_time, end_time])

                    call_audio, call_video = 0, 0

                    try:
                        # Видео
                        if kind.mime.startswith("video/") is True:
                            if channel == 0:
                                # Варианты кодирования
                                if self.__type_encode == TYPES_ENCODE[0]:
                                    # https://trac.ffmpeg.org/wiki/Encode/MPEG-4
                                    ff_v = 'ffmpeg -loglevel quiet -ss {} -i "{}" -{} 0 -to {} "{}"'.format(
                                        start_time,
                                        self.__curr_path,
                                        self.__type_encode,
                                        diff_time,
                                        self.__part_video_path,
                                    )
                                elif self.__type_encode == TYPES_ENCODE[1]:
                                    # https://trac.ffmpeg.org/wiki/Encode/H.264
                                    ff_v = 'ffmpeg -loglevel quiet -ss {} -i "{}" -{} {} -preset {} -to {} "{}"'.format(
                                        start_time,
                                        self.__curr_path,
                                        self.__type_encode,
                                        self.__crf_value,
                                        self.__presets_crf_encode,
                                        diff_time,
                                        self.__part_video_path,
                                    )
                            if channels_audio == 1:  # Моно канал
                                ff_a = 'ffmpeg -loglevel quiet -i "{}" -vn -c:a pcm_f32le ' '-ss {} -to {} "{}"'.format(
                                    self.__curr_path, start_time, end_time, self.__part_audio_path
                                )
                            elif channels_audio == 2:  # Стерео канал
                                ff_a = (
                                    'ffmpeg -loglevel quiet -i "{}" -vn -c:a pcm_f32le -map_channel 0.1.{} -ss {} '
                                    '-to {} "{}"'.format(
                                        self.__curr_path, channel, start_time, end_time, self.__part_audio_path
                                    )
                                )

                        # Аудио
                        if kind.mime.startswith("audio/") is True:
                            if channels_audio == 1:  # Моно канал
                                ff_a = 'ffmpeg -loglevel quiet -i "{}" -ss {} -to {} -c copy "{}"'.format(
                                    self.__curr_path, start_time, end_time, self.__part_audio_path
                                )
                            elif channels_audio == 2:  # Стерео канал
                                ff_a = 'ffmpeg -loglevel quiet -i "{}" -map_channel 0.0.{} -ss {} -to {} "{}"'.format(
                                    self.__curr_path, channel, start_time, end_time, self.__part_audio_path
                                )
                    except IndexError:
                        not_saved_files()
                    except Exception:
                        not_saved_files()
                    else:
                        # Видео
                        if kind.mime.startswith("video/") is True and channel == 0:
                            call_video = subprocess.call(ff_v, shell=True)
                        # Аудио
                        call_audio = subprocess.call(ff_a, shell=True)

                        try:
                            if call_audio == 1 or call_video == 1:
                                raise OSError
                        except OSError:
                            not_saved_files()
                        except Exception:
                            not_saved_files()
                        else:
                            try:
                                # Видео
                                if kind.mime.startswith("video/") is True:
                                    # Чтение файла
                                    _, _, _ = torchvision.io.read_video(self.__part_video_path)
                                _, _ = torchaudio.load(self.__part_audio_path)
                            except Exception:
                                not_saved_files()
        return True

    def __audio_analysis_vosk_sr(self) -> bool:
        """Анализ аудиодорожки (Vosk)

        Returns:
            bool: **True** если анализ аудиодорожки произведен, в обратном случае **False**
        """

        # Тип файла
        kind = filetype.guess(self.__curr_path)

        # Видео
        if kind.mime.startswith("video/") is True:
            track = 2
        # Аудио
        if kind.mime.startswith("audio/") is True:
            track = 1

        # Количество каналов в аудиодорожке
        channels_audio = MediaInfo.parse(self.__curr_path).to_data()["tracks"][track]["channel_s"]
        if channels_audio > 2:
            self.__unprocessed_files.append(self.__curr_path)
            return False

        if channels_audio == 1:
            self.__front = FRONT["mono"]  # Моно канал
        elif channels_audio == 2:
            self.__front = FRONT["stereo"]  # Стерео канал

        # Текущее время (TimeStamp)
        # см. datetime.fromtimestamp()
        self.__curr_ts = str(datetime.now().timestamp()).replace(".", "_")

        def join_path(dir_va):
            return os.path.join(self.path_to_dataset_vosk_sr, dir_va, self.__splitted_path)

        # Временные метки найдены
        if len(self.__subprocess_vosk_sr) > 0:
            try:
                # Видео
                if kind.mime.startswith("video/") is True:
                    if join_path(self.dir_va_names[0]) not in self.__dataset_video_vad:
                        # Директория с разделенными видеофрагментами
                        self.__dataset_video_vad.append(join_path(self.dir_va_names[0]))
                        if not os.path.exists(self.__dataset_video_vad[-1]):
                            # Директория не создана
                            if self.create_folder(self.__dataset_video_vad[-1], out=False) is False:
                                raise IsNestedDirectoryVNotFoundError
                # Аудио
                if join_path(self.dir_va_names[1]) not in self.__dataset_audio_vad:
                    # Директория с разделенными аудиофрагментами
                    self.__dataset_audio_vad.append(join_path(self.dir_va_names[1]))
                    if not os.path.exists(self.__dataset_audio_vad[-1]):
                        # Директория не создана
                        if self.create_folder(self.__dataset_audio_vad[-1], out=False) is False:
                            raise IsNestedDirectoryANotFoundError
            except (IsNestedDirectoryVNotFoundError, IsNestedDirectoryANotFoundError):
                self.__unprocessed_files.append(self.__curr_path)
                return False
            except Exception:
                self.__unprocessed_files.append(self.__curr_path)
                return False

        # Проход по всем каналам
        for channel in range(0, channels_audio):
            # Распознавание речи по видео
            if type(self.__subprocess_vosk_sr) is list:
                # Проход по всем найденным меткам
                for cnt, curr_timestamps in enumerate(self.__subprocess_vosk_sr):
                    res_vosk_sr = curr_timestamps[0].lower()

                    if res_vosk_sr == "":
                        continue  # Речь не найдена

                    # Начальное время
                    start_time = timedelta(seconds=curr_timestamps[1]) - timedelta(
                        milliseconds=self.__speech_left_pad_ms
                    )
                    # Конечное время
                    end_time = timedelta(seconds=curr_timestamps[2]) + timedelta(
                        milliseconds=self.__speech_right_pad_ms
                    )

                    if start_time < timedelta(seconds=0) is True:
                        start_time = timedelta(seconds=0)

                    diff_time = end_time - start_time  # Разница между начальным и конечным временем

                    # Путь до аудиофрагмента
                    self.__part_audio_path = os.path.join(
                        self.__dataset_audio_vad[-1],
                        Path(self.__curr_path).stem
                        + "_"
                        + str(cnt)
                        + self.__front[channel]
                        + "_"
                        + self.__curr_ts
                        + "."
                        + EXT_AUDIO,
                    )

                    # Видео
                    if kind.mime.startswith("video/") is True and channel == 0:
                        # Путь до видеофрагмента
                        self.__part_video_path = os.path.join(
                            self.__dataset_video_vad[-1],
                            Path(self.__curr_path).stem
                            + "_"
                            + str(cnt)
                            + "_"
                            + self.__curr_ts
                            + Path(self.__curr_path).suffix.lower(),
                        )

                    def not_saved_files():
                        return self.__not_saved_files.append([self.__curr_path, start_time, end_time])

                    call_audio, call_video = 0, 0

                    try:
                        # Видео
                        if kind.mime.startswith("video/") is True:
                            if channel == 0:
                                # Варианты кодирования
                                if self.__type_encode == TYPES_ENCODE[0]:
                                    # https://trac.ffmpeg.org/wiki/Encode/MPEG-4
                                    ff_v = 'ffmpeg -loglevel quiet -ss {} -i "{}" -{} 0 -to {} "{}"'.format(
                                        start_time,
                                        self.__curr_path,
                                        self.__type_encode,
                                        diff_time,
                                        self.__part_video_path,
                                    )
                                elif self.__type_encode == TYPES_ENCODE[1]:
                                    # https://trac.ffmpeg.org/wiki/Encode/H.264
                                    ff_v = 'ffmpeg -loglevel quiet -ss {} -i "{}" -{} {} -preset {} -to {} "{}"'.format(
                                        start_time,
                                        self.__curr_path,
                                        self.__type_encode,
                                        self.__crf_value,
                                        self.__presets_crf_encode,
                                        diff_time,
                                        self.__part_video_path,
                                    )
                            if channels_audio == 1:  # Моно канал
                                ff_a = (
                                    # 'ffmpeg -loglevel quiet -i "{}" -vn -codec:a copy '
                                    'ffmpeg -loglevel quiet -i "{}" -vn -codec:a pcm_f32le '
                                    '-ac {} -ss {} -to {} "{}"'.format(
                                        self.__curr_path, channels_audio, start_time, end_time, self.__part_audio_path
                                    )
                                )
                            elif channels_audio == 2:  # Стерео канал
                                ff_a = (
                                    'ffmpeg -loglevel quiet -i "{}" -vn -codec:a pcm_f32le -map_channel 0.1.{} -ss {} '
                                    '-to {} "{}"'.format(
                                        self.__curr_path, channel, start_time, end_time, self.__part_audio_path
                                    )
                                )

                        # Аудио
                        if kind.mime.startswith("audio/") is True:
                            if channels_audio == 1:  # Моно канал
                                ff_a = 'ffmpeg -loglevel quiet -i "{}" -ss {} -to {} -c copy "{}"'.format(
                                    self.__curr_path, start_time, end_time, self.__part_audio_path
                                )
                            elif channels_audio == 2:  # Стерео канал
                                ff_a = 'ffmpeg -loglevel quiet -i "{}" -map_channel 0.0.{} -ss {} -to {} "{}"'.format(
                                    self.__curr_path, channel, start_time, end_time, self.__part_audio_path
                                )
                    except IndexError:
                        not_saved_files()
                    except Exception:
                        not_saved_files()
                    else:
                        # Видео
                        if kind.mime.startswith("video/") is True and channel == 0:
                            call_video = subprocess.call(ff_v, shell=True)
                        # Аудио
                        call_audio = subprocess.call(ff_a, shell=True)

                        try:
                            if call_audio == 1 or call_video == 1:
                                raise OSError
                        except OSError:
                            not_saved_files()
                        except Exception:
                            not_saved_files()
                        else:
                            try:
                                # Видео
                                if kind.mime.startswith("video/") is True:
                                    # Чтение файла
                                    _, _, _ = torchvision.io.read_video(self.__part_video_path)
                                _, _ = torchaudio.load(self.__part_audio_path)
                            except Exception:
                                not_saved_files()

        # Распознавание речи по аудио
        if type(self.__subprocess_vosk_sr) is dict:
            print("Тестирование")

            # Проход по всем найденным меткам
            for key, val in enumerate(self.__subprocess_vosk_sr.items()):
                # Проход по всем найденным меткам
                for cnt, curr_timestamps in enumerate(val[1]):
                    res_vosk_sr = curr_timestamps[0].lower()

                    if res_vosk_sr == "":
                        continue  # Речь не найдена

                    start_time = timedelta(seconds=curr_timestamps[1])  # Начальное время
                    end_time = timedelta(seconds=curr_timestamps[2])  # Конечное время

                    diff_time = end_time - start_time  # Разница между начальным и конечным временем

                    # Путь до аудиофрагмента
                    self.__part_audio_path = os.path.join(
                        self.__dataset_audio_vad[-1],
                        Path(self.__curr_path).stem
                        + "_"
                        + str(cnt)
                        + self.__front[key]
                        + "_"
                        + self.__curr_ts
                        + "."
                        + EXT_AUDIO,
                    )

                    def not_saved_files():
                        return self.__not_saved_files.append([self.__curr_path, start_time, end_time])

                    call_audio = 0

                    try:
                        # Аудио
                        if kind.mime.startswith("audio/") is True:
                            if channels_audio == 1:  # Моно канал
                                ff_a = 'ffmpeg -loglevel quiet -i "{}" -ss {} -to {} -c copy "{}"'.format(
                                    self.__curr_path, start_time, end_time, self.__part_audio_path
                                )
                            elif channels_audio == 2:  # Стерео канал
                                ff_a = 'ffmpeg -loglevel quiet -i "{}" -map_channel 0.0.{} -ss {} -to {} "{}"'.format(
                                    self.__curr_path, key, start_time, end_time, self.__part_audio_path
                                )
                    except IndexError:
                        not_saved_files()
                    except Exception:
                        not_saved_files()
                    else:
                        # Аудио
                        call_audio = subprocess.call(ff_a, shell=True)

                        try:
                            if call_audio == 1:
                                raise OSError
                        except OSError:
                            not_saved_files()
                        except Exception:
                            not_saved_files()
                        else:
                            try:
                                _, _ = torchaudio.load(self.__part_audio_path)
                            except Exception:
                                not_saved_files()

        return True

    def __audio_augmentation(self) -> bool:
        """Аугментация аудиофайла

        Returns:
            bool: **True** если аугментация аудиофайла произведена, в обратном случае **False**
        """

        # Тип файла
        kind = filetype.guess(self.__curr_path)

        aug_audio, sample_rate = audags.change_volume(self.__curr_path, volume_db=self.__volume_db_aug)

        aug_audio, sample_rate = audags.low_pass_filter(
            aug_audio,
            sample_rate=sample_rate,
            cutoff_hz=self.__cutoff_hz_aug,
        )

        self.__curr_ts = str(datetime.now().timestamp()).replace(".", "_")

        path = os.path.join(
            self.__dataset_audio_vad[-1],
            Path(self.__curr_path).stem
            + "_"
            + self.__curr_ts
            + "."
            + EXT_AUDIO_AUG,
        )

        aug_audio, sample_rate = audags.normalize(
            aug_audio,
            sample_rate=sample_rate,
            output_path=path
        )
        return True

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def vad(
        self,
        depth: int = 1,
        type_encode: str = TYPES_ENCODE[1],
        crf_value: int = CRF_VALUE,
        presets_crf_encode: str = PRESETS_CRF_ENCODE[5],
        sr_input_type: str = SR_INPUT_TYPES[0],
        sampling_rate: int = SAMPLING_RATE_VAD[1],
        threshold: float = THRESHOLD_VAD,
        min_speech_duration_ms: int = MIN_SPEECH_DURATION_MS_VAD,
        min_silence_duration_ms: int = MIN_SILENCE_DURATION_MS_VAD,
        window_size_samples: int = WINDOW_SIZE_SAMPLES_VAD[SAMPLING_RATE_VAD[1]][2],
        speech_pad_ms: int = SPEECH_PAD_MS,
        force_reload: bool = True,
        clear_dirvad: bool = False,
        out: bool = True,
    ) -> bool:
        """VAD (Voice Activity Detector) или (детектирование голосовой активности)

        Args:
            depth (int): Глубина иерархии для получения данных
            type_encode (str): Тип кодирования
            crf_value (int): Качество кодирования (от **0** до **51**)
            presets_crf_encode (str): Скорость кодирования и сжатия
            sr_input_type (str): Тип файлов для распознавания речи
            sampling_rate (int): Частота дискретизации (**8000** или **16000**)
            threshold (float): Порог вероятности речи (от **0.0** до **1.0**)
            min_speech_duration_ms (int): Минимальная длительность речевого фрагмента в миллисекундах
            min_silence_duration_ms (int): Минимальная длительность тишины в выборках между отдельными речевыми
                                           фрагментами
            window_size_samples (int): Количество выборок в каждом окне (**512**, **1024**, **1536** для частоты
                                       дискретизации **16000** или **256**, **512**, **768** для частоты дискретизации
                                       **8000**)
            speech_pad_ms (int): Внутренние отступы для итоговых речевых фрагментов
            force_reload (bool): Принудительная загрузка модели из сети
            clear_dirvad (bool): Очистка директории для сохранения фрагментов аудиовизуального сигнала
            out (bool): Отображение

        Returns:
            bool: **True** если детектирование голосовой активности произведено, в обратном случае **False**

        .. versionadded:: 0.1.0

        .. versionchanged:: 0.1.1

        .. deprecated:: 0.1.0
        """

        try:
            # Проверка аргументов
            if (
                type(depth) is not int
                or depth < 1
                or type(crf_value) is not int
                or not (0 <= crf_value <= 51)
                or type(threshold) is not float
                or not (0.0 <= threshold <= 1.0)
                or type(min_speech_duration_ms) is not int
                or min_speech_duration_ms < 1
                or type(min_silence_duration_ms) is not int
                or min_silence_duration_ms < 1
                or type(speech_pad_ms) is not int
                or speech_pad_ms < 1
                or type(force_reload) is not bool
                or type(clear_dirvad) is not bool
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self.inv_args(__class__.__name__, self.vad.__name__, out=out)
            return False
        else:
            try:
                # Проверка настроек
                if type(type_encode) is not str or (type_encode in TYPES_ENCODE) is False:
                    raise TypeEncodeVideoError
                if type(presets_crf_encode) is not str or (presets_crf_encode in PRESETS_CRF_ENCODE) is False:
                    raise PresetCFREncodeVideoError
                if type(sr_input_type) is not str or (sr_input_type in [x.lower() for x in SR_INPUT_TYPES]) is False:
                    raise SRInputTypeError
                if type(sampling_rate) is not int or (sampling_rate in [x for x in SAMPLING_RATE_VAD]) is False:
                    raise SamplingRateError
                if (
                    type(window_size_samples) is not int
                    or (window_size_samples in [x for x in WINDOW_SIZE_SAMPLES_VAD[sampling_rate]]) is False
                ):
                    raise WindowSizeSamplesError
            except TypeEncodeVideoError:
                self.message_error(
                    self._wrong_type_encode.format(
                        self.message_line(", ".join(x.replace(".", "") for x in TYPES_ENCODE))
                    ),
                    out=out,
                )
                return False
            except PresetCFREncodeVideoError:
                self.message_error(
                    self._wrong_preset_crf_encode.format(
                        self.message_line(", ".join(x.replace(".", "") for x in PRESETS_CRF_ENCODE))
                    ),
                    out=out,
                )
                return False
            except SRInputTypeError:
                self.message_error(
                    self._wrong_sr_input_type.format(
                        self.message_line(", ".join(x.replace(".", "") for x in SR_INPUT_TYPES))
                    ),
                    out=out,
                )
                return False
            except SamplingRateError:
                self.message_error(
                    self._wrong_sampling_rate_vad.format(
                        self.message_line(", ".join(str(x) for x in SAMPLING_RATE_VAD))
                    ),
                    out=out,
                )
                return False
            except WindowSizeSamplesError:
                self.message_error(
                    self._wrong_window_size_samples_type.format(
                        self.message_line(str(sampling_rate)),
                        self.message_line(", ".join(str(x) for x in WINDOW_SIZE_SAMPLES_VAD[sampling_rate])),
                    ),
                    out=out,
                )
                return False
            else:
                # Только для внутреннего использования внутри класса
                self.__type_encode = type_encode
                self.__crf_value = crf_value
                self.__presets_crf_encode = presets_crf_encode
                self.__sr_input_type = sr_input_type
                self.__sampling_rate_vad = sampling_rate
                self.__threshold_vad = threshold
                self.__min_speech_duration_ms_vad = min_speech_duration_ms
                self.__min_silence_duration_ms_vad = min_silence_duration_ms
                self.__window_size_samples_vad = window_size_samples
                self.__speech_pad_ms_vad = speech_pad_ms
                # Метаданные для видео и аудио
                self.__file_metadata["video_fps"], self.__file_metadata["audio_fps"] = 0.0, 0

                torch.set_num_threads(1)  # Установка количества потоков для внутриоперационного параллелизма на ЦП

                torch.hub.set_dir(self.path_to_save_models)  # Установка пути к директории для сохранения VAD модели

                # Информационное сообщение
                self.message_info(
                    self._download_model_from_repo.format(
                        self.message_line(self._vad_model),
                        urllib.parse.urljoin("https://github.com/", self._github_repo_vad),
                    ),
                    out=out,
                )

                try:
                    # Подавление вывода
                    with io.capture_output():
                        # Загрузка VAD модели
                        self.__model_vad, utils = torch.hub.load(
                            repo_or_dir=self._github_repo_vad, model=self._vad_model, force_reload=force_reload
                        )
                except FileNotFoundError:
                    self.message_error(
                        self._clear_folder_not_found.format(self.message_line(self.path_to_save_models)),
                        space=self._space,
                        out=out,
                    )
                    return False
                except RuntimeError:
                    self.message_error(self._url_error.format(""), space=self._space, out=out)
                    return False
                except urllib.error.HTTPError as e:
                    self.message_error(
                        self._url_error.format(self._url_error_code.format(self.message_line(str(e.code)))),
                        space=self._space,
                        out=out,
                    )
                    return False
                except urllib.error.URLError:
                    self.message_error(self._url_error.format(""), space=self._space, out=out)
                    return False
                except Exception:
                    self.message_error(self._unknown_err, space=self._space, out=out)
                    return False
                else:
                    self.__get_speech_ts, _, read_audio, _, _ = utils

                    # Информационное сообщение
                    self.message_info(
                        self._subfolders_search.format(
                            self.message_line(self.path_to_dataset),
                            self.message_line(str(depth)),
                        ),
                        out=out,
                    )

                    # Создание директории, где хранятся данные
                    if self.create_folder(self.path_to_dataset, out=False) is False:
                        return False

                    # Получение вложенных директорий, где хранятся данные
                    nested_paths = self.get_paths(self.path_to_dataset, depth=depth, out=False)

                    # Вложенные директории не найдены
                    try:
                        if len(nested_paths) == 0:
                            raise IsNestedCatalogsNotFoundError
                    except IsNestedCatalogsNotFoundError:
                        self.message_error(self._subfolders_not_found, space=self._space, out=out)
                        return False

                    # Информационное сообщение
                    self.message_info(
                        self._files_av_find.format(
                            self.message_line(", ".join(x.replace(".", "") for x in self.ext_search_files)),
                            self.message_line(self.path_to_dataset),
                            self.message_line(str(depth)),
                        ),
                        out=out,
                    )

                    paths = []  # Пути до аудиовизуальных файлов

                    # Проход по всем вложенным директориям
                    for nested_path in nested_paths:
                        # Формирование списка с видеофайлами
                        for p in Path(nested_path).glob("*"):
                            # Добавление текущего пути к видеофайлу в список
                            if p.suffix.lower() in self.ext_search_files:
                                paths.append(p.resolve())

                    # Директория с набором данных не содержит аудиовизуальных файлов с необходимыми расширениями
                    try:
                        self.__len_paths = len(paths)  # Количество аудиовизуальных файлов

                        if self.__len_paths == 0:
                            raise TypeError
                    except TypeError:
                        self.message_error(self._files_not_found, space=self._space, out=out)
                        return False
                    except Exception:
                        self.message_error(self._unknown_err, space=self._space, out=out)
                        return False
                    else:
                        # Очистка директории для сохранения фрагментов аудиовизуального сигнала
                        if clear_dirvad is True and os.path.exists(self.path_to_dataset_vad) is True:
                            if self.clear_folder(self.path_to_dataset_vad, out=False) is False:
                                return False

                        self.__dataset_video_vad = []  # Пути до директорий с разделенными видеофрагментами
                        self.__dataset_audio_vad = []  # Пути до директорий с разделенными аудиофрагментами

                        self.__unprocessed_files = []  # Пути к файлам на которых VAD не отработал

                        # Информационное сообщение
                        self.message_info(self._files_analysis, out=out)

                        # Локальный путь
                        self.__local_path = lambda lp: os.path.join(
                            *Path(lp).parts[-abs((len(Path(lp).parts) - len(Path(self.path_to_dataset).parts))) :]
                        )

                        # Проход по всем найденным аудиовизуальных файлам
                        for i, path in enumerate(paths):
                            self.__curr_path = path  # Текущий аудиовизуальный файл
                            self.__i = i + 1  # Счетчик

                            self.message_progressbar(
                                self._curr_progress.format(
                                    self.__i,
                                    self.__len_paths,
                                    round(self.__i * 100 / self.__len_paths, 2),
                                    self.message_line(self.__local_path(self.__curr_path)),
                                ),
                                space=self._space,
                                out=out,
                            )

                            self.__splitted_path = str(
                                self.__curr_path.parent.relative_to(Path(self.path_to_dataset))
                            ).strip()

                            self.__curr_path = str(self.__curr_path)

                            # Пропуск невалидных значений
                            if not self.__splitted_path or re.search(r"\s", self.__splitted_path) is not None:
                                continue

                            # Тип файла
                            kind = filetype.guess(self.__curr_path)

                            try:
                                # Видео
                                if kind.mime.startswith("video/") is True:
                                    # Чтение файла
                                    _, self.__aframes, self.__file_metadata = torchvision.io.read_video(
                                        self.__curr_path
                                    )

                                # Аудио
                                if kind.mime.startswith("audio/") is True:
                                    (self.__aframes, self.__file_metadata["audio_fps"]) = torchaudio.load(
                                        self.__curr_path
                                    )
                            except Exception:
                                self.__unprocessed_files.append(self.__curr_path)
                                self.message_progressbar(close=True, out=out)
                                continue
                            else:
                                # Аудио
                                if kind.mime.startswith("audio/") is True:
                                    self.__aframes = self.__aframes.to(torch.float32)

                                self.__audio_analysis()  # Анализ аудиодорожки

                        self.message_progressbar(close=True, out=out)

                        # Файлы на которых VAD не отработал
                        unprocessed_files_unique = np.unique(np.array(self.__unprocessed_files)).tolist()

                        if len(unprocessed_files_unique) == 0 and len(self.__not_saved_files) == 0:
                            self.message_true(self._vad_true, space=self._space, out=out)
                            return True

    def vosk(self, new_name: Optional[str] = None, force_reload: bool = True, out: bool = True) -> bool:
        """Загрузка и активация модели Vosk для детектирования голосовой активности и распознавания речи

        Args:
            new_name (str): Имя директории для разархивирования
            force_reload (bool): Принудительная загрузка модели из сети
            out (bool) Отображение

        Returns:
            bool: **True** если модель Vosk загружена и активирована, в обратном случае **False**
        """

        try:
            # Проверка аргументов
            if (
                ((type(new_name) is not str or not new_name) and new_name is not None)
                or type(force_reload) is not bool
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self.inv_args(__class__.__name__, self.vosk.__name__, out=out)
            return False
        else:
            name = "vosk"  # Модель для распознавания речи

            SetLogLevel(-1)  # Уровень LOG

            lsr = self.vosk_language_sr  # Язык для распознавания речи
            dlsr = self.vosk_dict_language_sr  # Размер словаря для распознавания речи

            url = urllib.parse.urljoin(self._vosk_models_url[name], self._vosk_models_for_sr[name][lsr][dlsr])

            try:
                # Загрузка файла из URL
                res_download_file_from_url = self.download_file_from_url(url=url, force_reload=force_reload, out=out)
            except Exception:
                self.message_error(self._unknown_err, start=True, space=self._space, out=out)
                return False
            else:
                # Файл загружен
                if res_download_file_from_url == 200:
                    try:
                        # Распаковка архива
                        res_unzip = self.unzip(
                            path_to_zipfile=os.path.join(
                                self.path_to_save_models, self._vosk_models_for_sr[name][lsr][dlsr]
                            ),
                            new_name=new_name,
                            force_reload=force_reload,
                        )
                    except Exception:
                        self.message_error(self._unknown_err, start=True, out=out)
                        return False
                    else:
                        # Файл распакован
                        if res_unzip is True:
                            try:
                                # Информационное сообщение
                                self.message_info(
                                    self._vosk_model_activation.format(
                                        self.message_line(Path(self._vosk_models_for_sr[name][lsr][dlsr]).stem)
                                    ),
                                    start=True,
                                    out=out,
                                )

                                self.__speech_model = Model(str(self._path_to_unzip))  # Активация модели
                                # Активация распознавания речи
                                self.__speech_rec = KaldiRecognizer(self.__speech_model, self.__freq_sr)
                                self.__speech_rec.SetWords(True)  # Данные о начале и конце слова/фразы
                            except Exception:
                                self.message_error(self._unknown_err, out=out)
                            else:
                                return True
                else:
                    return False

    def vosk_sr(
        self,
        depth: int = 1,
        type_encode: str = TYPES_ENCODE[1],
        crf_value: int = CRF_VALUE,
        presets_crf_encode: str = PRESETS_CRF_ENCODE[5],
        new_name: Optional[str] = None,
        speech_left_pad_ms: int = VOSK_SPEECH_LEFT_PAD_MS,
        speech_right_pad_ms: int = VOSK_SPEECH_RIGHT_PAD_MS,
        force_reload: bool = True,
        clear_dirvosk_sr: bool = False,
        out: bool = True,
    ) -> bool:
        """VAD + SR (Voice Activity Detector + Speech Recognition) или (детектирование голосовой активности и
        распознавание речи)

        Args:
            depth (int): Глубина иерархии для получения данных
            type_encode (str): Тип кодирования
            crf_value (int): Качество кодирования (от **0** до **51**)
            presets_crf_encode (str): Скорость кодирования и сжатия
            new_name (str): Имя директории для разархивирования
            speech_left_pad_ms (int): Внутренний левый отступ для итоговых речевых фрагментов
            speech_right_pad_ms (int): Внутренний правый отступ для итоговых речевых фрагментов
            force_reload (bool): Принудительная загрузка модели из сети
            clear_dirvosk_sr (bool): Очистка директории для сохранения фрагментов аудиовизуального сигнала
            out (bool) Отображение

        Returns:
            bool: **True** если детектирование голосовой активности и распознавание речи произведено, в обратном случае
            **False**
        """

        try:
            # Проверка аргументов
            if (
                type(depth) is not int
                or depth < 1
                or type(crf_value) is not int
                or not (0 <= crf_value <= 51)
                or ((type(new_name) is not str or not new_name) and new_name is not None)
                or type(speech_left_pad_ms) is not int
                or speech_left_pad_ms < 0
                or type(speech_right_pad_ms) is not int
                or speech_right_pad_ms < 0
                or type(force_reload) is not bool
                or type(clear_dirvosk_sr) is not bool
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self.inv_args(__class__.__name__, self.vosk_sr.__name__, out=out)
            return False
        else:
            try:
                # Проверка настроек
                if type(type_encode) is not str or (type_encode in TYPES_ENCODE) is False:
                    raise TypeEncodeVideoError
                if type(presets_crf_encode) is not str or (presets_crf_encode in PRESETS_CRF_ENCODE) is False:
                    raise PresetCFREncodeVideoError

            except TypeEncodeVideoError:
                self.message_error(
                    self._wrong_type_encode.format(
                        self.message_line(", ".join(x.replace(".", "") for x in TYPES_ENCODE))
                    ),
                    out=out,
                )
                return False
            except PresetCFREncodeVideoError:
                self.message_error(
                    self._wrong_preset_crf_encode.format(
                        self.message_line(", ".join(x.replace(".", "") for x in PRESETS_CRF_ENCODE))
                    ),
                    out=out,
                )
                return False
            else:
                # Только для внутреннего использования внутри класса
                self.__type_encode = type_encode
                self.__crf_value = crf_value
                self.__presets_crf_encode = presets_crf_encode

                self.__speech_left_pad_ms = speech_left_pad_ms
                self.__speech_right_pad_ms = speech_right_pad_ms

                # Информационное сообщение
                self.message_info(
                    self._subfolders_search.format(
                        self.message_line(self.path_to_dataset),
                        self.message_line(str(depth)),
                    ),
                    out=out,
                )

                # Создание директории, где хранятся данные
                if self.create_folder(self.path_to_dataset, out=False) is False:
                    return False

                # Получение вложенных директорий, где хранятся данные
                nested_paths = self.get_paths(self.path_to_dataset, depth=depth, out=False)

                # Вложенные директории не найдены
                try:
                    if len(nested_paths) == 0:
                        raise IsNestedCatalogsNotFoundError
                except IsNestedCatalogsNotFoundError:
                    self.message_error(self._subfolders_not_found, space=self._space, out=out)
                    return False

                # Информационное сообщение
                self.message_info(
                    self._files_av_find.format(
                        self.message_line(", ".join(x.replace(".", "") for x in self.ext_search_files)),
                        self.message_line(self.path_to_dataset),
                        self.message_line(str(depth)),
                    ),
                    out=out,
                )

                paths = []  # Пути до аудиовизуальных файлов

                # Проход по всем вложенным директориям
                for nested_path in nested_paths:
                    # Формирование списка с видеофайлами
                    for p in Path(nested_path).glob("*"):
                        # Добавление текущего пути к видеофайлу в список
                        if p.suffix.lower() in self.ext_search_files:
                            paths.append(p.resolve())

                # Директория с набором данных не содержит аудиовизуальных файлов с необходимыми расширениями
                try:
                    self.__len_paths = len(paths)  # Количество аудиовизуальных файлов

                    if self.__len_paths == 0:
                        raise TypeError
                except TypeError:
                    self.message_error(self._files_not_found, space=self._space, out=out)
                    return False
                except Exception:
                    self.message_error(self._unknown_err, space=self._space, out=out)
                    return False
                else:
                    # Очистка директории для сохранения фрагментов аудиовизуального сигнала
                    if clear_dirvosk_sr is True and os.path.exists(self.path_to_dataset_vosk_sr) is True:
                        if self.clear_folder(self.path_to_dataset_vosk_sr, out=False) is False:
                            return False

                    self.__dataset_video_vad = []  # Пути до директорий с разделенными видеофрагментами
                    self.__dataset_audio_vad = []  # Пути до директорий с разделенными аудиофрагментами

                    self.__unprocessed_files = []  # Пути к файлам на которых VAD не отработал

                    # Загрузка и активация модели Vosk для распознавания речи
                    if self.vosk(new_name=new_name, force_reload=force_reload, out=out) is False:
                        return False

                    # Информационное сообщение
                    self.message_info(self._files_analysis, out=out)

                    # Локальный путь
                    self.__local_path = lambda lp: os.path.join(
                        *Path(lp).parts[-abs((len(Path(lp).parts) - len(Path(self.path_to_dataset).parts))) :]
                    )

                    # Проход по всем найденным аудиовизуальных файлам
                    for i, path in enumerate(paths):
                        self.__curr_path = path  # Текущий аудиовизуальный файл
                        self.__i = i + 1  # Счетчик

                        self.message_progressbar(
                            self._curr_progress.format(
                                self.__i,
                                self.__len_paths,
                                round(self.__i * 100 / self.__len_paths, 2),
                                self.message_line(self.__local_path(self.__curr_path)),
                            ),
                            space=self._space,
                            out=out,
                        )

                        self.__splitted_path = str(
                            self.__curr_path.parent.relative_to(Path(self.path_to_dataset))
                        ).strip()

                        self.__curr_path = str(self.__curr_path)

                        # Пропуск невалидных значений
                        if not self.__splitted_path or re.search(r"\s", self.__splitted_path) is not None:
                            continue

                        # Тип файла
                        kind = filetype.guess(self.__curr_path)

                        try:
                            # Активация распознавания речи
                            self.__speech_rec = KaldiRecognizer(self.__speech_model, self.__freq_sr)
                            self.__speech_rec.SetWords(True)  # Данные о начале и конце слова/фразы
                        except Exception:
                            continue
                        else:
                            try:
                                # Видео
                                if kind.mime.startswith("video/") is True:
                                    #  Дочерний процесс распознавания речи (Vosk) - видео
                                    self.__subprocess_vosk_sr = self.__subprocess_vosk_sr_video(out=False)
                                # Аудио
                                if kind.mime.startswith("audio/") is True:
                                    #  Дочерний процесс распознавания речи (Vosk) - аудио
                                    self.__subprocess_vosk_sr = self.__subprocess_vosk_sr_audio(out=False)
                            except Exception:
                                self.__unprocessed_files.append(self.__curr_path)
                                self.message_progressbar(close=True, out=out)
                                continue
                            else:
                                self.__audio_analysis_vosk_sr()  # Анализ аудиодорожки

                    self.message_progressbar(close=True, out=out)

                    # Файлы на которых VAD не отработал
                    unprocessed_files_unique = np.unique(np.array(self.__unprocessed_files)).tolist()

                    if len(unprocessed_files_unique) == 0 and len(self.__not_saved_files) == 0:
                        self.message_true(self._vad_true, space=self._space, out=out)

    def augmentation(
        self,
        depth: int = 1,
        volume_db: float = VOLUME_DB_AUGMENTATION[1],
        cutoff_hz: float = CUTOFF_HZ_AUGMENTATION[1],
        clear_diraug: bool = False,
        out: bool = True,
    ) -> bool:
        """Аугментация аудиовизуальных сигналов

        Args:
            depth (int): Глубина иерархии для получения данных
            volume_db (float): TODO: fix description (от **0.0** до **1000.0**)
            cutoff_hz (float): TODO: fix description (от **0.0** до **1000.0**)
            clear_diraug (bool): Очистка директории для сохранения аугментированных аудиовизуальных сигналов
            out (bool): Отображение

        Returns:
            bool: **True** если аугментация аудиовизуальных сигналов произведено, в обратном случае **False**

        .. versionadded:: 0.1.0

        .. versionchanged:: 0.1.1

        .. deprecated:: 0.1.0
        """

        try:
            # Проверка аргументов
            if (
                type(depth) is not int
                or depth < 1
                or type(clear_diraug) is not bool
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self.inv_args(__class__.__name__, self.vad.__name__, out=out)
            return False
        else:
            try:
                # Проверка настроек
                if type(volume_db) is not float or (0 <= volume_db <= 1000) is False:
                    raise VolumeDbError
                if type(cutoff_hz) is not float or (0 <= cutoff_hz <= 1000) is False:
                    raise CutoffHzError
            except VolumeDbError:
                self.message_error(
                    self._wrong_volume_db_aug.format(
                        self.message_line(" - ".join(str(x) for x in VOLUME_DB_AUGMENTATION))
                    ),
                    out=out,
                )
                return False
            except CutoffHzError:
                self.message_error(
                    self._wrong_cutoff_hz_aug.format(
                        self.message_line(" - ".join(str(x) for x in CUTOFF_HZ_AUGMENTATION))
                    ),
                    out=out,
                )
                return False
            else:
                # Только для внутреннего использования внутри класса
                self.__volume_db_aug = volume_db
                self.__cutoff_hz_aug = cutoff_hz
                # Метаданные для видео и аудио
                self.__file_metadata["video_fps"], self.__file_metadata["audio_fps"] = 0.0, 0

                # Информационное сообщение
                self.message_info(
                    self._subfolders_search.format(
                        self.message_line(self.path_to_input_augmentation_directory),
                    ),
                    out=out,
                )

                # Создание директории, где хранятся данные
                if self.create_folder(self.path_to_input_augmentation_directory, out=False) is False:
                    return False

                # Получение вложенных директорий, где хранятся данные
                nested_paths = self.get_paths(self.path_to_input_augmentation_directory, depth=depth, out=False)

                # Вложенные директории не найдены
                try:
                    if len(nested_paths) == 0:
                        raise IsNestedCatalogsNotFoundError
                except IsNestedCatalogsNotFoundError:
                    self.message_error(self._subfolders_not_found, space=self._space, out=out)
                    return False

                # Информационное сообщение
                self.message_info(
                    self._files_av_find.format(
                        self.message_line(", ".join(x.replace(".", "") for x in self.ext_search_files)),
                        self.message_line(self.path_to_input_augmentation_directory),
                        self.message_line(str(depth)),
                    ),
                    out=out,
                )

                paths = []  # Пути до аудиовизуальных файлов

                # Проход по всем вложенным директориям
                for nested_path in nested_paths:
                    # Формирование списка с видеофайлами
                    for p in Path(nested_path).glob("*"):
                        # Добавление текущего пути к видеофайлу в список
                        if p.suffix.lower() in self.ext_search_files:
                            paths.append(p.resolve())

                # Директория с набором данных не содержит аудиовизуальных файлов с необходимыми расширениями
                try:
                    self.__len_paths = len(paths)  # Количество аудиовизуальных файлов

                    if self.__len_paths == 0:
                        raise TypeError
                except TypeError:
                    self.message_error(self._files_not_found, space=self._space, out=out)
                    return False
                except Exception:
                    self.message_error(self._unknown_err, space=self._space, out=out)
                    return False
                else:
                    # Очистка директории для сохранения обработанных аудиовизуальных сигналов
                    if clear_diraug is True and os.path.exists(self.path_to_output_augmentation_directory) is True:
                        if self.clear_folder(self.path_to_output_augmentation_directory, out=False) is False:
                            return False

                    self.__output_augmentation_audio = []  # Пути до директорий с аугментированными аудиофрагментами

                    self.__unprocessed_files = []  # Пути к файлам на которых аугментация не отработала

                    # Информационное сообщение
                    self.message_info(self._files_analysis, out=out)

                    # Локальный путь
                    self.__local_path = lambda lp: os.path.join(
                        *Path(lp).parts[-abs((len(Path(lp).parts) - len(Path(self.path_to_input_augmentation_directory).parts))) :]
                    )

                    # Проход по всем найденным аудиовизуальных файлам
                    for i, path in enumerate(paths):
                        self.__curr_path = path  # Текущий аудиовизуальный файл
                        self.__i = i + 1  # Счетчик

                        self.message_progressbar(
                            self._curr_progress.format(
                                self.__i,
                                self.__len_paths,
                                round(self.__i * 100 / self.__len_paths, 2),
                                self.message_line(self.__local_path(self.__curr_path)),
                            ),
                            space=self._space,
                            out=out,
                        )

                        self.__splitted_path = str(
                            self.__curr_path.parent.relative_to(Path(self.path_to_input_augmentation_directory))
                        ).strip()

                        self.__curr_path = str(self.__curr_path)

                        # Пропуск невалидных значений
                        if not self.__splitted_path or re.search(r"\s", self.__splitted_path) is not None:
                            continue

                        try:
                            self.__audio_augmentation()
                        except Exception:
                            self.__unprocessed_files.append(self.__curr_path)
                            self.message_progressbar(close=True, out=out)
                            continue

                    self.message_progressbar(close=True, out=out)

                    # Файлы на которых аугментация не отработала
                    unprocessed_files_unique = np.unique(np.array(self.__unprocessed_files)).tolist()

                    if len(unprocessed_files_unique) == 0 and len(self.__not_saved_files) == 0:
                        self.message_true(self._aug_true, space=self._space, out=out)
                        return True