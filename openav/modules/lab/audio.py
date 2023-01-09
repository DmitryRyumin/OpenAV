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
for warn in [UserWarning, FutureWarning]: warnings.filterwarnings('ignore', category = warn)

from dataclasses import dataclass # Класс данных

import os           # Взаимодействие с файловой системой
import re           # Регулярные выражения
import torch        # Машинное обучение от Facebook
import torchvision  # Работа с видео от Facebook
import urllib.parse # Парсинг URL
import filetype     # Определение типа файла и типа MIME

from IPython.utils import io        # Подавление вывода
from pathlib import Path, PosixPath # Работа с путями в файловой системе
from datetime import timedelta      # Работа со временем

# Типы данных
from typing import List, Dict, Union, Optional

from types import FunctionType

# Персональные
from openav.modules.core.exceptions import TypeEncodeVideoError, PresetCFREncodeVideoError, SRInputTypeError, \
                                           IsNestedCatalogsNotFoundError, IsNestedDirectoryVNotFoundError, \
                                           IsNestedDirectoryANotFoundError, SamplingRateError
from openav.modules.file_manager.json_manager import Json # Класс для работы с Json

# ######################################################################################################################
# Константы
# ######################################################################################################################
TYPES_ENCODE: List[str] = ['qscale', 'crf'] # Типы кодирования
# Скорость кодирования и сжатия
PRESETS_CRF_ENCODE: List[str] = [
    'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'
]
SR_INPUT_TYPES: List[str] = ['audio', 'video'] # Типы файлов для распознавания речи
THRESHOLD_VAD: float = 0.56 # Порог вероятности речи (от 0.0 до 1.0)
SAMPLING_RATE_VAD: int = [8000, 16000] # Частота дискретизации
MIN_SPEECH_DURATION_MS_VAD: int = 250 # Минимальная длительность речевого фрагмента в миллисекундах
MIN_SILENCE_DURATION_MS_VAD: int = 50 # Минимальная длительность тишины в выборках между отдельными речевыми фрагментами
# Количество выборок в каждом окне
# (512, 1024, 1536 для частоты дискретизации 16000 или 256, 512, 768 для частоты дискретизации 8000)
WINDOW_SIZE_SAMPLES_VAD: int = 1536
SPEECH_PAD_MS: int = 150 # Внутренние отступы для итоговых речевых фрагментов

# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class  AudioMessages(Json):
    """Класс для сообщений"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__() # Выполнение конструктора из суперкласса

        self._wrong_type_encode: str = self._('Тип кодирования видео должен быть одним из "{}"') + self._em
        self._wrong_preset_crf_encode: str = self._('Скорость кодирования и сжатия видео должна быть '
                                                               'одной из "{}"') + self._em
        self._wrong_sr_input_type: str = self._('Тип файлов для распознавания должен быть одним из "{}"') + self._em
        self._wrong_sampling_rate_vad: str = self._('Частота дискретизации речевого сигнала должна быть одной из "{}"')\
                                             + self._em

        self._download_model_from_repo: str = self._('Загрузка VAD модели "{}" из репозитория {}') + self._em

        self._subfolders_search: str = self._(
            'Поиск вложенных директорий в директории "{}" (глубина вложенности: {})'
        ) + self._em
        self._subfolders_not_found: str = self._('В указанной директории вложенные директории не найдены') + \
                                          self._em

        self._files_av_find: str = self._(
            'Поиск файлов с расширениями "{}" в директории "{}" (глубина вложенности: {})'
        ) + self._em

        self._files_analysis: str = self._('Анализ файлов') + self._em

        self._url_error: str = self._('Не удалось скачать модель{}') + self._em
        self._url_error_code: str = self._(' (ошибка {})')

# ######################################################################################################################
# Аудио
# ######################################################################################################################
@dataclass
class Audio(AudioMessages):
    """Класс для обработки аудиомодальности"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__() # Выполнение конструктора из суперкласса

        self._github_repo_vad: str = 'snakers4/silero-vad' # Репозиторий для загрузки VAD
        self._vad_model: str = 'silero_vad' # VAD модель

        # ----------------------- Только для внутреннего использования внутри класса

        self.__type_encode: str = '' # Тип кодирования
        self.__presets_crf_encode: str = '' # Скорость кодирования и сжатия
        self.__sr_input_type: str = '' # Тип файлов для распознавания речи
        self.__model_vad: Optional[torch.jit._script.RecursiveScriptModule] = None # VAD модель
        self.__get_speech_ts: Optional[FunctionType] = None # Временные метки VAD
        self.__dataset_video_vad: List[str] = [] # Пути до директорий с разделенными видеофрагментами
        self.__dataset_audio_vad: List[str] = [] # Пути до директорий с разделенными аудиофрагментами
        self.__unprocessed_files: List[str] = [] # Пути к файлам на которых VAD не отработал
        self.__len_paths: int = 0 # Количество аудиовизуальных файлов
        self.__curr_path: PosixPath = '' # Текущий аудиовизуальный файл
        self.__i: int = 0 # Счетчик
        self.__local_path: [[PosixPath], str] = '' # Локальный путь
        self.__aframes: torch.Tensor = torch.empty((), dtype = torch.float) # Аудиокадры
        # Метаданные для видео и аудио
        self.__file_metadata: Dict[str, Union[int, float]] = {'video_fps': 0.0, 'audio_fps': 0}

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (приватные)
    # ------------------------------------------------------------------------------------------------------------------

    def __audio_analysis(self, out: bool):
        """Анализ аудиодорожки

        Args:
            out (bool): Отображение

        Returns:
            bool: **True** если анализ аудиодорожки произведен, в обратном случае **False**
        """

        # Количество каналов в аудиодорожке
        channels_audio = self.__aframes.shape[0]
        if channels_audio > 2: self.__unprocessed_files.append(self.__curr_path); return False

        # Проход по всем каналам
        for channel in range(0, channels_audio):
            try:
                # Получение временных меток
                speech_timestamps = self.__get_speech_ts(
                    audio = self.__aframes[channel],
                    model = self.__model_vad,
                    threshold = 0.56,
                    sampling_rate = 16000,
                    min_speech_duration_ms = 250,
                    max_speech_duration_s = float('inf'),
                    min_silence_duration_ms = 50,
                    window_size_samples = 1536,
                    speech_pad_ms = 150,
                    return_seconds = False,
                    visualize_probs = False,
                    progress_tracking_callback = None
                )
            except Exception: self.__unprocessed_files.append(self.__curr_path); return False
            else:
                start_time = timedelta(seconds = speech_timestamps[0]['start'] / 16000) # Начальное время
                end_time = timedelta(seconds = speech_timestamps[0]['end'] / 16000) # Конечное время

                print(len(speech_timestamps))
                print(speech_timestamps, start_time, end_time)

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def vad(
        self, depth: int = 1, type_encode: str = TYPES_ENCODE[1], presets_crf_encode: str = PRESETS_CRF_ENCODE[5],
        sr_input_type: str = SR_INPUT_TYPES[0], threshold: float = THRESHOLD_VAD,
        sampling_rate: int = SAMPLING_RATE_VAD, min_speech_duration_ms: int = MIN_SPEECH_DURATION_MS_VAD,
        min_silence_duration_ms: int = MIN_SILENCE_DURATION_MS_VAD,
        window_size_samples: int = WINDOW_SIZE_SAMPLES_VAD, speech_pad_ms: int = SPEECH_PAD_MS,
        force_reload: bool = True, clear_dirvad: bool = False, out: bool = True
    ) -> bool:
        """ VAD (Voice Activity Detector) или (детектирование голосовой активности)

        Args:
            depth (int): Глубина иерархии для получения данных
            type_encode (str): Тип кодирования
            presets_crf_encode (str): Скорость кодирования и сжатия
            sr_input_type (str): Тип файлов для распознавания речи
            threshold (float): Порог вероятности речи (от 0.0 до 1.0)
            sampling_rate (int): Частота дискретизации (8000 или 16000)
            min_speech_duration_ms (int): Минимальная длительность речевого фрагмента в миллисекундах
            min_silence_duration_ms (int): Минимальная длительность тишины в выборках между отдельными речевыми
                                           фрагментами
            window_size_samples (int): Количество выборок в каждом окне (512, 1024, 1536 для частоты дискретизации
                                       16000 или 256, 512, 768 для частоты дискретизации 8000)
            speech_pad_ms (int): Внутренние отступы для итоговых речевых фрагментов
            force_reload (bool): Принудительная загрузка модели из сети
            clear_dirvad (bool): Очистка директории для сохранения фрагментов аудиовизуального сигнала
            out (bool): Отображение

        Returns:
            bool: **True** если детектирование голосовой активности произведено, в обратном случае **False**
        """

        try:
            # Проверка аргументов
            if (type(depth) is not int or depth < 1 or type(threshold) is not float or not (0.0 <= threshold <= 1.0)
                    or type(force_reload) is not bool or type(out) is not bool or type(clear_dirvad) is not bool):
                raise TypeError
        except TypeError: self.inv_args(__class__.__name__, self.vad.__name__, out = out); return False
        else:
            try:
                # Проверка настроек
                if type(type_encode) is not str or (type_encode in TYPES_ENCODE) is False: raise TypeEncodeVideoError
                if type(presets_crf_encode) is not str or (presets_crf_encode in PRESETS_CRF_ENCODE) is False:
                    raise PresetCFREncodeVideoError
                if type(sr_input_type) is not str or (sr_input_type in [x.lower() for x in SR_INPUT_TYPES]) is False:
                    raise SRInputTypeError
                if type(sampling_rate) is not int or (sampling_rate in [x for x in SAMPLING_RATE_VAD]) is False:
                    raise SamplingRateError
            except TypeEncodeVideoError:
                self.message_error(self._wrong_type_encode.format(
                    self.message_line(', '.join(x.replace('.', '') for x in TYPES_ENCODE))
                ), out = out); return False
            except PresetCFREncodeVideoError:
                self.message_error(self._wrong_preset_crf_encode.format(
                    self.message_line(', '.join(x.replace('.', '') for x in PRESETS_CRF_ENCODE))
                ), out = out); return False
            except SRInputTypeError:
                self.message_error(self._wrong_sr_input_type.format(
                    self.message_line(', '.join(x.replace('.', '') for x in SR_INPUT_TYPES))
                ), out = out); return False
            except SamplingRateError:
                self.message_error(self._wrong_sampling_rate_vad.format(
                    self.message_line(', '.join(str(x) for x in SAMPLING_RATE_VAD))
                ), out = out); return False
            else:
                # Только для внутреннего использования внутри класса
                self.__type_encode = type_encode
                self.__presets_crf_encode = presets_crf_encode
                self.__sr_input_type = sr_input_type

                torch.set_num_threads(1) # Установка количества потоков для внутриоперационного параллелизма на ЦП

                torch.hub.set_dir(self.path_to_save_models) # Установка пути к директории для сохранения VAD модели

                # Информационное сообщение
                self.message_info(self._download_model_from_repo.format(
                    self.message_line(self._vad_model),
                    urllib.parse.urljoin('https://github.com/', self._github_repo_vad)
                ), out = out)

                try:
                    # Подавление вывода
                    with io.capture_output():
                        # Загрузка VAD модели
                        self.__model_vad, utils = torch.hub.load(
                            repo_or_dir = self._github_repo_vad, model = self._vad_model, force_reload =
                            force_reload
                        )
                except FileNotFoundError:
                    self.message_error(
                        self._clear_folder_not_found.format(self.message_line(self.path_to_save_models)),
                        space = self._space,
                        out = out
                    ); return False
                except RuntimeError:
                    self.message_error(self._url_error.format(''), space = self._space, out = out); return False
                except urllib.error.HTTPError as e:
                    self.message_error(self._url_error.format(
                        self._url_error_code.format(self.message_line(str(e.code)))
                    ), space = self._space, out = out); return False
                except urllib.error.URLError:
                    self.message_error(self._url_error.format(''), space = self._space, out = out); return False
                except Exception: self.message_error(self._unknown_err, space = self._space, out = out); return False
                else:
                    self.__get_speech_ts, _, read_audio, _, _ = utils

                    # Информационное сообщение
                    self.message_info(self._subfolders_search.format(
                        self.message_line(self.path_to_dataset),
                        self.message_line(str(depth)),
                    ), out = out)

                    # Создание директории, где хранятся данные
                    if self.create_folder(self.path_to_dataset, out = False) is False: return False

                    # Получение вложенных директорий, где хранятся данные
                    nested_paths = self.get_paths(self.path_to_dataset, depth = depth, out = False)

                    # Вложенные директории не найдены
                    try:
                        if len(nested_paths) == 0: raise IsNestedCatalogsNotFoundError
                    except IsNestedCatalogsNotFoundError:
                        self.message_error(self._subfolders_not_found, space = self._space, out = out); return False

                    # Информационное сообщение
                    self.message_info(
                        self._files_av_find.format(
                            self.message_line(', '.join(x.replace('.', '') for x in self.ext_search_files)),
                            self.message_line(self.path_to_dataset),
                            self.message_line(str(depth))
                        ),
                        out = out
                    )

                    paths = [] # Пути до аудиовизуальных файлов

                    # Проход по всем вложенным директориям
                    for nested_path in nested_paths:
                        # Формирование списка с видеофайлами
                        for p in Path(nested_path).glob('*'):
                            # Добавление текущего пути к видеофайлу в список
                            if p.suffix.lower() in self.ext_search_files: paths.append(p.resolve())

                    # Директория с набором данных не содержит аудиовизуальных файлов с необходимыми расширениями
                    try:
                        self.__len_paths = len(paths) # Количество аудиовизуальных файлов

                        if self.__len_paths == 0: raise TypeError
                    except TypeError:
                        self.message_error(self._files_not_found, space = self._space, out = out); return False
                    except Exception:
                        self.message_error(self._unknown_err, space = self._space, out = out); return False
                    else:
                        # Очистка директории для сохранения фрагментов аудиовизуального сигнала
                        if clear_dirvad is True and os.path.exists(self.path_to_dataset_vad) is True:
                            if self.clear_folder(self.path_to_dataset_vad, out = False) is False: return False

                        self.__dataset_video_vad = [] # Пути до директорий с разделенными видеофрагментами
                        self.__dataset_audio_vad = [] # Пути до директорий с разделенными аудиофрагментами

                        self.__unprocessed_files = [] # Пути к файлам на которых VAD не отработал

                        # Информационное сообщение
                        self.message_info(self._files_analysis, out = out)

                        # Локальный путь
                        self.__local_path = lambda lp: os.path.join(
                            *Path(lp).parts[-abs((len(Path(lp).parts) - len(Path(self.path_to_dataset).parts))):])

                        # Проход по всем найденным аудиовизуальных файлам
                        for i, path in enumerate(paths):
                            self.__curr_path = path # Текущий аудиовизуальный файл
                            self.__i = i + 1 # Счетчик

                            self.message_progressbar(self._curr_progress.format(
                                self.__i, self.__len_paths,
                                round(self.__i * 100 / self.__len_paths, 2),
                                self.message_line(self.__local_path(self.__curr_path))
                            ), space = self._space, out = out)

                            splitted_path = str(self.__curr_path.parent.relative_to(Path(self.path_to_dataset))).strip()

                            self.__curr_path = str(self.__curr_path)

                            # Пропуск невалидных значений
                            if not splitted_path or re.search('\s', splitted_path) is not None: continue

                            join_path = lambda dir_va: os.path.join(self.path_to_dataset_vad, dir_va, splitted_path)

                            try:
                                # Директория с разделенными видеофрагментами
                                self.__dataset_video_vad.append(join_path(self.dir_va_names[0]))
                                if not os.path.exists(self.__dataset_video_vad[-1]):
                                    # Директория не создана
                                    if self.create_folder(self.__dataset_video_vad[-1], out = False) is False:
                                        raise IsNestedDirectoryVNotFoundError

                                # Директория с разделенными аудиофрагментами
                                self.__dataset_audio_vad.append(join_path(self.dir_va_names[1]))
                                if not os.path.exists(self.__dataset_audio_vad[-1]):
                                    # Директория не создана
                                    if self.create_folder(self.__dataset_audio_vad[-1], out = False) is False:
                                        raise IsNestedDirectoryANotFoundError
                            except (IsNestedDirectoryVNotFoundError, IsNestedDirectoryANotFoundError):
                                self.__unprocessed_files.append(self.__curr_path); continue
                            except Exception:
                                self.__unprocessed_files.append(self.__curr_path)
                                self.message_progressbar(close = True, out = out); continue
                            else:
                                try:
                                    # Чтение файла
                                    _, self.__aframes, self.__file_metadata = torchvision.io.read_video(
                                        self.__curr_path
                                    )
                                except Exception:
                                    self.__unprocessed_files.append(self.__curr_path)
                                    self.message_progressbar(close = True, out = out); continue
                                else:
                                    # Анализ аудиодорожки
                                    self.__audio_analysis(out = out)
                                    return


                                continue

                                # Тип файла
                                kind = filetype.guess(self.__curr_path)

                                # Видео
                                if kind.mime.startswith('video/') is True: pass



                                continue

                                # import subprocess  # Работа с процессами
                                # # https://trac.ffmpeg.org/wiki/audio%20types
                                # # Выполнение в новом процессе
                                # with subprocess.Popen(
                                #         ['ffmpeg', '-loglevel', 'quiet', '-i', str(self.__curr_path)] + [] +
                                #         ['-ar', str(16000), '-ac', str(1), '-f', 's16le', '-'],
                                #         stdout = subprocess.PIPE) as process:
                                #     t = None
                                #
                                #     while True:
                                #         data = process.stdout.read(4000)
                                #         if len(data) == 0: break
                                #
                                #         if t is None:
                                #             t = torch.frombuffer(data, dtype = torch.float32)
                                #         else:
                                #             t = torch.cat((t, torch.frombuffer(data, dtype = torch.float32)), 0)
                                #         # print(t.shape, t[0:10])

                                # t = t.unsqueeze(0)
                                # t = t.view(t.shape[0], 1)
                                # print(type(t), t.shape)
                                # print(torch.flatten(t).shape)

                                # torchaudio.save('/Users/dl/Desktop/test/' + str(self.__i) + '.wav', t, 16000)

                                # Чтение аудиофайла
                                # self.__wav = read_audio(
                                #     '/Users/dl/@DmitryRyumin/Databases/LRW/LRW_AUDIO/ABSOLUTELY_00001.wav',
                                #     sampling_rate = 16000
                                # )
                                # # print(type(self.__wav), self.__wav[0:100])
                                # self.__wav = self.__wav.unsqueeze(0)
                                # torchaudio.save('/Users/dl/Desktop/test/' + str(self.__i) + '.wav', self.__wav, 16000)

                                # v, a, h = torchvision.io.read_video(
                                #     self.__curr_path
                                # )
                                #
                                # print(v.shape)
                                # print(a.shape)
                                # print(h)
                                # kind = filetype.guess(self.__curr_path)
                                # print(kind)
                                #
                                # torchaudio.save('/Users/dl/Desktop/test/' + str(self.__i) + '.wav', a, 16000)

                        self.message_progressbar(close = True, out = out)





