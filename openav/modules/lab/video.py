#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Видеомодальность
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
import numpy as np  # Научные вычисления
import re  # Регулярные выражения
import filetype  # Определение типа файла и типа MIME

# Типы данных
from typing import List

from pathlib import Path  # Работа с путями в файловой системе

# Персональные
from openav.modules.core.exceptions import (
    IsNestedCatalogsNotFoundError,
)
from openav.modules.file_manager.json_manager import Json  # Класс для работы с Json

# Метрики оценки нейросетевой модели
METRICS_VIDEO: List[str] = [
    "accuracy",
]
DPI: List[int] = [72, 96, 150, 300, 600, 1200]  # DPI
COLOR_MODE: List[str] = ["gray", "rgb"]  # Цветовая гамма конечного изображения
EXT_VIDEO: List[str] = ["mov", "mp4", "webm"]  # Расширения искомых файлов


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class VideoMessages(Json):
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

        self._subfolders_search: str = (
            self._('Поиск вложенных директорий в директории "{}" (глубина вложенности: {})') + self._em
        )
        self._subfolders_not_found: str = self._("В указанной директории вложенные директории не найдены") + self._em
        self._files_av_find: str = (
            self._('Поиск файлов с расширениями "{}" в директории "{}" (глубина вложенности: {})') + self._em
        )

        self.preprocess_video_files: str = self._("Предобработка речевых видеоданных") + self._em

        self._preprocess_true: str = self._("Все файлы успешно предобработаны") + self._em


# ######################################################################################################################
# Видео
# ######################################################################################################################
@dataclass
class Video(VideoMessages):
    """Класс для обработки видеомодальности

    Args:
        path_to_logs (str): Смотреть :attr:`~openav.modules.core.logging.Logging.path_to_logs`
        lang (str): Смотреть :attr:`~openav.modules.core.language.Language.lang`
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        # ----------------------- Только для внутреннего использования внутри класса

        self.__dataset_preprocess_video: List[str] = []  # Пути до директорий с изображениями губ
        self.__unprocessed_files: List[str] = []  # Пути к файлам из которых области губ не извлечены
        self.__not_saved_files: List[str] = []  # Пути к файлам которые не сохранились при обработке

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (приватные)
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def preprocess_video(
        self,
        depth: int = 1,
        dpi: int = 1200,
        save_raw_data: bool = True,
        clear_dir_video: bool = False,
        out: bool = True,
    ) -> bool:
        """Предобработка речевых видеоданных

        Args:
            depth (int): Глубина иерархии для получения данных
            dpi (int): DPI
            save_raw_data (bool): Сохранение сырых данных с областями губ в формате .npy
            clear_dir_video (bool): Очистка директории для сохранения видео данных после предобработки
            out (bool) Отображение

        Returns:
            bool: **True** если предобработка речевых видеоданных произведена, в обратном случае
            **False**
        """

        try:
            # Проверка аргументов
            if (
                type(depth) is not int
                or depth < 1
                or type(dpi) is not int
                or (dpi in DPI) is False
                or type(save_raw_data) is not bool
                or type(clear_dir_video) is not bool
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self.inv_args(__class__.__name__, self.preprocess_video.__name__, out=out)
            return False
        else:
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

            paths = []  # Пути до визуальных файлов

            # Проход по всем вложенным директориям
            for nested_path in nested_paths:
                # Формирование списка с видеофайлами
                for p in Path(nested_path).glob("*"):
                    # Добавление текущего пути к видеофайлу в список
                    if p.suffix.lower() in self.ext_search_files:
                        paths.append(p.resolve())

            # Директория с набором данных не содержит визуальных файлов с необходимыми расширениями
            try:
                self.__len_paths = len(paths)  # Количество визуальных файлов

                if self.__len_paths == 0:
                    raise TypeError
            except TypeError:
                self.message_error(self._files_not_found, space=self._space, out=out)
                return False
            except Exception:
                self.message_error(self._unknown_err, space=self._space, out=out)
                return False
            else:
                pass

                # Очистка директории для сохранения фрагментов визуального сигнала
                if clear_dir_video is True and os.path.exists(self.path_to_dataset_video) is True:
                    if self.clear_folder(self.path_to_dataset_video, out=False) is False:
                        return False

                self.__dataset_preprocess_video = []  # Пути до директорий с изображениями губ

                self.__unprocessed_files = []  # Пути к файлам из которых области губ не извлечены

                # Информационное сообщение
                self.message_info(self.preprocess_video_files, out=out)

                # Локальный путь
                self.__local_path = lambda lp: os.path.join(
                    *Path(lp).parts[-abs((len(Path(lp).parts) - len(Path(self.path_to_dataset).parts))) :]
                )

                # Проход по всем найденным визуальных файлам
                for i, path in enumerate(paths):
                    self.__curr_path = path  # Текущий визуальный файл
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

                    self.__splitted_path = str(self.__curr_path.parent.relative_to(Path(self.path_to_dataset))).strip()

                    self.__curr_path = str(self.__curr_path)

                    # Пропуск невалидных значений
                    if not self.__splitted_path or re.search(r"\s", self.__splitted_path) is not None:
                        continue

                    # Тип файла
                    kind = filetype.guess(self.__curr_path)

                #     try:
                #         # Видео или аудио
                #         if kind.mime.startswith("video/") is True or kind.mime.startswith("audio/") is True:
                #             # Формирование мел-спектрограммы
                #             waveform, sample_rate = librosa.load(self.__curr_path, sr=sample_rate)
                #             waveform = torch.Tensor(waveform)

                #             torchaudio_melspec = torchaudio.transforms.MelSpectrogram(
                #                 sample_rate=sample_rate,
                #                 n_fft=n_fft,
                #                 win_length=None,
                #                 hop_length=hop_length,
                #                 center=center,
                #                 pad_mode=pad_mode,
                #                 power=power,
                #                 norm=norm,
                #                 onesided=True,
                #                 n_mels=n_mels,
                #                 f_max=None,
                #             )(waveform)

                #             # Преобразование мел-спектрограммы в децибелы
                #             melspectogram_db_transform = torchaudio.transforms.AmplitudeToDB()
                #             melspec_db = melspectogram_db_transform(torchaudio_melspec)

                #             # Преобразование мел-спектрограммы в numpy-массив
                #             melspec_np = melspec_db.numpy()

                #             # Текущее время (TimeStamp)
                #             # см. datetime.fromtimestamp()
                #             self.__curr_ts = str(datetime.now().timestamp()).replace(".", "_")

                #             # Путь до мел-спектрограммы
                #             melspec_path = os.path.join(
                #                 self.path_to_dataset_audio,
                #                 Path(self.__curr_path).stem + "_" + self.__curr_ts + "." + EXT_AUDIO_SPEC,
                #             )

                #             if not os.path.exists(self.path_to_dataset_audio):
                #                 # Директория не создана
                #                 if self.create_folder(self.path_to_dataset_audio, out=False) is False:
                #                     raise FileNotFoundError

                #             # Нормализация значений мел-спектрограммы в диапазон [0, 1]
                #             melspec_np = (melspec_np - melspec_np.min()) / (melspec_np.max() - melspec_np.min())

                #             # Переворот массива по вертикали
                #             melspec_np = np.flip(melspec_np, axis=0)

                #             # Применение цветовой карты
                #             # color_gradients: viridis, plasma, inferno, magma, cividis
                #             cmap = cm.get_cmap(color_gradients)
                #             melspec_rgb = cmap(melspec_np)[:, :, :3]  # Извлечение только RGB-каналов

                #             # Нормализация значений в диапазон [0, 255]
                #             melspec_rgb = (melspec_rgb * 255).astype("uint8")

                #             # Создание и сохранение изображения с помощью Pillow
                #             img = Image.fromarray(melspec_rgb)
                #             img.save(melspec_path, dpi=(dpi, dpi))

                #             if save_raw_data:
                #                 # Сохранение сырых данных мел-спектрограммы в формате .npy
                #                 raw_data_path = melspec_path.replace("." + EXT_AUDIO_SPEC, "." + EXT_NPY)
                #                 np.save(raw_data_path, melspec_np)
                #     except Exception:
                #         self.__unprocessed_files.append(self.__curr_path)
                #         self.message_progressbar(close=True, out=out)
                #         continue

                self.message_progressbar(close=True, out=out)

                # Файлы на которых предварительная обработка не отработала
                unprocessed_files_unique = np.unique(np.array(self.__unprocessed_files)).tolist()

                if len(unprocessed_files_unique) == 0 and len(self.__not_saved_files) == 0:
                    self.message_true(self._preprocess_true, space=self._space, out=out)
