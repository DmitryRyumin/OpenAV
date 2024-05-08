#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Видеомодальность
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
import warnings
import logging
import absl.logging
import sys
import os  # Взаимодействие с файловой системой
import math

# Настройте фильтрацию предупреждений до импорта mediapipe
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Установите уровень логирования
logging.getLogger().setLevel(logging.ERROR)

# Установите уровень логирования absl
absl.logging.set_verbosity(absl.logging.ERROR)

from dataclasses import dataclass  # Класс данных

import numpy as np  # Научные вычисления
import re  # Регулярные выражения
import filetype  # Определение типа файла и типа MIME
from PIL import Image  # Считывание изображений

with open(os.devnull, "w") as devnull:
    sys.stdout = devnull
    sys.stderr = devnull

    import mediapipe as mp
    import cv2

mp.solutions.face_mesh.FaceMesh()

# Типы данных
from typing import List, Set, Optional
from types import ModuleType

from pathlib import Path  # Работа с путями в файловой системе

# Персональные
from openav.modules.core.exceptions import (
    IsNestedCatalogsNotFoundError,
)
from openav.modules.file_manager.json_manager import Json  # Класс для работы с Json

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# ######################################################################################################################
# Константы
# ######################################################################################################################

# Метрики оценки нейросетевой модели
METRICS_VIDEO: List[str] = [
    "accuracy",
]
DPI: List[int] = [72, 96, 150, 300, 600, 1200]  # DPI
COLOR_MODE: List[str] = ["gray", "rgb"]  # Цветовая гамма конечного изображения
RESIZE_RESAMPLE_MODE: List[str] = ["nearest", "bilinear", "lanczos"]  # Фильтры для масштабирования
EXT_VIDEO: List[str] = ["mov", "mp4", "webm"]  # Расширения искомых файлов
EXT_LIP: str = "png"
EXT_NPY: str = "npy"  # Расширения для сохранения сырых данных


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

        self.__mp_face_mesh: Optional[ModuleType] = None

        self.__mp_drawing: Optional[ModuleType] = None
        self.__drawing_spec: Optional[mp.solutions.drawing_utils.DrawingSpec] = None

        self._lip_coords: Set[int] = set(
            [
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
                185,
                40,
                39,
                37,
                0,
                267,
                269,
                270,
                409,
                291,
                78,
                95,
                88,
                178,
                87,
                14,
                317,
                402,
                318,
                324,
                308,
                191,
                80,
                81,
                82,
                13,
                312,
                311,
                310,
                415,
                308,
            ]
        )

        self.__min_max_coords: List[int] = [-1, -1, -1, -1]  # min x, max x, min y, max y

        self._area_lip: List[int] = []
        self._area_lip_original: int = 0
        self._cnt_lip: int = 1

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
        resize: bool = True,
        resize_resample: str = "nearest",
        size_width: int = 112,
        size_height: int = 112,
        color_mode: str = "rgb",
        dpi: int = 1200,
        save_raw_data: bool = True,
        clear_dir_video: bool = False,
        out: bool = True,
    ) -> bool:
        """Предобработка речевых видеоданных

        Args:
            depth (int): Глубина иерархии для получения данных
            resize (bool):  Изменение размера кадра с найденной областью губ
            resize_resample (str): Фильтр для масштабирования
            size_width (int): Ширина области губ
            size_height (int): Высота области губ
            color_mode (str):  Цветовая гамма
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
                or type(resize) is not bool
                or type(size_width) is not int
                or type(size_height) is not int
                or type(color_mode) is not str
                or (color_mode in COLOR_MODE) is False
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

                try:
                    self.__mp_face_mesh = mp.solutions.face_mesh

                    self.__mp_drawing = mp.solutions.drawing_utils
                    self.__drawing_spec = self.__mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
                except Exception:
                    self.message_error(self._unknown_err, space=self._space, out=out)
                    return False
                else:
                    # Проход по всем найденным визуальных файлам
                    for i, path in enumerate(paths):
                        self._cnt_lip = 1

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
                                cap = cv2.VideoCapture(self.__curr_path)

                                if not os.path.exists(self.path_to_dataset_video):
                                    # Директория не создана
                                    if self.create_folder(self.path_to_dataset_video, out=False) is False:
                                        raise FileNotFoundError

                                path_to_subfolder = os.path.join(
                                    self.path_to_dataset_video, Path(self.__curr_path).stem
                                )

                                # Очистка директории
                                if clear_dir_video is True and os.path.exists(path_to_subfolder) is True:
                                    if (
                                        self.clear_folder(
                                            path_to_subfolder,
                                            out=False,
                                        )
                                        is False
                                    ):
                                        return False

                                if not os.path.exists(path_to_subfolder):
                                    # Директория не создана
                                    if (
                                        self.create_folder(
                                            path_to_subfolder,
                                            out=False,
                                        )
                                        is False
                                    ):
                                        raise FileNotFoundError

                                with self.__mp_face_mesh.FaceMesh(
                                    static_image_mode=False,
                                    max_num_faces=1,
                                    min_detection_confidence=0.5,
                                ) as face_mesh:
                                    while cap.isOpened():
                                        _, curr_frame = cap.read()

                                        if curr_frame is None:
                                            break

                                        results = face_mesh.process(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB))

                                        if results.multi_face_landmarks:
                                            frame_height, frame_width, _ = curr_frame.shape

                                            for face_landmarks in results.multi_face_landmarks:
                                                land_x = []
                                                land_y = []

                                                for idx5, landmark in enumerate(face_landmarks.landmark):
                                                    x = min(math.floor(landmark.x * frame_width), frame_width - 1)
                                                    y = min(math.floor(landmark.y * frame_height), frame_height - 1)

                                                    if idx5 in self._lip_coords:
                                                        land_x.append(x)
                                                        land_y.append(y)

                                                self.__min_max_coords[0] = min(land_x)
                                                self.__min_max_coords[1] = max(land_x)
                                                self.__min_max_coords[2] = min(land_y)
                                                self.__min_max_coords[3] = max(land_y)

                                                area = (self.__min_max_coords[1] - self.__min_max_coords[0]) * (
                                                    self.__min_max_coords[3] - self.__min_max_coords[2]
                                                )

                                                if self._cnt_lip == 0:
                                                    self._area_lip_original = area

                                                self._area_lip.append(area)

                                                lip_roi = curr_frame[
                                                    self.__min_max_coords[2] : self.__min_max_coords[3],
                                                    self.__min_max_coords[0] : self.__min_max_coords[1],
                                                    :,
                                                ]

                                                lip_roi_path = os.path.join(
                                                    path_to_subfolder, str(self._cnt_lip) + "." + EXT_LIP
                                                )

                                                lip_roi = lip_roi[:, :, ::-1]

                                                # Создание и сохранение изображения с помощью Pillow
                                                img = Image.fromarray(lip_roi)

                                                if color_mode == COLOR_MODE[0]:
                                                    img = img.convert("L")

                                                if resize is True:
                                                    if resize_resample is RESIZE_RESAMPLE_MODE[0]:
                                                        resize_resample = Image.NEAREST
                                                    elif resize_resample is RESIZE_RESAMPLE_MODE[1]:
                                                        resize_resample = Image.BILINEAR
                                                    elif resize_resample is RESIZE_RESAMPLE_MODE[2]:
                                                        resize_resample = Image.LANCZOS
                                                    else:
                                                        resize_resample = Image.NEAREST

                                                    img = img.resize(
                                                        (size_width, size_height), resample=resize_resample
                                                    )

                                                img.save(lip_roi_path, dpi=(dpi, dpi))

                                                if save_raw_data:
                                                    # Сохранение сырых данных в формате .npy
                                                    raw_data_path = lip_roi_path.replace("." + EXT_LIP, "." + EXT_NPY)
                                                    np.save(raw_data_path, lip_roi)
                                        else:
                                            self.__min_max_coords = [-1, -1, -1, -1]

                                        self._cnt_lip += 1
                        except Exception:
                            self.__unprocessed_files.append(self.__curr_path)
                            self.message_progressbar(close=True, out=out)
                            continue

                    self.message_progressbar(close=True, out=out)

                    # Файлы на которых предварительная обработка не отработала
                    unprocessed_files_unique = np.unique(np.array(self.__unprocessed_files)).tolist()

                    if len(unprocessed_files_unique) == 0 and len(self.__not_saved_files) == 0:
                        self.message_true(self._preprocess_true, space=self._space, out=out)
