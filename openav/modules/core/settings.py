#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Настройки
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
# Подавление Warning
import warnings
for warn in [UserWarning, FutureWarning]: warnings.filterwarnings('ignore', category = warn)

import os # Взаимодействие с файловой системой

from dataclasses import dataclass # Класс данных

from colorama import init # Цветной текст терминала

# Типы данных
from typing import List

# Персональные
from openav.modules.core.messages import Messages # Сообщения

# ######################################################################################################################
# Константы
# ######################################################################################################################
PATH_TO_SAVE_MODELS: str = './models' # Путь к директории для сохранения моделей
PATH_TO_DATASET: str = './dataset' # Путь к директории набора данных
# Путь к директории набора данных состоящего из фрагментов аудиовизуального сигнала
PATH_TO_DATASET_VAD: str = './dataset_vad'
IGNORE_DIRS: List[str] = [] # Директории не входящие в выборку
DIR_VA_NAMES: List[str] = ['Video', 'Audio'] # Названия директорий для видео и аудио
EXT_SEARCH_FILES: List[str] = ['mov', 'mp4', 'wav'] # Расширения искомых файлов
CHUNK_SIZE: int = 1000000 # Размер загрузки файла из сети за 1 шаг

# ######################################################################################################################
# Настройки
# ######################################################################################################################
@dataclass
class Settings(Messages):
    """Класс для настроек

    Args:
        lang (str): Смотреть :attr:`~openav.modules.core.language.Language.lang`
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__() # Выполнение конструктора из суперкласса

        init() # Инициализация терминала с возможностью цветного текста

        self.__color_green: str = '\033[92m' # Зеленый
        self.__color_red: str = '\033[91m'   # Красный
        self.__color_blue: str = '\033[94m'  # Синий
        self.__text_bold: str = '\033[1m'    # Жирный
        self.__clear_line: str = '\x1b[2K'   # Очистка линии в терминале
        self.__text_end: str = '\033[0m'     # Выход

        self.path_to_save_models: str = PATH_TO_SAVE_MODELS # Путь к директории для сохранения моделей
        self.path_to_dataset: str = PATH_TO_DATASET # Путь к директории набора данных
        # Путь к директории набора данных состоящего из фрагментов аудиовизуального сигнала
        self.path_to_dataset_vad: str = PATH_TO_DATASET_VAD
        self.ignore_dirs: List[str] = IGNORE_DIRS # Директории не входящие в выборку
        self.dir_va_names: List[str] = DIR_VA_NAMES # Названия директорий для видео и аудио
        self.ext_search_files: List[str] = EXT_SEARCH_FILES # Расширения искомых файлов
        self.chunk_size: int = CHUNK_SIZE # Размер загрузки файла из сети за 1 шаг

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def color_green(self) -> str:
        """Получение зеленого цвета текста в терминале

        Returns:
            str: Цвет текста в терминале
        """

        return self.__color_green

    @property
    def color_red(self) -> str:
        """Получение красного цвета текста в терминале

        Returns:
            str: Цвет текста в терминале
        """

        return self.__color_red

    @property
    def color_blue(self) -> str:
        """Получение синего цвета текста в терминале

        Returns:
            str: Цвет текста в терминале
        """

        return self.__color_blue

    @property
    def text_bold(self) -> str:
        """Получение жирного начертания текста в терминале

        Returns:
            str: Жирное начертание текста в терминале
        """

        return self.__text_bold

    @property
    def clear_line(self) -> str:
        """Получение очистки линии в терминале

        Returns:
            str: Очистка линии в терминале
        """

        return self.__clear_line

    @property
    def text_end(self) -> str:
        """Получение сброса оформления текста в терминале

        Returns:
            str: Сброс оформления текста в терминале
        """

        return self.__text_end

    @property
    def path_to_save_models(self) -> str:
        """Получение/установка пути к директории для сохранения моделей

        Args:
            (str): Путь

        Returns:
            str: Путь
        """

        return self.__path_to_save_models

    @path_to_save_models.setter
    def path_to_save_models(self, path: str):
        """Установка пути к директории для сохранения моделей"""

        try:
            # Проверка аргументов
            if type(path) is not str or not path: raise TypeError
        except TypeError: pass
        else: self.__path_to_save_models = os.path.normpath(path.strip())

    @property
    def path_to_dataset(self) -> str:
        """Получение/установка пути к директории набора данных

        Args:
            (str): Путь

        Returns:
            str: Путь
        """

        return self.__path_to_dataset

    @path_to_dataset.setter
    def path_to_dataset(self, path: str):
        """Установка пути к директории набора данных"""

        try:
            # Проверка аргументов
            if type(path) is not str or not path: raise TypeError
        except TypeError: pass
        else: self.__path_to_dataset = os.path.normpath(path.strip())

    @property
    def path_to_dataset_vad(self) -> str:
        """Получение/установка пути к директории набора данных состоящего из фрагментов аудиовизуального сигнала

        Args:
            (str): Путь

        Returns:
            str: Путь
        """

        return self.__path_to_dataset_vad

    @path_to_dataset_vad.setter
    def path_to_dataset_vad(self, path: str):
        """Установка пути к директории набора данных состоящего из фрагментов аудиовизуального сигнала"""

        try:
            # Проверка аргументов
            if type(path) is not str or not path: raise TypeError
        except TypeError: pass
        else: self.__path_to_dataset_vad = os.path.normpath(path.strip())

    @property
    def ignore_dirs(self) -> List[str]:
        """Получение/установка списка с директориями не входящими в выборку

        Args:
            (List[str]): Список с директориями

        Returns:
            List[str]: Список с директориями
        """

        return self.__ignore_dirs

    @ignore_dirs.setter
    def ignore_dirs(self, l: List[str]) -> None:
        """Установка списка с директориями не входящими в выборку"""

        if type(l) is list:
            try: self.__ignore_dirs = [x.strip() for x in l]
            except Exception: pass

    @property
    def dir_va_names(self) -> List[str]:
        """Получение/установка списка с названиями директорий для видео и аудио

        Args:
            (List[str]): Список с директориями

        Returns:
            List[str]: Список с директориями
        """

        return self.__dir_va_names

    @dir_va_names.setter
    def dir_va_names(self, l: List[str]) -> None:
        """Установка списка с названиями директорий для видео и аудио"""

        if type(l) is list and len(l) == 2:
            try: self.__dir_va_names = [x.strip() for x in l]
            except Exception: pass

    @property
    def ext_search_files(self) -> List[str]:
        """Получение/установка списка с расширениями искомых файлов

        Args:
            (List[str]): Список с расширениями искомых файлов

        Returns:
            List[str]: Список с расширениями искомых файлов
        """

        return self.__ext_search_files

    @ext_search_files.setter
    def ext_search_files(self, l: List[str]) -> None:
        """Установка списка с расширениями искомых файлов"""

        if type(l) is list and len(l) > 0:
            try: self.__ext_search_files = ['.' + x.strip().lower() for x in l]
            except Exception: pass

    @property
    def chunk_size(self) -> int:
        """Получение/установка размера загрузки файла из сети за 1 шаг

        Args:
            (int): Размер загрузки файла из сети за 1 шаг

        Returns:
            int: Размер загрузки файла из сети за 1 шаг
        """

        return self.__chunk_size

    @chunk_size.setter
    def chunk_size(self, size: int) -> None:
        """Установка директории для сохранения данных"""

        if type(size) is int and size > 0: self.__chunk_size = size