#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Работа с JSON
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
import os  # Работа с файловой системой
import json  # Кодирование и декодирование данные в удобном формате

import importlib.resources as pkg_resources  # Работа с ресурсами внутри пакетов

from dataclasses import dataclass  # Класс данных

from typing import Dict, Union  # Типы данных
from types import ModuleType

# Персональные
from openav.modules.file_manager.download import Download  # Загрузка файлов


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class JsonMessages(Download):
    """Класс для сообщений"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self._load_data: str = self._('Загрузка данных из файла "{}"') + self._em
        self._invalid_file: str = self._("Данные не загружены") + self._em
        self._config_empty: str = self._("Файл пуст") + self._em
        self._load_data_resources: str = self._('Загрузка данных из ресурсов "{}"') + self._em
        self._load_data_resources_not_found: str = self._("Ресурс не найден") + self._em


# ######################################################################################################################
# Работа с JSON
# ######################################################################################################################
@dataclass
class Json(JsonMessages):
    """Класс для работы с JSON"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

    # ------------------------------------------------------------------------------------------------------------------
    #  Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def load_json(
        self, path_to_file: str, create: bool = False, out: bool = True
    ) -> Dict[str, Union[str, bool, int, float,],]:
        """Загрузка JSON файла

        Args:
            path_to_file (str): Путь к файлу JSON
            create (bool): Создание файла JSON в случае его отсутствия
            out (bool): Печатать процесс выполнения

        Returns:
            Dict[str, Union[str, bool, int, float]]: Словарь из json файла
        """

        # Проверка аргументов
        if type(path_to_file) is not str or not path_to_file or type(create) is not bool or type(out) is not bool:
            self.inv_args(__class__.__name__, self.load_json.__name__, out=out)
            return {}

        # Поиск JSON файла не удался
        if self.search_file(path_to_file, "json", create, out) is False:
            return {}

        path_to_file = os.path.normpath(path_to_file)

        # Вывод сообщения
        self.message_info(
            self._load_data.format(self.message_line(os.path.basename(path_to_file))), space=self._space, out=out
        )

        # Открытие файла
        with open(path_to_file, mode="r", encoding="utf-8") as json_data_file:
            try:
                config = json.load(json_data_file)
            except json.JSONDecodeError:
                self.message_error(self._invalid_file, space=self._space, out=out)
                return {}

        # Файл пуст
        if not config:
            self.message_error(self._config_empty, space=self._space, out=out)
            return {}

        return config  # Результат

    def load_json_resources(
        self, module: ModuleType, path_to_file: str, out: bool = True
    ) -> Dict[str, Union[str, bool, int, float]]:
        """Загрузка JSON файла из ресурсов модуля

        Args:
            module (ModuleType): Модуль
            path_to_file (str): Путь к файлу JSON
            out (bool): Печатать процесс выполнения

        Returns:
            Dict[str, Union[str, bool, int, float]]: Словарь из json файла
        """

        # Проверка аргументов
        if (
            isinstance(module, ModuleType) is False
            or type(path_to_file) is not str
            or not path_to_file
            or type(out) is not bool
        ):
            self.inv_args(__class__.__name__, self.load_json_resources.__name__, out=out)
            return {}

        # Вывод сообщения
        self.message_info(self._load_data_resources.format(self.message_line(module.__name__)), out=out)

        # Ресурс с JSON файлом не найден
        if pkg_resources.is_resource(module, path_to_file) is False:
            self.message_error(self._load_data_resources_not_found, space=self._space, out=out)
            return {}

        # Открытие файла
        with pkg_resources.open_text(module, path_to_file, encoding="utf-8", errors="strict") as json_data_file:
            try:
                config = json.load(json_data_file)
            except json.JSONDecodeError:
                self.message_error(self._invalid_file, space=self._space, out=out)
                return {}

        # Файл пуст
        if not config:
            self.message_error(self._config_empty, space=self._space, out=out)
            return {}

        return config  # Результат
