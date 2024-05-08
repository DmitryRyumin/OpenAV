#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Работа с YAML
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
import os  # Работа с файловой системой
import yaml  # Кодирование и декодирование данные в удобном формате

import importlib.resources as pkg_resources  # Работа с ресурсами внутри пакетов

from dataclasses import dataclass  # Класс данных

from typing import Dict, Union  # Типы данных
from types import ModuleType

# Персональные
from openav.modules.file_manager.json_manager import Json  # Работа с JSON


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class YamlMessages(Json):
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


# ######################################################################################################################
# Работа с YAML
# ######################################################################################################################
@dataclass
class Yaml(YamlMessages):
    """Класс для работы с YAML

    Args:
        path_to_logs (str): Смотреть :attr:`~openav.modules.core.logging.Logging.path_to_logs`
        lang (str): Смотреть :attr:`~openav.modules.core.language.Language.lang`
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

    # ------------------------------------------------------------------------------------------------------------------
    #  Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def load_yaml(
        self, path_to_file: str, create: bool = False, out: bool = True
    ) -> Dict[str, Union[str, bool, int, float,],]:
        """Загрузка YAML файла

        Args:
            path_to_file (str): Путь к файлу YAML
            create (bool): Создание файла YAML в случае его отсутствия
            out (bool): Печатать процесс выполнения

        Returns:
            Dict[str, Union[str, bool, int, float]]: Словарь из yaml файла
        """

        # Проверка аргументов
        if type(path_to_file) is not str or not path_to_file or type(create) is not bool or type(out) is not bool:
            self.inv_args(__class__.__name__, self.load_yaml.__name__, out=out)
            return {}

        # Поиск YAML файла не удался
        if self.search_file(path_to_file, "yaml", create, out) is False:
            return {}

        path_to_file = os.path.normpath(path_to_file)

        # Вывод сообщения
        self.message_info(
            self._load_data.format(self.message_line(os.path.basename(path_to_file))), space=self._space, out=out
        )

        # Открытие файла
        with open(path_to_file, mode="r", encoding="utf-8") as yaml_data_file:
            try:
                config = yaml.load(yaml_data_file, Loader=yaml.FullLoader)
            except yaml.YAMLError:
                self.message_error(self._invalid_file, space=self._space, out=out)
                return {}

        # Файл пуст
        if not config:
            self.message_error(self._config_empty, space=self._space, out=out)
            return {}

        return config  # Результат

    def load_yaml_resources(
        self, module: ModuleType, path_to_file: str, out: bool = True
    ) -> Dict[str, Union[str, bool, int, float]]:
        """Загрузка YAML файла из ресурсов модуля

        Args:
            module (ModuleType): Модуль
            path_to_file (str): Путь к файлу YAML
            out (bool): Печатать процесс выполнения

        Returns:
            Dict[str, Union[str, bool, int, float]]: Словарь из yaml файла
        """

        # Проверка аргументов
        if (
            isinstance(module, ModuleType) is False
            or type(path_to_file) is not str
            or not path_to_file
            or type(out) is not bool
        ):
            self.inv_args(__class__.__name__, self.load_yaml_resources.__name__, out=out)
            return {}

        # Вывод сообщения
        self.message_info(self._load_data_resources.format(self.message_line(module.__name__)), out=out)

        # Ресурс с YAML файлом не найден
        if pkg_resources.is_resource(module, path_to_file) is False:
            self.message_error(self._load_data_resources_not_found, space=self._space, out=out)
            return {}

        # Открытие файла
        with pkg_resources.open_text(module, path_to_file, encoding="utf-8", errors="strict") as yaml_data_file:
            try:
                config = yaml.load(yaml_data_file, Loader=yaml.FullLoader)
            except yaml.YAMLError:
                self.message_error(self._invalid_file, space=self._space, out=out)
                return {}

        # Файл пуст
        if not config:
            self.message_error(self._config_empty, space=self._space, out=out)
            return {}

        return config  # Результат
