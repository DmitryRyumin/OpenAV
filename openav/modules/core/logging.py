#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Логирование
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
# Подавление Warning
import warnings

for warn in [UserWarning, FutureWarning]:
    warnings.filterwarnings("ignore", category=warn)

import os  # Взаимодействие с файловой системой
import sys  # Доступ к некоторым переменным и функциям Python
import re  # Регулярные выражения

from dataclasses import dataclass, field  # Класс данных

import logging  # Логирование

# Типы данных
from typing import Optional

# Персональные
from openav import __title__

# ######################################################################################################################
# Константы
# ######################################################################################################################
ARG_PATH_TO_LOGS = "--path_to_logs"  # Аргумент в парсере командной строки
PATH_TO_LOGS: str = os.path.normpath("./logs")  # Путь к директории для сохранения LOG файлов
NAME_LOG: str = __title__ + ".log"  # Имя LOG файла


# ######################################################################################################################
# Логирование
# ######################################################################################################################
@dataclass
class Logging:
    """Класс для логирования

    Args:
        path_to_logs (str): Путь к директории для сохранения LOG файлов
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    path_to_logs: str
    __path_to_logs: str = field(default=PATH_TO_LOGS, init=False, repr=False)
    """Путь к директории для сохранения LOG файлов

    .. note::
        private (приватный аргумент)
    """

    def __post_init__(self):
        self.path_to_logs = self.__get_arg()  # Установка пути к директории для сохранения LOG файлов

        self._logger_handler: Optional[logging.NullHandler] = None  # Обработчик логирования
        self.__logger: bool = self.__create_logging()  # Создание регистратора и обработчика для логирования

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def path_to_logs(self) -> str:
        """Получение/установка пути к директории для сохранения LOG файлов

        Args:
            (str): Путь

        Returns:
            str: Путь
        """

        return self.__path_to_logs

    @path_to_logs.setter
    def path_to_logs(self, path: str):
        """Установка пути к директории для сохранения LOG файлов"""

        try:
            # Проверка аргументов
            if type(path) is not str or not path:
                raise TypeError
        except TypeError:
            pass
        else:
            self.__path_to_logs = os.path.normpath(path.strip())

    @property
    def check_create_logger(self) -> bool:
        """Получение создания регистратора и обработчика для логирования

        Returns:
            bool: **True** если регистратор и обработчик созданы, в обратном случае **False**
        """

        return self.__logger

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (приватные)
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __get_arg() -> str:
        """Получение аргумента в парсере командной строки (путь к директории для сохранения LOG файлов)

        .. note::
            private (приватный метод)

        Returns:
            str: Путь к директории для сохранения LOG файлов
        """

        for cnt, arg in enumerate(sys.argv):
            if arg == ARG_PATH_TO_LOGS:
                try:
                    return sys.argv[cnt + 1]
                except IndexError:
                    return ""
        return ""

    def __create_logging(self) -> bool:
        """Создание регистратора и обработчика для логирования

        .. note::
            private (приватный метод)

        Returns:
            bool: **True** если регистратор и обработчик созданы, в обратном случае **False**
        """

        if not os.path.exists(self.path_to_logs):
            try:
                os.makedirs(self.path_to_logs)
            except Exception:
                return False

        try:
            # Создание регистратора
            logging.basicConfig(
                filename=os.path.join(self.path_to_logs, NAME_LOG),
                encoding="utf-8",
                filemode="w",
                format="%(asctime)s.%(msecs)03d - %(name)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                level=logging.DEBUG,
            )
        except FileNotFoundError:
            return False
        except Exception:
            return False
        else:
            # Создание обработчика
            self._logger_handler = logging.NullHandler()

            self._logger_handler.setLevel(logging.DEBUG)

            logging.getLogger("requests").setLevel(logging.CRITICAL)
            logging.getLogger("urllib3").setLevel(logging.CRITICAL)
            logging.getLogger("").addHandler(self._logger_handler)

            return True

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def re_inv_chars(path: str) -> str:
        """Удаление недопустимых символов из пути

        Args:
            (str): Путь

        Returns:
            str: Путь
        """

        # Проверка аргументов
        if type(path) is not str or not path:
            return ""

        return re.sub('[\\/:"*?<>|]+', "", path)
