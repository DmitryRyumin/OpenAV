#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ядро
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
# Подавление Warning
import warnings
for warn in [UserWarning, FutureWarning]: warnings.filterwarnings('ignore', category = warn)

from dataclasses import dataclass # Класс данных

import os           # Взаимодействие с файловой системой
import sys          # Доступ к некоторым переменным и функциям Python
import argparse     # Парсинг аргументов и параметров командной строки
import numpy as np  # Научные вычисления
import pandas as pd # Обработка и анализ данных
import prettytable  # Отображение таблиц в терминале
import colorama     # Цветной текст терминала
import IPython      # Интерактивная оболочка для языка программирования
import torch        # Машинное обучение от Facebook
import torchaudio   # Работа с аудио от Facebook
import torchvision  # Работа с видео от Facebook
import av           # Работа с FFmpeg
import filetype     # Определение типа файла и типа MIME

from datetime import datetime       # Работа со временем
from prettytable import PrettyTable # Отображение таблиц в терминале

from IPython import get_ipython

# Типы данных
from typing import List, Dict, Union, Any, Optional

# Персональные
import openav                                     # Библиотека в целом
from openav.modules.core.exceptions import TypeMessagesError
from openav.modules.trml.shell import Shell       # Работа с Shell
from openav.modules.core.settings import Settings # Глобальный файл настроек

# ######################################################################################################################
# Константы
# ######################################################################################################################
TYPE_MESSAGES: List[str] = ['info', 'correct', 'error'] # Типы возможных сообщений

# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class CoreMessages(Settings):
    """Класс для сообщений"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__() # Выполнение конструктора из суперкласса

        self._libs_vers: str = self._('Версии установленных библиотек') + self._em
        self._package: str = self._('Пакет')

        self._trac_file: str = self._('Файл')
        self._trac_line: str = self._('Линия')
        self._trac_method: str = self._('Метод')
        self._trac_type_err: str = self._('Тип ошибки')

        self._undefined_message: str = '... ' + self._('неопределенное сообщение') + ' ...'
        self._wrong_type_messages: str = self._('Тип сообщения должен быть одним из "{}"') + self._em

# ######################################################################################################################
# Ядро модулей
# ######################################################################################################################
@dataclass
class Core(CoreMessages):
    """Класс-ядро модулей"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__() # Выполнение конструктора из суперкласса

        self._ap: Optional[argparse.ArgumentParser] = None # Парсер для параметров командной строки
        self._args: Optional[Dict[str, Any]] = None        # Аргументы командной строки

        self._space: int = 4 # Значение для количества пробелов в начале текста

        self._df_pkgs: pd.DataFrame = pd.DataFrame() # DataFrame c версиями установленных библиотек

        # ----------------------- Только для внутреннего использования внутри класса

        self.__max_space: int = 24 # Максимальное значение для количества пробелов в начале текста

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_notebook(self) -> bool:
        """Получение результата определения запуска библиотеки в Jupyter или аналогах

        Returns:
            bool: **True** если библиотека запущена в Jupyter или аналогах, в обратном случае **False**
        """

        return self.__is_notebook()

    @property
    def df_pkgs(self) -> pd.DataFrame:
        """Получение DataFrame c версиями установленных библиотек

        Returns:
            pd.DataFrame: **DataFrame** c версиями установленных библиотек
        """

        return self._df_pkgs

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (приватные)
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __is_notebook() -> bool:
        """Определение запуска библиотеки в Jupyter или аналогах

        .. note::
            private (приватный метод)

        Returns:
            bool: **True** если библиотека запущена в Jupyter или аналогах, в обратном случае **False**
        """

        try:
            # Определение режима запуска библиотеки
            shell = get_ipython().__class__.__name__
        except (NameError, Exception): return False # Запуск в Python
        else:
            if shell == 'ZMQInteractiveShell' or shell == 'Shell': return True
            elif shell == 'TerminalInteractiveShell': return False
            else: return False

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _traceback() -> Dict[str, Union[str, int]]:
        """Трассировка исключений

        .. note::
            protected (защищенный метод)

        Returns:
            Dict[str, Union[str, int]]: Словарь с описанием исключения
        """

        exc_type, exc_value, exc_traceback = sys.exc_info() # Получение информации об ошибке

        _trac = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__
        }

        return _trac

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы (сообщения)
    # ------------------------------------------------------------------------------------------------------------------

    def inv_args(self, class_name: str, build_name: str, out: bool = True) -> None:
        """Сообщение об указании неверных типов аргументов

        Args:
            class_name (str): Имя класса
            build_name (str): Имя метода/функции
            out (bool): Печатать процесс выполнения

        Returns:
            None
        """

        if type(out) is not bool: out = True

        trac = self._traceback() # Трассировка исключений

        try:
            # Проверка аргументов
            if type(class_name) is not str or not class_name or type(build_name) is not str or not build_name:
                raise TypeError
        except TypeError: class_name, build_name = __class__.__name__, self.inv_args.__name__

        inv_args = self._invalid_arguments.format(class_name + '.' + build_name)

        if self.is_notebook is False:
            if out is True:
                print('[{}{}{}] {}'.format(
                    self.color_red, datetime.now().strftime(self._format_time), self.text_end, inv_args
                ))

                indent = ('\r' + ' ' * self._space + '{}\n') * 4

                sys.stdout.write(indent.format(
                    f'{self._trac_file}: {trac["filename"]}',
                    f'{self._trac_line}: {trac["lineno"]}',
                    f'{self._trac_method}: {trac["name"]}',
                    f'{self._trac_type_err}: {trac["type"]}'
                ))
                sys.stdout.flush()

    def message_error(self, message: str, space: int = 0, out: bool = True) -> None:
        """Сообщение об ошибке

        Args:
            message (str): Сообщение
            space (int): Количество пробелов в начале текста
            out (bool): Отображение

        Returns:
            None
        """

        if type(out) is not bool: out = True

        try:
            # Проверка аргументов
            if type(message) is not str or not message or not (0 <= space <= self.__max_space): raise TypeError
        except TypeError: self.inv_args(__class__.__name__, self.message_error.__name__, out = out); return None

        if self.is_notebook is False:
            if out is True:
                print(' ' * space + '[{}{}{}] {}'.format(
                    self.color_red, datetime.now().strftime(self._format_time), self.text_end, message
                ))

    def message_info(self, message: str, space: int = 0, out: bool = True) -> None:
        """Информационное сообщение

        Args:
            message (str): Сообщение
            space (int): Количество пробелов в начале текста
            out (bool): Отображение

        Returns:
            None
        """

        if type(out) is not bool: out = True

        try:
            # Проверка аргументов
            if (type(message) is not str or not message or type(space) is not int
                or not (0 <= space <= self.__max_space)): raise TypeError
        except TypeError: self.inv_args(__class__.__name__, self.message_info.__name__, out = out); return None

        if self.is_notebook is False:
            if out is True: print(' ' * space + '[{}] {}'.format(datetime.now().strftime(self._format_time), message))

    def message_line(self, message: str, type_message: str = TYPE_MESSAGES[0], out: bool = True) -> str:
        """Информационное сообщение (в виде одной строки)

        Args:
            message (str): Сообщение
            type_message (str): Тип сообщения
            out (bool): Отображение

        Returns:
            str: Информационное сообщение (в виде одной строки)
        """

        if type(out) is not bool: out = True

        try:
            # Проверка аргументов
            if type(message) is not str or not message: raise TypeError
        except TypeError:
            self.inv_args(__class__.__name__, self.message_line.__name__, out = out)
            return self._undefined_message
        else:
            try:
                # Проверка типа сообщения
                if type(type_message) is not str or (type_message in TYPE_MESSAGES) is False: raise TypeMessagesError
            except TypeMessagesError:
                self.message_error(self._wrong_type_messages.format(
                    ', '.join(x.replace('.', '') for x in TYPE_MESSAGES)
                ), out = out); self._undefined_message
            else:
                # Тип сообщения
                if type_message == TYPE_MESSAGES[0]: tm = self.color_blue
                elif type_message == TYPE_MESSAGES[1]: tm = self.color_green
                elif type_message == TYPE_MESSAGES[2]: tm = self.color_red
                else: tm = self.text_bold

                if self.is_notebook is False:
                    if out is True: return ('{}' * 3).format(tm, message, self.text_end)

                return self._undefined_message

    def message_metadata_info(self, out: bool = True) -> None:
        """Информация об библиотеке

        Args:
            out (bool): Отображение

        Returns:
            None
        """

        if type(out) is not bool: out = True

        if self.is_notebook is False:
            space = " " * self._space

            generate_name_with_email = lambda list1, list2: ''.join(
                map(str, map(lambda l1, l2: f'{space * 2}{l1} [{l2}]\n', list1.split(', '), list2.split(', ')))
            )

            author = generate_name_with_email(
                openav.__author__ru__ if self.lang == 'ru' else openav.__author__en__, openav.__email__
            )

            maintainer = generate_name_with_email(
                openav.__maintainer__ru__ if self.lang == 'ru' else openav.__maintainer__en__,
                openav.__maintainer_email__
            )

            if out is True:
                print(('{}' * 5).format(
                    f'[{datetime.now().strftime(self._format_time)}] {self._metadata[0]}:\n',
                    f'{space}{self._metadata[1]}:\n{author}',
                    f'{space}{self._metadata[2]}:\n{maintainer}',
                    f'{space}{self._metadata[3]}: {openav.__release__}\n',
                    f'{space}{self._metadata[4]}: {openav.__license__}'
                ))

    def message_progressbar(self, message: str = '', space: int = 0, close: bool = False, out: bool = True) -> str:
        """Информационный индикатор выполнения

        Args:
            message (str): Сообщение
            space (int): Количество пробелов в начале текста
            close (bool): Закрыть информационный индикатор
            out (bool): Отображение

        Returns:
            None
        """

        if type(out) is not bool: out = True
        if type(close) is not bool: close = False

        if close is True: message = 'Закрыть'

        try:
            # Проверка аргументов
            if (type(message) is not str or not message or type(space) is not int
                or not (0 <= space <= self.__max_space)): raise TypeError
        except TypeError: self.inv_args(__class__.__name__, self.message_progressbar.__name__, out = out); return None

        if self.is_notebook is False:
            if out is True:
                message = '\r' + self.clear_line + (' ' * space) + '[{}] {}'.format(
                    datetime.now().strftime(self._format_time), message
                )
                if close is True: message = '\n'

                sys.stdout.write(message)
                sys.stdout.flush()

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def libs_vers(self, out: bool = True) -> bool:
        """Получение и отображение версий установленных библиотек

        Args:
            out (bool): Отображение

        Returns:
            bool: **True** если версии установленных библиотек отображены, в обратном случае **False**
        """

        # Сброс
        self._df_pkgs = pd.DataFrame() # Пустой DataFrame

        try:
            # Проверка аргументов
            if type(out) is not bool: raise TypeError
        except TypeError: self.inv_args(__class__.__name__, self.libs_vers.__name__, out = out); return False
        else:
            pkgs = {
                'Package': [
                    'PyTorch', 'TorchAudio', 'TorchVision', 'NumPy', 'Pandas', 'PyAV', 'FileType', 'IPython',
                    'Colorama', 'Prettytable'
                ],
                'Version': [i.__version__ for i in [
                    torch, torchaudio, torchvision, np, pd, av, filetype, IPython, colorama, prettytable
                ]]
            }

            self._df_pkgs = pd.DataFrame(data = pkgs) # Версии используемых библиотек
            self._df_pkgs.index += 1

            if self.is_notebook is False:
                # Вывод сообщения
                if out is True:
                    table_terminal = PrettyTable()
                    table_terminal.add_column(self._package, self._df_pkgs['Package'].values)
                    table_terminal.add_column(self._metadata[3], self._df_pkgs['Version'].values)
                    table_terminal.align = 'l'

                    self.message_info(self._libs_vers, space = 0, out = out)
                    print(table_terminal)

    def build_args(self, description: str, conv_to_dict: bool = True, out: bool = True) -> Dict[str, Any]:
        """Построение аргументов командной строки

        Args:
            description (str): Описание парсера командной строки
            conv_to_dict (bool): Преобразование списка аргументов командной строки в словарь
            out (bool): Печатать процесс выполнения

        Returns:
            Dict[str, Any]: Словарь со списком аргументов командной стройки
        """

        # Проверка аргументов
        if type(description) is not str or not description or type(conv_to_dict) is not bool or type(out) is not bool:
            self.inv_args(__class__.__name__, self.build_args.__name__, out = out); return {}

        # Парсер для параметров командной строки
        self._ap = argparse.ArgumentParser(description = description)

        if conv_to_dict is True:
            return vars(self._ap.parse_args()) # Преобразование списка аргументов командной строки в словарь

    def clear_shell(self, cls: bool = True, out: bool = True) -> bool:
        """Очистка консоли

        Args:
            cls (bool): Очистка консоли
            out (bool): Печатать процесс выполнения

        Returns:
             bool: **True** если консоль очищена, в обратном случае **False**
        """

        # Проверка аргументов
        if type(cls) is not bool or type(out) is not bool:
            self.inv_args(__class__.__name__, self.clear_shell.__name__, out = out); return False

        if cls is True: Shell.clear() # Очистка консоли
        return True
