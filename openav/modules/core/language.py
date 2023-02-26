#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Определение языка
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
# Интернационализация (I18N) и локализация (L10N) (см. https://www.loc.gov/standards/iso639-2/php/code_list.php)
#     - brew install gettext (Если не установлен)
#     - brew link gettext --force
#     1. gettext --help
#     2. locate pygettext.py
#     3. /usr/local/Cellar/python@3.9/3.9.13_4/Frameworks/Python.framework/Versions/3.9/share/doc/python3.9/examples/
#        Tools/i18n/pygettext.py -d openav -o openav/modules/locales/base.pot openav
#     4. msgfmt --help
#     5. locate msgfmt.py
#     6. /usr/local/Cellar/python@3.9/3.9.13_4/Frameworks/Python.framework/Versions/3.9/share/doc/python3.9/examples/
#        Tools/i18n/msgfmt.py openav/modules/locales/en/LC_MESSAGES/base.po openav/modules/locales/en/LC_MESSAGES/base

# Подавление Warning
import warnings

for warn in [UserWarning, FutureWarning]:
    warnings.filterwarnings("ignore", category=warn)

from dataclasses import dataclass, field  # Класс данных

import os  # Взаимодействие с файловой системой
import sys  # Доступ к некоторым переменным и функциям Python
import gettext  # Формирование языковых пакетов
import inspect  # Инспектор
import argparse  # Парсинг аргументов и параметров командной строки

# Типы данных
from typing import List, Dict, Optional
from types import MethodType

# Персональные
from openav.modules.core.logging import Logging  # Логирование

# ######################################################################################################################
# Константы
# ######################################################################################################################
LANG: str = "ru"  # Язык


# ######################################################################################################################
# Интернационализация (I18N) и локализация (L10N)
# ######################################################################################################################
@dataclass
class Language(Logging):
    """Класс для интернационализации (I18N) и локализации (L10N)

    Args:
        lang (str): Язык
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    lang: str
    __lang: str = field(default=LANG, init=False, repr=False)
    """Язык

    .. note::
        private (приватный аргумент)
    """

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        self.__i18n: List[Dict[str, MethodType]] = self.__get_locales()  # Получение языковых пакетов
        self._: MethodType = self.__set_locale(self.lang)  # Установка языка

        argparse._ = self.__i18n[1][self.lang]  # Установка языка для парсинга аргументов и параметров командной строки

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def lang(self) -> str:
        """Получение/установка текущего языка

        Args:
            (str): Язык, доступные варианты:

                * ``"ru"`` - Русский язык (``по умолчанию``)
                * ``"en"`` - Английский язык

        Returns:
            str: Язык
        """

        return self.__lang

    @lang.setter
    def lang(self, lang: str):
        """Установка текущего языка"""

        try:
            # Проверка аргументов
            if type(lang) is not str or not lang or (lang in self.locales) is False:
                raise TypeError
        except TypeError:
            pass
        else:
            self.__lang = lang

    @property
    def locales(self) -> List[str]:
        """Получение поддерживаемых языков

        Returns:
            List[str]: Список поддерживаемых языков
        """

        return self.__get_languages()  # Поддерживаемые языки

    @property
    def path_to_locales(self) -> str:
        """Получение директории с языковыми пакетами

        Returns:
            str: Директория с языковыми пакетами
        """

        # Нормализация пути
        return os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "locales")))

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (приватные)
    # ------------------------------------------------------------------------------------------------------------------

    def __get_languages(self) -> List[Optional[str]]:
        """Получение поддерживаемых языков

        .. note::
            private (приватный метод)

        Returns:
            List[Optional[str]]: Список поддерживаемых языков
        """

        # Директория с языками найдена
        if os.path.exists(self.path_to_locales):
            # Формирование списка с подерживаемыми языками
            return next(os.walk(self.path_to_locales))[1]

        return []

    def __get_locales(self) -> List[Dict[str, MethodType]]:
        """Получение языковых пакетов

        .. note::
            private (приватный метод)

        Returns:
            List[Dict[str, MethodType]]: Список словарей с языковыми пакетами
        """

        # Языки
        tr_argparse = {}
        trs_base = {}

        # Проход по всем языкам
        for curr_lang in self.locales:
            tr_argparse[curr_lang] = gettext.translation(
                "argparse",  # Домен
                localedir=self.path_to_locales,  # Директория с поддерживаемыми языками
                languages=[curr_lang],  # Язык
                fallback=True,  # Отключение ошибки
            ).gettext

            trs_base[curr_lang] = gettext.translation(
                "base",  # Домен
                localedir=self.path_to_locales,  # Директория с поддерживаемыми языками
                languages=[curr_lang],  # Язык
                fallback=True,  # Отключение ошибки
            ).gettext

        return [trs_base, tr_argparse]

    def __set_locale(self, lang: str = LANG) -> MethodType:
        """Установка языка

        .. note::
            private (приватный метод)

        Args:
            lang (str): Язык

        Returns:
            MethodType: MethodType перевода строк на один из поддерживаемых языков если метод запущен через конструктор
        """

        try:
            # Проверка аргументов
            if type(lang) is not str:
                raise TypeError
        except TypeError:
            pass
        else:
            # Проход по всем поддерживаемым языкам
            for curr_lang in self.locales:
                # В аргументах командной строки не найден язык
                if curr_lang not in sys.argv:
                    # В аргументах метода не найден язык
                    if lang != curr_lang:
                        continue

                self.lang = curr_lang  # Изменение языка

            # Метод запущен в конструкторе
            if inspect.stack()[1].function == "__init__" or inspect.stack()[1].function == "__post_init__":
                return self.__i18n[0][self.lang]
            else:
                self._ = self.__i18n[0][self.lang]
