#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Сообщения
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
# Подавление Warning
import warnings
for warn in [UserWarning, FutureWarning]: warnings.filterwarnings('ignore', category = warn)

from dataclasses import dataclass # Класс данных

# Типы данных
from typing import List

# Персональные
from openav.modules.core.language import Language # Определение языка

# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class Messages(Language):
    """Класс для сообщений"""

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__() # Выполнение конструктора из суперкласса

        self._metadata: List[str] = [
            self._('OpenAV - библиотека распознавания речевых команд на пользовательском словаре с использованием '
                   'аудиовизуальных данных диктора'),
            self._('Авторы'), self._('Сопровождающие'), self._('Версия'), self._('Лицензия')
        ]

        self._format_time: str = '%Y-%m-%d %H:%M:%S' # Формат времени

        self._em: str = ' ...' # Конец сообщений (End Messages)

        self._invalid_arguments: str = self._('Неверные типы или значения аргументов в "{}"') + self._em
        self._unknown_err: str = self._(
            'Не обработанную ошибку необходимо проанализировать и выявить причину'
        ) + self._em

        self._from_precent: str = self._('из')
        self._curr_progress: str = '{} ' + self._from_precent + ' {} ({}%)'+ self._em + ' {}' + self._em
