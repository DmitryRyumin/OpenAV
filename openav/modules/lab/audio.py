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

# Персональные
from openav.modules.file_manager.json_manager import Json # Класс для работы с Json

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