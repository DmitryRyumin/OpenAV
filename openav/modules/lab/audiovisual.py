#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Мультимодальное объединение аудио- и видеомодальностей
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################
# Подавление Warning
import warnings

for warn in [
    UserWarning,
    FutureWarning,
]:
    warnings.filterwarnings(
        "ignore",
        category=warn,
    )

from dataclasses import (
    dataclass,
)  # Класс данных

# Персональные
from openav.modules.lab.audio import (
    Audio,
)  # Аудиомодальность
from openav.modules.lab.video import (
    Video,
)  # Видеомодальность

# ######################################################################################################################
# Константы
# ######################################################################################################################


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class AVMessages(Audio, Video):
    """Класс для сообщений

    Args:
        lang (str): Смотреть :attr:`~openav.modules.core.language.Language.lang`
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса


# ######################################################################################################################
# Видео
# ######################################################################################################################
@dataclass
class AV(AVMessages):
    """Класс для мультимодального объединения аудио- и видеомодальностей

    Args:
        lang (str): Смотреть :attr:`~openav.modules.core.language.Language.lang`
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

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
