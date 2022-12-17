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

from dataclasses import dataclass # Класс данных

# Персональные
from openav.modules.core.messages import Messages # Сообщения

# ######################################################################################################################
# Константы
# ######################################################################################################################

# ######################################################################################################################
# Настройки
# ######################################################################################################################
@dataclass
class Settings(Messages):
    """Класс для настроек"""

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