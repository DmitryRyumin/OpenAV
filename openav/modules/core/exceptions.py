#!/usr/bin/env python
# -*- coding: utf-8 -*-


class CustomException(Exception):
    """Класс для всех пользовательских исключений"""

    pass


class TypeEncodeVideoError(CustomException):
    """Указан неподдерживаемый тип кодирования видео"""

    pass


class PresetCFREncodeVideoError(CustomException):
    """Указан неподдерживаемый параметр обеспечивающий определенную скорость кодирования и сжатия видео"""

    pass


class TypeMessagesError(CustomException):
    """Указан неподдерживаемый тип сообщения"""

    pass


class SRInputTypeError(CustomException):
    """Указан неподдерживаемый тип файла для распознавания речи"""

    pass


class SamplingRateError(CustomException):
    """Указана неподдерживаемая частота дискретизации речевого сигнала"""

    pass


class WindowSizeSamplesError(CustomException):
    """Указано неподдерживаемое количество выборок в каждом окне"""

    pass


class IsNestedCatalogsNotFoundError(CustomException):
    """Вложенные директории, где хранятся данные не найдены"""

    pass


class IsNestedDirectoryVNotFoundError(CustomException):
    """Вложенная директория, для видеофрагментов не найдена"""

    pass


class IsNestedDirectoryANotFoundError(CustomException):
    """Вложенная директория, для аудиофрагментов не найдена"""

    pass


class InvalidContentLength(CustomException):
    """Не определен размер файла для загрузки"""

    pass


class CropPXError(CustomException):
    """Указан неверный диапазон обрезки в пикселях"""

    pass


class CropPercentsError(CustomException):
    """Указан неверный диапазон обрезки в процентах"""

    pass


class FlipLRProbabilityError(CustomException):
    """Указано неверное значение вероятности отражения по вертикальной оси"""

    pass


class FlipUDProbabilityError(CustomException):
    """Указано неверное значение вероятности отражения по горизонтальной оси"""

    pass


class BlurError(CustomException):
    """Указан неверный диапазон значений размытия"""

    pass


class ScaleError(CustomException):
    """Указан неверный диапазон значений масштабирования"""

    pass


class RotateError(CustomException):
    """Указан неверный диапазон значений угла наклона"""

    pass


class ContrastError(CustomException):
    """Указан неверный диапазон значений контрастности"""

    pass


class MixUpAlphaError(CustomException):
    """Указан неверный коэффициент для MixUp-аугментации"""

    pass
