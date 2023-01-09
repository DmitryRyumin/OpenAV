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

class IsNestedCatalogsNotFoundError(CustomException):
    """Вложенные директории, где хранятся данные не найдены"""
    pass

class IsNestedDirectoryVNotFoundError(CustomException):
    """Вложенная директория, для видеофрагментов не найдена"""
    pass

class IsNestedDirectoryANotFoundError(CustomException):
    """Вложенная директория, для аудиофрагментов не найдена"""
    pass