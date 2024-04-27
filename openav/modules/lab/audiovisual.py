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

import os

from dataclasses import dataclass  # Класс данных

# Типы данных
from typing import List, Dict

from pathlib import Path  # Работа с путями в файловой системе
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from lion_pytorch import Lion
import datetime

# Персональные
from openav.modules.core.exceptions import IsNestedCatalogsNotFoundError
from openav.modules.lab.audio import Audio  # Аудиомодальность
from openav.modules.lab.video import Video  # Видеомодальность
from openav.modules.nn.av_dataset import AVDataset
from openav.modules.nn.utils import fix_seeds, train_one_epoch, val_one_epoch
from openav.modules.nn.models import AVModel

# ######################################################################################################################
# Константы
# ######################################################################################################################

SUBFOLDERS: List[str] = ["train", "val", "test"]
SHAPE_AUDIO: List[str] = ["channels", "n_mels", "samples"]
SHAPE_VIDEO: List[str] = ["frames", "channels", "width", "height"]
EXT_AV_VIDEO: List[str] = ["mov", "mp4", "webm"]  # Расширения искомых файлов
EXT_MODELS: str = "pt"


# ######################################################################################################################
# Сообщения
# ######################################################################################################################
@dataclass
class AVMessages(Audio, Video):
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

        self._subfolders_search: str = (
            self._('Поиск вложенных директорий в директории "{}" (глубина вложенности: {})') + self._em
        )

        self._files_audiovisual_find: str = (
            self._('Поиск файлов с расширениями "{}" в директории "{}" (глубина вложенности: {})') + self._em
        )

        self._files_audiovisual_find: str = (
            self._('Поиск файлов с расширениями "{}" в директории "{}" (глубина вложенности: {})') + self._em
        )

        self._sampling_nn: str = (
            self._("Разбиение найденных файлов на выборки (обучающая, валидационная, тестовая)") + self._em
        )

        self._sampling_nn_error: str = (
            self._(
                "Минимум одна выборка пустая (обучающая - {}, валидационная - {}, тестовая - {}) или количество меток "
                + "не совпадает с количеством файлов"
            )
            + self._em
        )

        self._sampling_nn_true: str = self._("Обучающая - {} {}, валидационная - {} {}, тестовая - {} {}") + self._em
        self._format_percentage = lambda x: "({}%)".format(x)


# ######################################################################################################################
# Видео
# ######################################################################################################################
@dataclass
class AV(AVMessages):
    """Класс для мультимодального объединения аудио- и видеомодальностей

    Args:
        path_to_logs (str): Смотреть :attr:`~openav.modules.core.logging.Logging.path_to_logs`
        lang (str): Смотреть :attr:`~openav.modules.core.language.Language.lang`
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------------------------------------------------------

    def __post_init__(self):
        super().__post_init__()  # Выполнение конструктора из суперкласса

        # ----------------------- Только для внутреннего использования внутри класса

        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.__model_session = "created_at_%d_%m_%Y_%H-%M"

    # ------------------------------------------------------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (приватные)
    # ------------------------------------------------------------------------------------------------------------------

    def __get_hierarchy_from_paths(self, paths, num_levels=0):
        hierarchies = []

        for path in paths:
            hierarchy = []
            while True:
                path, dir_name = os.path.split(path)
                if not dir_name:
                    hierarchy.append(path)
                    break
                hierarchy.append(dir_name)
            if num_levels > 0:
                hierarchy = hierarchy[:num_levels]
            if hierarchy[-1] in SUBFOLDERS:
                hierarchies.append(hierarchy)

        return hierarchies

    # ------------------------------------------------------------------------------------------------------------------
    # Внутренние методы (защищенные)
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Внешние методы
    # ------------------------------------------------------------------------------------------------------------------

    def train_audiovisual(
        self,
        subfolders: Dict[str, str],
        n_classes: int,
        classes: List[str],
        encoder_decoder: int,
        batch_size: int,
        max_segment: int,
        patience: int,
        epochs: int,
        seed: int,
        leaning_rate: float,
        hidden_units: int,
        hidden_features: int,
        input_dim: int,
        shape_audio: Dict[str, int],
        shape_video: Dict[str, int],
        path_to_model_fa: str,
        path_to_model_fv: str,
        path_to_save_models: str,
        out: bool = True,
    ) -> bool:
        """Автоматическое обучение на аудиовизуальных данных

        Args:
            subfolders (Dict[str, str]): Словарь с подкаталогами с данными
            n_classes (int): Количество классов
            classes (List[str]): Список классов
            encoder_decoder (int): Количество энкодеров и декодеров
            batch_size (int): Размер батча
            max_segment (int): Максимальная длительность сегмента видео
            patience (int): Количество неудачных эпох
            epochs (int): Количество эпох
            seed (int): Начальное состояние обучения
            leaning_rate (float): Скорость обучения
            hidden_units (int): Количество скрытых нейронов
            hidden_features (int): Количество скрытых признаков
            input_dim (int): Количество входных признаков
            shape_audio (Dict[str, int]): Входная размерность аудио лог-мел спектрограммы
            shape_video (Dict[str, int]): Входная размерность видеокадров
            path_to_model_fa (str): Путь к нейросетевой модели (аудио)
            path_to_model_fv (str): Путь к нейросетевой модели (видео)
            path_to_save_models (str): Путь к директории для сохранения моделей
            out (bool) Отображение

        Returns:
            bool: **True** если автоматическое обучение на аудиовизуальных данных произведено, в обратном случае
            **False**
        """

        try:
            # Проверка аргументов
            if (
                type(subfolders) is not dict
                or len(subfolders) == 0
                or not all(subfolder in subfolders for subfolder in SUBFOLDERS)
                or type(n_classes) is not int
                or not (0 < n_classes)
                or type(classes) is not list
                or len(classes) == 0
                or type(encoder_decoder) is not int
                or not (1 <= encoder_decoder <= 50)
                or type(batch_size) is not int
                or not 0 <= batch_size
                or type(max_segment) is not int
                or not (1 <= max_segment <= 10)
                or type(epochs) is not int
                or not (0 <= epochs <= 1000)
                or type(seed) is not int
                or not (0 < seed)
                or type(leaning_rate) is not float
                or type(hidden_units) is not int
                or not (0 < hidden_units)
                or type(hidden_features) is not int
                or not (0 < hidden_features)
                or type(patience) is not int
                or not (0 < patience)
                or type(input_dim) is not int
                or not (0 < input_dim)
                or type(shape_audio) is not dict
                or len(shape_audio) == 0
                or not all(shape in shape_audio for shape in SHAPE_AUDIO)
                or type(shape_video) is not dict
                or len(shape_video) == 0
                or not all(shape in shape_video for shape in SHAPE_VIDEO)
                or type(path_to_model_fa) is not str
                or not path_to_model_fa
                or type(path_to_model_fv) is not str
                or not path_to_model_fv
                or not Path(path_to_model_fa).is_file()
                or Path(path_to_model_fa).suffix.replace(".", "") != EXT_MODELS
                or not Path(path_to_model_fv).is_file()
                or Path(path_to_model_fv).suffix.replace(".", "") != EXT_MODELS
                or type(path_to_save_models) is not str
                or not path_to_save_models
                or type(out) is not bool
            ):
                raise TypeError
        except TypeError:
            self.inv_args(__class__.__name__, self.train_audiovisual.__name__, out=out)
            return False
        else:
            depth = 3
            classes = [cls.lower() for cls in classes]

            shape_audio = (
                None,
                max_segment,
                shape_audio[SHAPE_AUDIO[0]],
                shape_audio[SHAPE_AUDIO[1]],
                shape_audio[SHAPE_AUDIO[2]],
            )
            shape_video = (
                None,
                max_segment,
                shape_video[SHAPE_VIDEO[0]],
                shape_video[SHAPE_VIDEO[1]],
                shape_video[SHAPE_VIDEO[3]],
                shape_video[SHAPE_VIDEO[2]],
            )

            # Информационное сообщение
            self.message_info(
                self._subfolders_search.format(
                    self.message_line(self.path_to_dataset),
                    self.message_line(str(depth)),
                ),
                out=out,
            )

            nested_paths = self.get_paths(self.path_to_dataset, depth=depth, out=False)

            # Вложенные директории не найдены
            try:
                if len(nested_paths) == 0:
                    raise IsNestedCatalogsNotFoundError
            except IsNestedCatalogsNotFoundError:
                self.message_error(self._subfolders_not_found, space=self._space, out=out)
                return False

            # Информационное сообщение
            self.message_info(
                self._files_audiovisual_find.format(
                    self.message_line(", ".join(x.replace(".", "") for x in EXT_AV_VIDEO)),
                    self.message_line(self.path_to_dataset),
                    self.message_line(str(depth)),
                ),
                out=out,
            )

            hierarchy_from_paths = self.__get_hierarchy_from_paths(nested_paths, depth)

            # Информационное сообщение
            self.message_info(
                self._sampling_nn,
                out=out,
            )

            path_train, lb_train = [], []
            path_val, lb_val = [], []
            path_test, lb_test = [], []

            # Проход по всем вложенным директориям
            for nested_path in hierarchy_from_paths:
                if nested_path[0].lower() in classes:
                    # Формирование списка с видеофайлами
                    for p in Path(os.path.join(self.path_to_dataset, *reversed(nested_path))).glob("*"):
                        # Добавление текущего пути к видеофайлу в список
                        if p.suffix.lower().replace(".", "") in EXT_AV_VIDEO:
                            [index for index, cls in enumerate(classes) if cls.lower() == nested_path[0].lower()]

                            if nested_path[-1] == subfolders[SUBFOLDERS[0]]:
                                path_train.append(p.resolve())
                                lb_train.append(classes.index(nested_path[0].lower()))
                            elif nested_path[-1] == subfolders[SUBFOLDERS[1]]:
                                path_val.append(p.resolve())
                                lb_val.append(classes.index(nested_path[0].lower()))
                            elif nested_path[-1] == subfolders[SUBFOLDERS[2]]:
                                path_test.append(p.resolve())
                                lb_test.append(classes.index(nested_path[0].lower()))
                            else:
                                pass

            # Директории с поднаборами данных не содержат визуальных файлов с необходимыми расширениями
            try:
                len_path_train = len(path_train)
                len_lb_train = len(lb_train)

                len_path_val = len(path_val)
                len_lb_val = len(lb_val)

                len_path_test = len(path_test)
                len_lb_test = len(lb_test)

                if (
                    len_path_train == 0
                    or len_lb_train == 0
                    or len_path_val == 0
                    or len_lb_val == 0
                    or len_path_test == 0
                    or len_lb_test == 0
                    or len_path_train != len_lb_train
                    or len_path_val != len_lb_val
                    or len_path_test != len_lb_test
                ):
                    raise ValueError
            except ValueError:
                self.message_error(
                    self._sampling_nn_error.format(
                        self.message_line(str(len_path_train)),
                        self.message_line(str(len_path_val)),
                        self.message_line(str(len_lb_test)),
                    ),
                    space=self._space,
                    out=out,
                )
                return False
            except Exception:
                self.message_error(self._unknown_err, space=self._space, out=out)
                return False
            else:
                total_samples = len_path_train + len_path_val + len_path_test

                train_percentage = round((len_path_train / total_samples) * 100, 2)
                val_percentage = round((len_path_val / total_samples) * 100, 2)
                test_percentage = round((len_path_test / total_samples) * 100, 2)

                self.message_info(
                    self._sampling_nn_true.format(
                        self.message_line(str(len_path_train)),
                        self.message_line(self._format_percentage(train_percentage)),
                        self.message_line(str(len_path_val)),
                        self.message_line(self._format_percentage(val_percentage)),
                        self.message_line(str(len_path_test)),
                        self.message_line(self._format_percentage(test_percentage)),
                    ),
                    space=self._space,
                    out=out,
                )

                train_data = AVDataset(
                    path_files=path_train, labels=lb_train, subset="train", len_video=29, max_segment=2
                )

                train_data = AVDataset(path_files=path_train, labels=lb_train, subset="train", max_segment=max_segment)
                test_data = AVDataset(path_files=path_test, labels=lb_test, subset="test", max_segment=max_segment)
                val_data = AVDataset(path_files=path_val, labels=lb_val, subset="val", max_segment=max_segment)

                train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
                test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

                torch.autograd.set_detect_anomaly(True)

                stop_flag_training = 0

                fix_seeds(seed)

                model = AVModel(
                    shape_audio=shape_audio,
                    shape_video=shape_video,
                    input_dim=input_dim,
                    h_u=hidden_units,
                    h_f=hidden_features,
                    n_class=n_classes,
                    encoder_decoder=encoder_decoder,
                ).to(self.__device)
                model.feature_audio.load_state_dict(torch.load(path_to_model_fa))
                model.feature_video.load_state_dict(torch.load(path_to_model_fv))

                for name, param in model.named_parameters():
                    if any(layer_name.split(".")[0] in name for layer_name in ["feature_audio", "feature_video"]):
                        param.requires_grad = False

                criterion = CrossEntropyLoss()
                optimizer = Lion(model.parameters(), lr=leaning_rate, weight_decay=0)

                max_acc = 0
                max_test_acc = 0

                date_time = datetime.datetime.now()
                date_path = date_time.strftime(self.__model_session)
                session_path = os.path.join(self.path_to_save_models, date_path)

                # Создание директории, где хранятся модели
                if self.create_folder(session_path, out=False) is False:
                    return False

                for e in range(1, epochs, 1):
                    print(f"Эпоха: {e} из {epochs}")

                    # обучение
                    model.train()
                    avg_loss = train_one_epoch(train_dataloader, optimizer, criterion, model, self.__device)

                    # валидирование
                    model.eval()
                    acc, avg_vloss = val_one_epoch(val_dataloader, criterion, model, self.__device)

                    # тестирование
                    model.eval()
                    test_acc, avg_tloss = val_one_epoch(test_dataloader, criterion, model, self.__device)

                    # аналогично можно добавить тестирования

                    print("LOSS train {} valid {} test {}".format(avg_loss, avg_vloss, avg_tloss))

                    if max_acc < acc:
                        stop_flag_training = 0
                        print(f"validation acc: {acc} | test accuracy: {test_acc}")
                        print(f"Validation Acc Increased ({max_acc:.6f}--->{acc:.6f}) \t Saving The Model")
                        max_acc = acc
                        torch.save(
                            model.state_dict(), "{}/e{}_{:.6f}.pth".format(session_path, e, acc)
                        )  # записываем веса модели

                    if max_test_acc < test_acc:
                        print(f"test accuracy: {test_acc}")
                        print(f"Test Acc Increased ({max_test_acc:.6f}--->{test_acc:.6f})")
                        max_test_acc = test_acc

                    else:
                        stop_flag_training += 1

                    if stop_flag_training > patience:
                        print("Обучение закончилось")
                        break
