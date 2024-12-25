.. include:: ../../reuse_content/general.rst

.. |a_download_mac| raw:: html

    <a href="https://www.python.org/downloads/macos/" target="_blank">

Установка и обновление на операционных системах MacOS/Linux
===========================================================

Подготовка рабочего пространства
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Перейти на сайт и |a_download_mac| скачать необходимый дистрибутив |/a| (например Python 3.12.8)

.. figure:: ../../_static/img/installation/mac/download_mac.png
        :align: center
        :alt: Загрузка дистрибутива

        |br|

2. Установить Python

.. figure:: ../../_static/img/installation/mac/install.png
        :align: center
        :alt: Завершенный процесс установки Python

        |br|

3. Установить библиотеку virtualenv для создания изолированных сред Python:

.. code-block:: sh

   pip install virtualenv

.. figure:: ../../_static/img/installation/mac/virtualenv.png
        :align: center
        :alt: Завершенный процесс установки библиотеки virtualenv

        |br|

4. Добавить путь к вашей версии Python 3.10 и выше в PATH:

.. code-block:: sh

   export PATH="ПУТЬ_К_PYTHON:$PATH"

.. figure:: ../../_static/img/installation/mac/path.png
        :align: center
        :alt: Добавление пути к установленному Python в PATH

        |br|

Создание изолированной среды OpenAV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Создать директорию envs и перейти в нее:

.. code-block:: sh

   cd ПУТЬ_К_ДИРЕКТОРИИ

.. figure:: ../../_static/img/installation/mac/cd_envs.png
        :align: center
        :alt: Переход в директорию envs через команду «cd»

        |br|

2. Разместить новую виртуальную среду OpenAV в указанной директории с указанной версией Python:

.. code-block:: sh

   virtualenv --python=ПУТЬ_К_PYTHON OpenAV

.. figure:: ../../_static/img/installation/mac/dir_placed.png
        :align: center
        :alt: Размещение новой виртуальной среды OpenAV в указанной директории с указанной версией Python

        |br|

3. Активировать созданную виртуальную среду OpenAV в директории:

.. code-block:: sh

   source OpenAV/bin/activate

.. figure:: ../../_static/img/installation/mac/activate.png
        :align: center
        :alt: Активация созданной виртуальной среды OpenAV в директории

        |br|

Работа с библиотекой OpenAV
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Установить библиотеку OpenAV:

.. code-block:: sh

   pip install openav

.. figure:: ../../_static/img/installation/mac/openav.png
        :align: center
        :alt: Установленная библиотека OpenAV со всеми зависимостями

        |br|
