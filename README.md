# Распознавание рукописного кириллического текста

Данный проект посвящен разработке гибридной нейронной сети для распознавания рукописного кириллического текста.

## Описание

Целью является создание модели, сочетающей трансформеры и сверточные сети, для эффективного распознавания рукописного текста.

Основные задачи:
1. Сбор и подготовка данных, включая использование [Cyrillic Handwriting Dataset](https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset).
2. Разработка гибридной архитектуры с сверточными и трансформерными компонентами.
3. Обучение и тестирование модели с оценкой по метрикам CER и WER.

## Использование

1. Клонировать репозиторий
2. Установить зависимости
3. Загрузить предобученную модель
4. Использовать класс `TransformerModel` для предсказаний

Подробности в файле `4_CreateTrainTestModel.ipynb`, `Report.pdf`.
                                          
