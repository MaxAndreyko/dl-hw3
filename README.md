# Репозиторий посвящен ДЗ №3: RNN и языковые модели
## Структура проекта
```
| .gitignore                         <- Файлы, которые не надо пушить в гит
| .pre-commit-config.yaml            <- Конфиг для pre-commit
| .requirements.txt                  <- Необходимые зависимости для проекта
| .setup.cfg                         <- Конфиг для flake8
| .setup.py                          <- Создает python-модуль из папки src/
| .shw-03-rnn.py
│
├── src                              <- Содержит все python-модули
│   ├── __init__.py                  <- Для инициализации папки как модуля
│   ├── dataset.py                   <- Создание датасета из текстов
│   ├── model.py                     <- Создание языковой модели
│   └── train.py                     <- Запуск обучения и валидации модели
│   └── utils.py                     <- Функции для задания 7
│
├── data                                       <- Папка для хранения данных
│   ├── external                               <- Внешние данные
│   │   └── jokes.txt                          <- Исходные датасет анекдотов
│   ├── interim                                <- Папка с промежуточными данными
│   │   ├── bpe_2000_generated_jokes.txt       <- Сгенерированные анекдоты с токенизатором BPE (vocab_size=2000)
│   │   ├── bpe_8000_generated_jokes.txt       <- Сгенерированные анекдоты с токенизатором BPE (vocab_size=8000)
│   │   ├── lsa_generated_jokes.txt            <- Сгенерированные анекдоты для задания с LSA с токенизатором Unigram (vocab_size=2000)
│   │   └── lsa_val_jokes.txt                  <- Валидационные анекдоты для задания с LSA с токенизатором Unigram (vocab_size=2000)
│   └── models                                 <- Папка для хранения моделей
│   │   ├── model_lstm.pt                      <- Модель LSTM, токенизатор BPE (vocab_size=2000)
│   │   ├── model_lstm_bpe_large.pt            <- Модель LSTM, токенизатор BPE (vocab_size=8000)
│   │   ├── model_lstm_unigram.pt              <- Модель LSTM, токенизатор Unigram (vocab_size=2000)
│   │   └── model_rnn.pt                       <- Модель RNN, токенизатор BPE (vocab_size=2000)
```
