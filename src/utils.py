from sklearn.manifold import TSNE
from string import punctuation
from pathlib import Path
import torch
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from src.dataset import TextDataset


def create_tsne_texts_embeddings(sp_model_prefix: str, model_path: str,
                            generated_texts_path: str,
                            data_file: str = 'data/external/jokes.txt',
                            tsne_components: int = 2,
                            tsne_perplexity: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Creates embeddings for validation end generated
       texts appropriate for plotting in 2D space

    Args:
        sp_model_prefix (str): Path prefix to load tokenizer model
        model_path (str): Path to .pt model weights
        generated_texts_path (str): Path to .txt with generated texts (save/load)
        data_file (str, optional): Path to .txt file with source texts. Defaults to 'data/external/jokes.txt'.
        tsne_components (int, optional): Number of components for TSNE. Defaults to 2.
        tsne_perplexity (int, optional): TSNE perplexity parameter. Defaults to 25.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Validation and generated TSNE components
    """

    print("Получение валидационных текстов...")
    valid_set = TextDataset(data_file=data_file, train=False, sp_model_prefix=sp_model_prefix)
    valid_set_texts = valid_set.texts

    print("Получение prefix для генерации текстов...")
    train_prefixes = list(map(lambda x: ' '.join(x.translate(str.maketrans('', '', punctuation)).strip().split()[:2]), valid_set_texts))

    print("Генерация текстов моделью...")
    generated_texts_path = Path(generated_texts_path)
    if not generated_texts_path.exists() or not generated_texts_path.is_file():

        model_path = Path(model_path)
        if not model_path.exists() or not model_path.is_file():
            print(f"Файла модели по пути {str(model_path)} не существует или файл поврежден")
            return None, None
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))

        # Генерируем тексты с помощью модели
        generated_texts = []
        for prefix in train_prefixes:
            generated = model.inference(prefix)
            generated_texts.append(generated)

            with open(generated_texts_path, "w") as f:
                for gen_text in generated_texts:
                    f.write(f"{gen_text}\n")

    else:
        with open(generated_texts_path, "r") as f:
            generated_texts = f.readlines()

    
    print("Объединение выборок в корпус...")
    # Преобразовываем тексты в векторы индексов
    valid_set_ids = np.array([list(ids)[0].numpy() for ids in valid_set]) 
    generated_ids = valid_set.text2ids(generated_texts) 

    gen_ids_pad = []
    for gen_id in generated_ids:
        gen_ids_pad.append(gen_id + [valid_set.pad_id] * (valid_set.max_length - len(gen_id))) # Добьем вектора индексов до максимальной длины
    gen_ids_pad = np.array(gen_ids_pad)
    corpus_ids = np.vstack((valid_set_ids, gen_ids_pad)) # Объединяем в numpy массив

    print("Снижение размерности с помощью TSNE...")
    corpus_tsne = TSNE(n_components=tsne_components,
                       learning_rate='auto', init='random',
                       perplexity=tsne_perplexity).fit_transform(corpus_ids)
    
    n_texts = len(valid_set)

    # Отделяем валидационные примеры и сгенерированные
    val_tsne = corpus_tsne[:n_texts]
    generated_tsne = corpus_tsne[n_texts:]

    print("Готово!")

    return val_tsne, generated_tsne        


def plot_texts(val_2d: np.ndarray, generated_2d: np.ndarray, title: str,
               val_text_label: str = "Валидационные тексты",
               gen_text_label: str = "Сгенерированные тексты") -> None:
    """Plots texts represented as 2D vectors as scatter plots

    Args:
        val_2d (np.ndarray): 2D validation texts' vectors
        generated_2d (np.ndarray): 2D validation texts' vectors
        title (str): Plot title
        val_text_label (str, optional): Label for validation texts representation. Defaults to "Валидационные тексты".
        gen_text_label (str, optional): Label for generated texts representation. Defaults to "Сгенерированные тексты".
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(val_2d[:, 0], val_2d[:, 1], label=val_text_label, color="r")
    plt.scatter(generated_2d[:, 0], generated_2d[:, 1], label=gen_text_label, color="b")
    plt.legend(loc="best")
    plt.title(title)
    plt.show()