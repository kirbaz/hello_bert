from dataclasses import dataclass
from typing import List, Generator, Tuple
import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizer
import torch


def attention_mask(batch: List[List[int]]) -> List[List[int]]:
    attention_matrix = []
    for weights in batch:
        attention_matrix.append([1 if x != 0 else 0 for x in weights])
    return attention_matrix

def review_embedding(tokens: List[List[List[int]]], model) -> List[List[float]]:
    """Return embeddings for batch of tokenized texts"""
    mask = attention_mask(tokens)
    tokens = torch.tensor(tokens)
    mask = attention_mask(tokens)

    with torch.no_grad():
        embedding = model(tokens, attention_mask=mask)

    feautures = embedding[0][:, 0, :].tolist()
    return feautures

def evaluate(model, embeddings, labels, cv=5) -> List[float]:
    # импорт нужных методов
    from sklearn.metrics import log_loss
    from sklearn.model_selection import KFold
    # Создание объекта KFold с 5 фолдами
    kfold = KFold(n_splits=cv)
    # Создание списка для хранения значений Cross-Entropy Loss для каждого фолда
    scores = []
    # Итерация по фолдам
    for train_index, test_index in kfold.split(embeddings):
        # Разделение данных на обучающую и тестовую выборки для текущего фолда
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Обучение модели на обучающей выборке текущего фолда
        model.fit(X_train, y_train)

        # Предсказание вероятностей классов на тестовой выборке текущего фолда
        predicted_probs = model.predict_proba(X_test)

        # Вычисление Cross-Entropy Loss для текущего фолда
        loss = log_loss(y_test, predicted_probs)

        # Добавление значения Cross-Entropy Loss в список
        scores.append(loss)

    return scores

@dataclass
class DataLoader:
    path: str
    tokenizer: PreTrainedTokenizer
    batch_size: int = 512
    max_length: int = 128
    dataset = []
    num_batches: int = 0
    padding: str = None

    def __post_init__(self):
        self.dataset = []
        with open(self.path, 'r', encoding='utf-8') as file:
            for line in file:
                self.dataset.append(line.strip().split(',', 4))
        self.dataset = self.dataset[1:]

    def __iter__(self) -> Generator[List[List[int]], None, None]:
        """Iterate over batches"""
        for i in range(len(self)):
            yield self.batch_tokenized(i)

    def __len__(self):
        """Number of batches"""
        df = pd.read_csv(self.path)
        num_batches = len(df) // self.batch_size
        if (len(df) % self.batch_size) != 0:
            num_batches += 1
        return num_batches

    def tokenize(self, batch: List[str]) -> List[List[int]]:
        """Tokenize list of texts"""
        encode_batch = [
            self.tokenizer.encode_plus(text, add_special_tokens=True,
                                       max_length=self.max_length, padding=True,
                                       truncation=True)['input_ids']
            for text in batch]
        return encode_batch

    def batch_loaded(self, i: int) -> Tuple[List[str], List[int]]:
        """Return loaded i-th batch of data (text, label)"""
        start_index = i * self.batch_size
        end_index = (i + 1) * self.batch_size
        batch = self.dataset[start_index:end_index]
        texts = [row[4] for row in batch]
        sentiment = [row[3] for row in batch]
        labels = [-1 if x == 'negative' else 0 if x == 'neutral' else 1 for x in sentiment]
        return texts, labels

    def batch_tokenized(self, i: int) -> Tuple[List[List[int]], List[int]]:
        """Return tokenized i-th batch of data"""
        texts, labels = self.batch_loaded(i)
        tokens = self.tokenize(texts)
        return tokens, labels
