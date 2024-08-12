from sklearn.base import BaseEstimator, TransformerMixin
from transformers import BertTokenizer, BertModel
from gensim.models import Word2Vec
import torch
import numpy as np
import pandas as pd
from typing import Tuple

class BERTTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature: str, batch_size: int = 32, max_length: int = 128):
        self.feature = feature
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        return self

    def transform(self, X: pd.DataFrame):
        embeddings = []
        for i in range(0, len(X), self.batch_size):
            batch_texts = X[self.feature].iloc[i:i+self.batch_size].tolist()
            inputs = self.tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length)
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature: str, vector_size: int = 100, window: int = 5, min_count: int = 1):
        self.feature = feature
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> 'Word2VecTransformer':
        sentences = [text.split() for text in X[self.feature]]
        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        embeddings = []
        for text in X[self.feature]:
            words = text.split()
            word_vecs = [self.model.wv[word] for word in words if word in self.model.wv]
            if word_vecs:
                embeddings.append(np.mean(word_vecs, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings)


class TFIDFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature: str, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.feature = feature
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)

    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> 'TFIDFTransformer':
        self.vectorizer.fit(X[self.feature])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.vectorizer.transform(X[self.feature]).toarray()
