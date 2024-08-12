import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

def split_data(articles: pd.DataFrame, user_interactions: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    merged_data = user_interactions.merge(articles, on='article_id')
    train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=42)
    X_train = train_data.drop(columns=['user_id', 'interaction_date'])
    y_train = train_data['interaction'].values
    X_test = test_data.drop(columns=['user_id', 'interaction_date'])
    y_test = test_data['interaction'].values
    return X_train, y_train, X_test, y_test
