#!/usr/bin/python3
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузка датасета из папки "raw"
dataset_path = 'data/raw_tt/ds_salaries.csv'
df = pd.read_csv(dataset_path)

# Разделение на train и test
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Проверка и создание папки "raw_tt"
os.makedirs('data/raw', exist_ok=True)

# Сохранение train и test в файлы
train.to_csv('data/raw/train.csv', index=False)
test.to_csv('data/raw/test.csv', index=False)
