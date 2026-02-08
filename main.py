import pandas as pd
from sklearn.datasets import fetch_california_housing
from datapreprocessor import Datapreprocessor
import numpy as np

hous=fetch_california_housing()
df=pd.DataFrame(hous.data, columns=hous.feature_names)
df['target']=hous.target

df['OceanProximity']=pd.Series(['NEAR BAY', 'INLAND', 'NEAR OCEAN', '<1H OCEAN', 'ISLAND']*(len(df)//5+1))[:len(df)]
df.loc[np.random.choice(df.index, 1000, replace=False), "MedInc"]=np.nan
df.loc[np.random.choice(df.index, 5000, replace=False), "HouseAge"]=np.nan

print("Оригинальный DataFrame, первые 5 строк")
print(df.head())
print("\nФорма: ", df.shape)
print("\nПропуски: \n", df.isnull().sum())

prepr=Datapreprocessor(df)
transformed_df=prepr.fit_transform()

print("Обработанный DataFrame, первые 5 строк")
print(transformed_df.head())
print("\nФорма: ", transformed_df.shape)
print("\nУдаленные столбцы: ", prepr.remove_columns)
print("One-hot столбцы: ", prepr.one_hot_columns)
print("Нормализированные числовые столбцы: ", prepr.numeric_columns)

