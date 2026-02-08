import pandas as pd

class Datapreprocessor:
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Входные данные должны быть формата pandas DataFrame")
        self.df=df.copy()
        self.remove_columns=[]
        self.one_hot_columns={}
        self.numeric_columns=[]

    def remove_missing(self, threshold:float=0.5):
        if not (0<=threshold<=1):
            raise ValueError("Threshold должен быть от 0 до 1")
        missing_frac=self.df.isnull().mean()
        to_remove=missing_frac[missing_frac>threshold].index.tolist()
        self.remove_columns.extend(to_remove)
        self.df=self.df.drop(columns=to_remove)

        for col in self.df.columns:
            if self.df[col].isnull().sum()>0:
                if self.df[col].dtype=='object':
                    mode=self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(mode)
                else:
                    if abs(self.df[col].skew())>1:
                        median=self.df[col].median()
                        self.df[col] = self.df[col].fillna(median)
                    else:
                        mean=self.df[col].mean()
                        self.df[col] = self.df[col].fillna(mean)

    def encode_categorical(self):
        cat_cols=self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            dum=pd.get_dummies(self.df[col], prefix=col)
            self.one_hot_columns[col]=dum.columns.tolist()
            self.df=pd.concat([self.df, dum], axis=1)
            self.df=self.df.drop(col, axis=1)
    
    def normalize_numeric(self, method:str='minmax'):
        if method != "minmax" and method !="std":
            raise ValueError("Method должен быть minmax или std")
        self.numeric_columns=self.df.select_dtypes(include=['number']).columns.tolist()
        for col in self.numeric_columns:
            if method=="minmax":
                min_val=self.df[col].min()
                max_val=self.df[col].max()
                if max_val-min_val!=0:
                    self.df[col]=(self.df[col]-min_val)/(max_val-min_val)
            elif method=="std":
                mean=self.df[col].mean()
                std=self.df[col].std()
                if std!=0:
                    self.df[col]=(self.df[col]-mean)/std
    
    def fit_transform(self, threshold: float=0.5, method: str='minmax'):
        self.remove_missing(threshold)
        self.encode_categorical()
        self.normalize_numeric(method)
        return self.df
