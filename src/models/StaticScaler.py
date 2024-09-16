import numpy as np
import pandas as pd


class StaticMinMaxScaler(object):
    def __init__(self, columns, maximum, minimum):
        self.columns = columns
        self.min = np.array(minimum)
        self.max = np.array(maximum)

    def fit(self, df: pd.DataFrame):
        self.min = df.loc[:, self.columns].min(axis=0)
        self.max = df.loc[:, self.columns].max(axis=0)
        return self

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        new_df.loc[:, self.columns] = (new_df.loc[:, self.columns] - self.min) / (
            self.max - self.min
        )
        return new_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.__call__(df)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.columns = df.columns
        self = self.fit(df)
        return self(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        new_df.loc[:, self.columns] = (
            new_df.loc[:, self.columns] * (self.max - self.min)
        ) + self.min
        return new_df
