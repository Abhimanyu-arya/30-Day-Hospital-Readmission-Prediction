import pandas as pd 
import numpy as np 
from pathlib import Path

BASE = Path(r'E:\ML') / 'Data'
X_train_df = pd.read_csv(BASE / 'X_train.csv')
X_test_df = pd.read_csv(BASE / 'X_test.csv')
X_val_df = pd.read_csv(BASE / 'X_val.csv')

X_train = X_train_df.to_numpy(dtype = float)
X_test = X_test_df.to_numpy(dtype = float)
X_val = X_val_df.to_numpy(dtype = float)

y_train = pd.read_csv(BASE / 'y_train.csv')['readmit_30d'].to_numpy()
y_test = pd.read_csv(BASE / 'y_test.csv')['readmit_30d'].to_numpy()
y_val = pd.read_csv(BASE / 'y_val.csv')['readmit_30d'].to_numpy()

fair_train = pd.read_csv(BASE / 'fair_train.csv')
fair_test = pd.read_csv(BASE / 'fair_test.csv')
fair_val = pd.read_csv(BASE / 'fair_val.csv')

feature_names = X_train_df.columns.to_list()


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(X_train_df.describe())
