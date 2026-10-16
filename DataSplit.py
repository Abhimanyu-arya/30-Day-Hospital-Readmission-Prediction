import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r'E:\ML') / 'Data'
seed = 42
frac = (0.7, 0.2, 0.1)
# train test val

RAW = BASE / 'Final_Data.csv'
df = pd.read_csv(RAW)
print(df.shape)
print(df.isna().sum().sum())

rng = np.random.default_rng(seed)
patients = df['patient_nbr'].unique()
perm = rng.permutation(patients)
n = len(perm)

n_train = int(frac[0]*n)
n_test = int(frac[1]*n)

train_p = perm[:n_train]
test_p = perm[n_train: n_train + n_test]
val_p = perm[n_train + n_test:]

train_mask = df['patient_nbr'].isin(train_p).to_numpy()
#numpy boolean array
test_mask = df['patient_nbr'].isin(test_p).to_numpy()
val_mask = df['patient_nbr'].isin(val_p).to_numpy()

#print(set(train_p) & set(val_p))
#print(set(train_p) & set(test_p))
#print(set(test_p) & set(val_p))

#print(np.sum(train_mask))
#print(np.sum(test_mask == True))
#print(np.sum(val_mask == True))

assert (train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int) == 1).all()

for name, mask, pts in [('train', train_mask, train_p), ('test', test_mask, test_p), ('val', val_mask, val_p)]:
    print(f"{name} rows={mask.sum()} ({mask.mean():.1%}), patients={len(pts):6d}  positive rate={df.loc[mask, 'readmit_30d'].mean():.4f}")


y = df['readmit_30d'].to_numpy()
race_cols = [c for c in df.columns if c.startswith('race_')]
fair = df[['gender', 'age_numeric'] + race_cols].copy()

X = df.drop(columns = ['patient_nbr', 'encounter_id', 'readmit_30d'])

cont_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
             'num_medications', 'number_outpatient', 'number_emergency',
             'number_inpatient', 'number_diagnoses', 'num_med_changes',
             'num_med_prescribed', 'age_numeric', 'max_glu_serum', 'A1Cresult']

missing_cols = [c for c in cont_cols if c not in X.columns]
#print(missing_cols)

mu = X.loc[train_mask, cont_cols].mean()
std = X.loc[train_mask, cont_cols].std(ddof=0).replace(0, 1)
X[cont_cols] = (X[cont_cols]-mu)/std 

feature_names = X.columns.tolist()

X[train_mask].to_csv(BASE / 'X_train.csv', index = False)
X[test_mask].to_csv(BASE / 'X_test.csv', index = False)
X[val_mask].to_csv(BASE / 'X_val.csv', index = False)

pd.Series(y[train_mask], name = 'readmit_30d').to_csv(BASE / 'y_train.csv', index = False)
pd.Series(y[test_mask], name = 'readmit_30d').to_csv(BASE / 'y_test.csv', index = False)
pd.Series(y[val_mask], name = 'readmit_30d').to_csv(BASE / 'y_val.csv', index = False)

fair[train_mask].to_csv(BASE / 'fair_train.csv', index = False)
fair[test_mask].to_csv(BASE / 'fair_test.csv', index = False)
fair[val_mask].to_csv(BASE / 'fair_val.csv', index = False)
print(X.shape)

dummy_cols = [c for c in X[train_mask].columns if c not in cont_cols]
print(X[train_mask][dummy_cols].sum().sort_values().head(15))