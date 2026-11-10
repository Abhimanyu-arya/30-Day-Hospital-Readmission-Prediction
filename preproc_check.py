import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r'E:\ML') / 'Data'
df = pd.read_csv(BASE / 'Final_Data.csv')
ps = pd.read_csv(BASE / 'patient_splits.csv')
cont = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
        'num_med_changes', 'num_med_prescribed', 'age_numeric', 'max_glu_serum', 'A1Cresult']

masks = {s: df['patient_nbr'].isin(set(ps.loc[ps['split'] == s, 'patient_nbr'])).to_numpy()
         for s in ['train', 'val', 'test']}
mu = df.loc[masks['train'], cont].mean()
sd = df.loc[masks['train'], cont].std(ddof=0).replace(0, 1)
mu_all = df[cont].mean(); sd_all = df[cont].std(ddof=0)

for s, m in masks.items():
    X = pd.read_csv(BASE / f'X_{s}.csv')
    assert len(X) == m.sum()
    rebuilt = (df.loc[m, cont] - mu) / sd                        # TRAIN statistics only
    err = np.abs(rebuilt.to_numpy() - X[cont].to_numpy()).max()
    leaky = (df.loc[m, cont] - mu_all) / sd_all                  # what using ALL rows would have given
    gap = np.abs(leaky.to_numpy() - X[cont].to_numpy()).max()
    other = [c for c in X.columns if c not in cont]
    untouched = np.abs(df.loc[m, other].to_numpy(float) - X[other].to_numpy(float)).max()
    print(f"{s:5s}: rebuilt-from-train max error {err:.1e} | all-rows version would differ by up to {gap:.3f} SD "
          f"| mean of standardized columns {X[cont].mean().abs().max():.4f} | dummy columns untouched: {untouched == 0}")