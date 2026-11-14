"""Recomputes the data-dependent rules of Cleaned_Data_Encoding.py on TRAINING patients only and
compares them with the versions that used all rows. Read-only: writes nothing.

Rules checked (same row filters, in the same order, as Cleaned_Data_Encoding.py):
  1. drug columns dropped by the 98.46% dominant-value threshold
  2. the top-12 medical specialties (and the specialty reference level)
  3. payer codes merged into OtherPayer (share below 1%)
"""
import pandas as pd
from pathlib import Path

BASE = Path(r'E:\ML') / 'Data'
df = pd.read_csv(BASE / 'cleaned_data.csv')
ps = pd.read_csv(BASE / 'patient_splits.csv')
train_ids = set(ps.loc[ps['split'] == 'train', 'patient_nbr'])
df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]      # same filter as the encoding script
df = df.assign(is_train=df['patient_nbr'].isin(train_ids).to_numpy())
tr = df[df['is_train']]
print(f"rows after the discharge filter: all {len(df)}, training patients {len(tr)}")

# ---- 1. drug columns, 98.46% rule ----
T = 98.46
drug_cols = df.columns[23:46].tolist()                                          # same positional slice
top = lambda frame, c: 100 * frame[c].value_counts(normalize=True).iloc[0]
t = pd.DataFrame({'all_rows': {c: top(df, c) for c in drug_cols}, 'train_only': {c: top(tr, c) for c in drug_cols}})
t['dropped_all'] = t['all_rows'] >= T
t['dropped_train'] = t['train_only'] >= T
print(f"\n1. dominant-value share per drug column (dropped if >= {T}%)")
print(t.sort_values('all_rows').round(3).to_string())
flips = t[t['dropped_all'] != t['dropped_train']].index.tolist()
print("decision differs for:", flips if flips else "none (same columns kept and dropped)")
kept = t[~t['dropped_all']]
print(f"kept columns: {len(kept)}; largest share among kept {kept['all_rows'].max():.3f}%; "
      f"smallest share among dropped {t[t['dropped_all']]['all_rows'].min():.3f}%")

# ---- 2. top-12 specialties ----
rank = lambda frame: frame['medical_specialty'].value_counts()
r_all, r_tr = rank(df), rank(tr)
s_all, s_tr = set(r_all.nlargest(12).index), set(r_tr.nlargest(12).index)
print("\n2. top-12 specialties")
print("same set:", s_all == s_tr, "| only in all-rows list:", sorted(s_all - s_tr), "| only in train list:", sorted(s_tr - s_all))
print("ranks 10 to 14, all rows:  ", [(k, int(v)) for k, v in r_all.iloc[9:14].items()])
print("ranks 10 to 14, train only:", [(k, int(v)) for k, v in r_tr.iloc[9:14].items()])
def ref(frame, top12):
    g = frame['medical_specialty'].where(frame['medical_specialty'].isin(top12), 'Other')
    return g[~g.isin(['Missing', 'Other'])].value_counts().idxmax()
print("specialty reference level: all rows", ref(df, s_all), "| train only", ref(tr, s_tr))

# ---- 3. payer codes, 1% rule ----
dg = df[df['gender'] != 'Unknown/Invalid']                                      # payer rule runs after the gender filter
f_all = dg['payer_code'].value_counts(normalize=True)
f_tr = dg[dg['is_train']]['payer_code'].value_counts(normalize=True)
rare_all, rare_tr = set(f_all[f_all < 0.01].index), set(f_tr[f_tr < 0.01].index)
print("\n3. payer codes merged into OtherPayer (share < 1%)")
print("same set:", rare_all == rare_tr, "| merged on all rows:", sorted(rare_all), "| merged on train only:", sorted(rare_tr))
near = pd.DataFrame({'all_rows_%': 100 * f_all, 'train_only_%': 100 * f_tr}).dropna()
print("codes with a share between 0.5% and 2%:")
print(near[(near.max(axis=1) > 0.5) & (near.min(axis=1) < 2)].round(3).to_string())