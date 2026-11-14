"""Exploratory plots for the data-understanding section.
Descriptive only: uses the TRAINING patients' encounters (plus the positive rate of each split in the
class-balance figure). Writes PNGs to Data/figures and the numbers behind them to Data/data_summary.csv."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r'E:\ML') / 'Data'
FIG = BASE / 'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
C1, C2, GREY = '#2a5db0', '#c0504d', '#8a93a3'

df = pd.read_csv(BASE / 'Final_Data.csv')
ps = pd.read_csv(BASE / 'patient_splits.csv')
ps['patient_nbr'] = ps['patient_nbr'].astype('int64')
split = df['patient_nbr'].astype('int64').map(ps.set_index('patient_nbr')['split'])
assert split.notna().all()
tr = df[split == 'train']
overall = tr['readmit_30d'].mean()
print(f"training encounters: {len(tr)}, readmission rate {overall:.4f}")
rows = []


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate_table(fig, labels, masks, frame):
    out = []
    for lab, m in zip(labels, masks):
        sub = frame[m]
        k, n = int(sub['readmit_30d'].sum()), len(sub)
        lo, hi = wilson(k, n) if n else (np.nan, np.nan)
        out.append((lab, n, k, k / n if n else np.nan, lo, hi))
        rows.append({'figure': fig, 'group': lab, 'n': n, 'readmissions': k, 'rate': k / n if n else np.nan, 'lo': lo, 'hi': hi})
    return pd.DataFrame(out, columns=['group', 'n', 'k', 'rate', 'lo', 'hi'])


def rate_plot(ax, t, title, xlabel, rotate=0):
    x = np.arange(len(t))
    ax.bar(x, t['rate'] * 100, color=C1, alpha=.85)
    ax.errorbar(x, t['rate'] * 100, yerr=[(t['rate'] - t['lo']) * 100, (t['hi'] - t['rate']) * 100], fmt='none', ecolor='black', capsize=3, lw=1)
    ax.axhline(overall * 100, color=C2, ls='--', lw=1, label=f'all training encounters ({overall * 100:.1f}%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\nn={n:,}" for g, n in zip(t['group'], t['n'])], rotation=rotate, ha='right' if rotate else 'center', fontsize=8)
    ax.set_ylabel('30-day readmission rate (%)')
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8)


# 1. class balance (counts in train; rate in all three splits)
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
cnt = tr['readmit_30d'].value_counts().sort_index()
a.bar(['not readmitted\nwithin 30 days', 'readmitted\nwithin 30 days'], cnt.values, color=[GREY, C2])
for i, v in enumerate(cnt.values):
    a.text(i, v, f"{v:,}\n({100 * v / len(tr):.1f}%)", ha='center', va='bottom', fontsize=9)
a.set_ylim(0, cnt.max() * 1.18)
a.set_ylabel('encounters')
a.set_title(f'Training split: {len(tr):,} encounters', fontsize=10)
t = rate_table('class_balance', ['train', 'val', 'test'], [split == s for s in ['train', 'val', 'test']], df.assign())
rate_plot(b, t, 'Readmission rate by split (95% intervals)', 'split')
b.get_legend().remove() if b.get_legend() else None
fig.suptitle('Class balance: about 1 in 9 encounters is a 30-day readmission', fontsize=11)
fig.tight_layout()
fig.savefig(FIG / 'class_balance.png', dpi=160)
plt.close(fig)

# 2. missingness in the raw data (shares from the cleaning step)
miss = [('weight', 96.86, 'column dropped'), ('medical_specialty', 49.08, 'kept as "Missing"'), ('payer_code', 39.56, 'kept as "Missing"'),
        ('race', 2.23, 'assigned "Unknown"'), ('diag_3', 1.40, '"None", later "Missing"'), ('diag_2', 0.35, '"None", later "Missing"'),
        ('diag_1', 0.02, 'rows dropped')]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ys = np.arange(len(miss))[::-1]
ax.barh(ys, [m[1] for m in miss], color=C1, alpha=.85)
for y, m in zip(ys, miss):
    ax.text(m[1] + 1, y, f"{m[1]:.2f}%  ->  {m[2]}", va='center', fontsize=8.5)
ax.set_yticks(ys)
ax.set_yticklabels([m[0] for m in miss])
ax.set_xlim(0, 135)
ax.set_xlabel('share of encounters missing in the raw data (%)')
ax.set_title('Missing data and how each column was handled', fontsize=10)
fig.tight_layout()
fig.savefig(FIG / 'missingness.png', dpi=160)
plt.close(fig)
for m in miss:
    rows.append({'figure': 'missingness', 'group': m[0], 'n': np.nan, 'readmissions': np.nan, 'rate': m[1] / 100, 'lo': np.nan, 'hi': np.nan})

# 3. prior inpatient visits: rate (left) and distribution (right)
ni = tr['number_inpatient']
bins = [('0', ni == 0), ('1', ni == 1), ('2', ni == 2), ('3-4', ni.between(3, 4)), ('5-7', ni.between(5, 7)), ('8+', ni >= 8)]
t = rate_table('prior_inpatient', [b_[0] for b_ in bins], [b_[1] for b_ in bins], tr)
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios': [1.3, 1]})
rate_plot(a, t, 'Readmission rate by prior-year inpatient visits', 'inpatient visits in the year before the encounter')
vc = ni.value_counts().sort_index()
b.bar(vc.index, vc.values, color=GREY)
b.set_yscale('log')
b.set_xlabel('inpatient visits in the year before the encounter')
b.set_ylabel('encounters (log scale)')
b.set_title(f'Heavy tail: maximum {int(ni.max())} visits', fontsize=10)
fig.tight_layout()
fig.savefig(FIG / 'rate_prior_inpatient.png', dpi=160)
plt.close(fig)

# 4. age (bin midpoints, 10-year bins)
mids = sorted(tr['age_numeric'].unique())
t = rate_table('age', [f"{int(m - 5)}-{int(m + 5)}" for m in mids], [tr['age_numeric'] == m for m in mids], tr)
fig, ax = plt.subplots(figsize=(8, 3.8))
rate_plot(ax, t, 'Readmission rate by age group', 'age group (years)')
fig.tight_layout()
fig.savefig(FIG / 'rate_age.png', dpi=160)
plt.close(fig)

# 5. discharge destination (Home is the reference: all discharge dummies are 0)
dis = [c for c in tr.columns if c.startswith('discharge_group_')]
lab = pd.Series('Home', index=tr.index)
for c in dis:
    lab[tr[c] == 1] = c.replace('discharge_group_', '')
order = lab.value_counts().index.tolist()
t = rate_table('discharge', order, [lab == g for g in order], tr)
fig, ax = plt.subplots(figsize=(8, 3.9))
rate_plot(ax, t, 'Readmission rate by discharge destination', 'discharge group (Home is the reference group)')
fig.tight_layout()
fig.savefig(FIG / 'rate_discharge.png', dpi=160)
plt.close(fig)

out = pd.DataFrame(rows)
out.to_csv(BASE / 'data_summary.csv', index=False)
print(out[out.figure != 'missingness'].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("saved:", ', '.join(sorted(p.name for p in FIG.glob('*.png'))))