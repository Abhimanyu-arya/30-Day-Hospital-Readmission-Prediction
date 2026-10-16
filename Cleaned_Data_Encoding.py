import pandas as pd
import numpy as np 
from pathlib import Path

BASE = Path(r'E:\ML') / 'Data'
RAW = BASE / 'cleaned_data.csv'
df = pd.read_csv(RAW)

df['readmit_30d'] = (df['readmitted'] == '<30').astype(int)
df = df.drop(columns = 'readmitted')

#print(df['readmit_30d'].value_counts())

#print(df['discharge_disposition_id'].value_counts().sort_index())

expired_hospice_ids = [11, 13, 14, 19, 20, 21] 
before = len(df)

df = df[~df['discharge_disposition_id'].isin(expired_hospice_ids)]

after = len(df)

#print(f"Dropped {before-after} rows ({((before-after)/before)*100:.2f}%)")

#print(df['discharge_disposition_id'].dtype)

#print(df.iloc[0, 23:45])
drug_cols = df.columns[23:46].tolist() 

#print(drug_cols)

#for col in drug_cols:
    #print(df[col].value_counts(normalize = True).round(4)*100)

nzv_report = []
for col in drug_cols:
    vc = df[col].value_counts()
    vc_pct = df[col].value_counts(normalize = True)
    nzv_report.append({
        'column': col,
        'n_unique': df[col].nunique(),
        'top_value': vc_pct.index[0],
        'top_pct': round(vc_pct.iloc[0]*100, 3),
        'rarest_cnt': vc.iloc[-1],
    })

nzv_df = pd.DataFrame(nzv_report).sort_values('top_pct', ascending = False)
#print(nzv_df.to_string())
#print(nzv_df.shape)

threshold = 98.46
to_drop = nzv_df[nzv_df['top_pct'] >= threshold]['column'].tolist()

#print(to_drop)
#for col in to_drop:
    #print(df[col].value_counts(normalize = True).round(4)*100)

#for col in to_drop:
    #print(df.groupby(col, observed = True)['readmit_30d'].mean())

df['num_med_changes'] = (df[drug_cols].isin(['Up', 'Down'])).sum(axis=1)
df['num_med_prescribed'] = (df[drug_cols] != 'No').sum(axis=1)

df = df.drop(columns = to_drop)
#print(df.shape)


age_map = {
    '[0-10)': 5,
    '[10-20)': 15,
    '[20-30)': 25,
    '[30-40)': 35,
    '[40-50)': 45,
    '[50-60)': 55,
    '[60-70)': 65,
    '[70-80)': 75,
    '[80-90)': 85,
    '[90-100)': 95,
}

df['age_numeric'] = df['age'].map(age_map)
df = df.drop(columns  = 'age')

#print(df['age_numeric'].isna().sum())
#print(df.head())

def icd9_mapping(code):
    if(pd.isna(code)) or code == 'None':
        return 'Missing'
    code = str(code) 
    if code.startswith('V'):
        return 'Other'
    if code.startswith('E'):
        return 'Injury'
    
    try:
        code = float(code)
    except ValueError:
        return 'Other'
    if 250 <= code < 251:
        return 'Diabetes'
    elif 390 <= code <= 459 or code == 785:
        return 'Circulatory'
    elif 460 <= code <= 519 or code == 786:
        return 'Respiratory'
    elif 520 <= code <= 579 or code == 787:
        return 'Digestive'
    elif 800 <= code <= 999:
        return 'Injury'
    elif 710 <= code <= 739:
        return 'Musculoskeletal'
    elif 580 <= code <= 629 or code == 788:
        return 'Genitourinary'
    elif 140 <= code <= 239:
        return 'Neoplasms'
    else:
        return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[f'{col}_category'] = df[col].apply(icd9_mapping)
#for col in ['diag_1_category', 'diag_2_category', 'diag_3_category']:
    #print(df[col].value_counts())


other_codes = df.loc[df['diag_1_category'] == 'Other', 'diag_1']
#print(other_codes.value_counts().head(20))

df = df.drop(columns = ['diag_1', 'diag_2', 'diag_3'])

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#print(df['medical_specialty'].nunique())
print(df['medical_specialty'].value_counts())

top_12 = df['medical_specialty'].value_counts().nlargest(12).index.tolist()
#print(top_12)

top_12_coverage = df['medical_specialty'].isin(top_12).mean()*100
#print(top_12_coverage)
#93.47%

df['medical_specialty_grouped'] = df['medical_specialty'].where(df['medical_specialty'].isin(top_12), 'Other')
df = pd.get_dummies(df, columns = ['medical_specialty_grouped'], prefix = 'medical_specialty_grouped')
df = df.drop(columns = 'medical_specialty_grouped_Missing')

#print(df['medical_specialty_grouped'].value_counts())

df = df.drop(columns = ['medical_specialty']) 


remaining_categorical = df.select_dtypes(include = 'object').columns.tolist()
#print(remaining_categorical) 

#print(df['gender'].value_counts())

before_ = len(df)
df = df[df['gender'] != 'Unknown/Invalid']
after_ = len(df) 
#print(before_ - after_)

#print(df['gender'].value_counts())

df['gender'] = (df['gender'] == 'Male').astype(int)
#print(df['gender'].value_counts())

#print(df['change'].value_counts())

df['change'] = (df['change'] == 'Ch').astype(int)
#print(df['change'].value_counts())

#print(df['diabetesMed'].value_counts())
df['diabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)
#print(df['diabetesMed'].value_counts())

glucose_map = {'None': 0, 'Norm': 0, '>200': 1, '>300': 2}
df['max_glu_serum'] = df['max_glu_serum'].fillna('None')
df['glu_tested'] = (df['max_glu_serum'] != 'None').astype(int)
df['max_glu_serum'] = df['max_glu_serum'].map(glucose_map)

a1c_map = {'None': 0, 'Norm': 0, '>7': 1, '>8': 2}
df['A1Cresult'] = df['A1Cresult'].fillna('None')
df['a1c_tested'] = (df['A1Cresult'] != 'None').astype(int)
df['A1Cresult'] = df['A1Cresult'].map(a1c_map)
#print(df['A1Cresult'].value_counts())



strong_drug_cols = ['metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin']
df = pd.get_dummies(df, columns = strong_drug_cols, prefix = strong_drug_cols)
df = df.drop(columns = [f'{c}_No' for c in strong_drug_cols])


nominal_cols = ['race', 'diag_1_category', 'diag_2_category', 'diag_3_category']
df = pd.get_dummies(df, columns = nominal_cols, prefix = nominal_cols, drop_first = True)

freq = df['payer_code'].value_counts(normalize=True)
rare = freq[freq<0.01].index
df['payer_code'] = df['payer_code'].where(~df['payer_code'].isin(rare), 'OtherPayer')
#print(df['payer_code'].value_counts())
df = pd.get_dummies(df, columns = ['payer_code'], prefix = 'payer_code')
df = df.drop(columns = 'payer_code_MC')

#print(df['admission_type_id'].dtype)
#print(df['admission_source_id'].dtype)

id_cols = ['admission_type_id', 'admission_source_id']
#for c in id_cols:
    #print(df[c].value_counts())
src_map = {1: 'Referral', 2: 'Referral', 3: 'Referral',
           7: 'ER',
           4: 'Transfer', 5: 'Transfer', 6: 'Transfer', 10: 'Transfer',
           18: 'Transfer', 22: 'Transfer', 26: 'Transfer'}
type_map = {1: 'Emergency', 2: 'Urgent', 3: 'Elective'}

df['admission_source_group'] = (df['admission_source_id'].astype(int)
                                .map(src_map).fillna('Unknown'))
df['admission_type_group'] = (df['admission_type_id'].astype(int)
                              .map(type_map).fillna('Unknown'))
#print(df['admission_source_group'].value_counts())
#print(df['admission_type_group'].value_counts())
df = df.drop(columns=['admission_source_id', 'admission_type_id'])
df = pd.get_dummies(df, columns=['admission_source_group', 'admission_type_group'])
df = df.drop(columns=['admission_source_group_ER', 'admission_type_group_Emergency'])

#print(df['discharge_disposition_id'].value_counts())
dd_map = {
    1: 'Home',
    3: 'SNF_LTC', 4: 'SNF_LTC', 23: 'SNF_LTC', 24: 'SNF_LTC',
    6: 'HomeHealth', 8: 'HomeHealth',
    18: 'Unknown', 25: 'Unknown', 26: 'Unknown',
    22: 'Rehab',
    7: 'AMA',
}

df['discharge_group'] = (df['discharge_disposition_id'].astype(int).map(dd_map).fillna('OtherInstitution'))
print(df['discharge_group'].value_counts())
df = pd.get_dummies(df, columns = ['discharge_group'], prefix = 'discharge_group')
df = df.drop(columns = ['discharge_disposition_id', 'discharge_group_Home'])

#print(df.select_dtypes(include = 'object').columns.tolist()) 
bool_cols = df.select_dtypes(include = 'bool').columns
df[bool_cols] = df[bool_cols].astype(int)


print(df.shape)
print(df.isna().sum().sum())
print(df.dtypes.value_counts())
print([col for col in df.columns if col.startswith('insulin_')])
print([col for col in df.columns if col.startswith('race_')])
print('patient_nbr' in df.columns, 'encounter_id' in df.columns)


print([c for c in df.columns if c.startswith('discharge')])

df = df.reset_index(drop = True)

df.to_csv(BASE / 'Final_Data.csv', index = False)