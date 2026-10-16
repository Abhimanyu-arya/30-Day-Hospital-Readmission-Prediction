import pandas as pd 
import numpy as np 
from scipy.io import arff
from pathlib import Path

BASE = Path(r'E:\ML') / 'Data'
RAW = BASE / 'Raw_OPENML4541.arff'
data, meta = arff.loadarff(RAW) 

df = pd.DataFrame(data) 
str_col = df.select_dtypes([object]).columns 
for col in str_col :
    df[col] = df[col].str.decode('utf-8')

#print(df.head())
#print(df.shape) 
#print(df.info())
 
 
schema = pd.DataFrame([
    {'attribute': name,
    'type': meta[name][0],
    'allowed_values': meta[name][1]} for name in meta.names()
])
#print(schema)
#schema.to_csv('Data/schema_reference.csv', index = False)

df = df.replace('?', pd.NA)


missing_count = df.isna().sum()
#print(missing_count) 

missing_pct = df.isna().mean().sort_values(ascending = False)*100
#print(missing_pct)



df = df.drop(columns=['weight'])
df['medical_specialty'] = df['medical_specialty'].fillna('Missing')
df['payer_code'] = df['payer_code'].fillna('Missing')
df['race'] = df['race'].fillna('Unknown')
df = df.dropna(subset=['diag_1'])
df['diag_2'] = df['diag_2'].fillna('None')
df['diag_3'] = df['diag_3'].fillna('None')


missing_count = df.isna().sum() 
#print(missing_count) 

missing_pct = df.isna().mean().sort_values(ascending = False)*100
#print(missing_pct)
#print(df.shape)

rows_missing = df.isna().sum(axis = 1)
#print(rows_missing.value_counts().sort_index())



duplicate_count = df['patient_nbr'].duplicated().sum()
#print(duplicate_count)

for col in meta.names():
    if col not in df.columns:
        continue
    attr_types, attr_range = meta[col] 
    if attr_types == 'nominal':
        actual_values = set(df[col].dropna().unique()) 
        expected_values = set(attr_range)
        mismatches = actual_values - expected_values
        #if mismatches:
            #print(f"{col} Mismatches: {mismatches}")

known_mismatches = {
    'race': {'Unknown'},
    'payer_code': {'Missing'},
    'medical_specialty': {'Missing'},
    'diag_2': {'None'},
    'diag_3': {'None'},
}

for col in meta.names():
    if col not in df.columns:
        continue
    attr_type, attr_range = meta[col]
    if attr_type == 'nominal':
        actual_values = set(df[col].dropna().unique())
        expected_values = set(attr_range) | known_mismatches.get(col, set())
        mismatches = actual_values - expected_values
        #if mismatches:
            #print(f"{col} Mismatches: {mismatches}")

df.to_csv(BASE / 'cleaned_data.csv', index = False)