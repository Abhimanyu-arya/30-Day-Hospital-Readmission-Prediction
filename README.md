# 30-Day-Hospital-Readmission-Prediction
ML course project 

the data can be downloaded from OPENML ID 4541 DIABETES130US file
Downloaded the data from OPENML website in arff format
used a BASE path for consistent imports and saves
split it into data and meta, decoded it through utf-8
saved the schema from meta values
replaced all the missing data with NA for consistency ( Note None is read as Nan and handled with fillna('None')
checked missing column and row values before and after cleaning

column            missing %     decision                        reason
weight	          96.86%	Drop column	                Too sparse to be usable
medical_specialty 49.08%	kept as missing 	        Missingness might be informative, could be used for ablation experiment
payer_code	  39.56%	kept as missing         	Missingness might be informative, could be used for ablation experiment
race	          2.23%	        assign "Unknown"	        Preserves rows for fairness analysis
diag_1	          0.02%	        Drop rows	                Negligible row loss
diag_2	          0.35%	        assign None                     missing 2nd diagnosis is useful
diag_3	          1.40%	        assign None                     missing 3rd diagnosis is useful

checked mismatches from the given meta and the  updated values like "Unknown" added during cleaning, for robustness 

readmitted had 3 values previously, <30. >30 and NO, converted into 0 or 1. 1 representing <30 - patient was admitted under 30 days, and 0 represented NO and >30 - patient was either not admitted again or was admitted after 30 days.

hospice and expired patients which were represented by discharge disposition ids - 11, 13, 14, 19, 20, 21 - were removed from the dataset as it wasn't within the scope of this model to predict that outcome and keeping it would corrupt
the model. It dropped ~2% of the rows(post cleaning)

for drug columns (numbered 23 to 45(0-indexed) in the given dataset) a nzv ( near zero variance) report was prepared to check which columns should be dropped. columns with the top percentage value exceeding 98.46% threshold were flagged as weak.
for the nzv report, i looked at the readmission rate while choosing the threshold, it doesn't use the label to build the feature so the leakage should be minimum.
out of the 23 drug columns, 7 had real variations, 2 were completely constant, the other 14 columns if kept as is were weak, so all the drug columns, including the weak and the strong, were aggregated to 2 strong columns,
number of medication changes, and number of medications prescribed, and the weak ones were dropped, and the 2 aggregated columns along with the 7 strong columns are kept.

patient_nbr, encounter_id and readmit_30d are kept as is till train-test-validation split, it wont be passed into the model.

age had values like [10-20), it was encoded into the ranges' mean value like 15.

next i encoded icd-9(international classification of diseases, ninth revision), it had 3 types of values, first a number, second V+number, representing special cases, third E+number, represengint accident visits.
icd-9 codes with none or NA was assigned Missing, codes starting with V - assigned Other, E - assigned Injury. the other regular codes were grouped into major categories - namely Diabetes, Circulatory, Respiratory, Digestive, Injury, Musculoskeletal, Genitourinary and Neoplasms, rest as Other.
Some clinical categories like mental health and blood disorders are big enough to have their own categories but were grouped into the Other column, to limit the category count, and can be used for ablation in the future.
Also, there are hundreds of columns in icd-9, i have decided to aggregate them into 9 categories after looking at some publications.

next i encoded the medical specialty column. after observing the highest frequency columns, i decided to go with the top 12 columns, out of a total 73 unique columns, which covered 93.47% of the dataset, rest were grouped into 'Other'. therefore we had 11 real columns and 1 'Missing' column.

next, through 
remaining_categorical = df.select_dtypes(include = 'object').columns.tolist()
i checked all the columns with data type object still left, and encoded them one by one
gender converted to 0 (female) and 1(male), unknown gender had 3 cases which i decided to drop
change - 0 - no change in medication, 1 - change was made in the medication
diabetesMed - 1 - yes - 0 - no - whether a diabetes medication was used
max_glu_tested column - 0 - not tested - 1 - tested.
max glucose serum(test result) mapping: None-0 (not tested), Norm-0 (normal), >200- 1 (>200mg/dL), >300- 2(>300mg/dL)
a1c-tested column - 0 - not tested - 1 - tested.
A1Cresult - None - 0 (not performed), Norm - 0 ( normal), >7 - 1(result above 7%), >8 - 2 (result above 8%)
A1Cresult is the result of Hb1Ac test which tells the control of glucose over last 2-3 months. A1C is the test this entire datasets source was built around - stack et al.'s study was about measuring whether A1C affects readmission rate.

strong drug columns like metformin, nominal columns like race and id columns like admission type id were hot encoded. normal 0-1-2-3 mapping wasnt used because model would read it as 3>2>1>0, which fits in the earlier cases of glucose and A1C, where the levels are ordered, here we have expanded the columns into- for example Insulin: No, Steady, Up, Down -  from a single column of insulin, now taking 0 or 1 values.
when we create 4 columns, only one of it takes the value 1, all of the other get 0, so this gives us an exact linear relationship, so we drop the 1st alphabetic column, so every column reads 'versus the dropped column' which avoids the issue of perfect multicollinearity in logistic regression using newtons method,(our design matrix would become singular, which breaks the normal regression solution and the newtons method) which, although can be avoided using regularization, can be avoided by just dropping a column.


discharge disposition ids, which were nominal codes earlier were grouped into 7 groups by clinical meaning through id mappings, the groups are: Home (code 1), SNF(skilled nursing facility)/long-term care (3, 4, 23, 24), Home health (6, 8), Unknown (18, 25), Rehab (22), Left against medical advice (7), and Other institution, which collects all remaining small codes. Left-against-medical-advice is kept separate despite its size because it describes patient behaviour, not a facility. The groups were then hot encoded, and drop first = True was not used (which would have dropped AMA alphabetically) , instead ,The home column was dropped manually - Home is used as a reference, and the 6 columns are read as "versus being discharged home". Home was also chosen as the reference because ridge regularization shrinks each dummy variable toward the reference group, so the shrinkage pulls toward a well estimated baseline.
in the strong drug columns also, the columns with No value was used as a reference instead of using drop_first = true
I havent grouped by outcome because it would be using the label to design feature, which may lead to some leakage. 

boolean columns were converted into 0 or 1 columns. 0 - False, 1 - True
at the end of the script, the shape, number of na values and dtype.value_counts() were verified and the final dataset was saved before the train test validation split.
The final dataset is 99320 rows and 128 model features.


In the split script, train test split was implemented from scratch. the train test val ratio was taken as 0.7 0.2 & 0.1, and numpy rng was implemented on unique patient numbers. successful split was verified, no overlapping, covers all patients. 
seed = 42 for reproducibility
within the train_mask, normalization was done on the 13 continous columns.
gender, age, and race columns were saved seperately to verify equal performance across all groups.
the positive rate for each group turned out to be - 11.35%, 11.40%, 11.63%.
X - train, test, val was saved in csv format
y - train, test val was saved in csv format
fair - train, test, val was saved in csv format

csv was used because the headers are easily accessible and doesnt have dependencies, also the inspection is easy.



