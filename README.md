# From-Scratch Logistic Regression for Diabetic Readmission Prediction

Newton's method, calibration, cost-sensitive thresholds and fairness analysis on the Diabetes 130-US Hospitals dataset (OpenML ID 4541, 1999 to 2008). ES 335 project, due Nov 17, 2026.

---

## 0. Question and scope

- **Question.** Can a logistic regression written from scratch, using information available when a diabetic patient is discharged, flag encounters that end in a readmission within 30 days, and is flagging worth it once the cost of a missed readmission is weighed against the cost of an unnecessary follow-up?
- **One row** is one hospital encounter. **Target:** `readmit_30d` (1 if readmitted within 30 days, 0 otherwise). Positive rate about 11.4%.
- **Prediction time:** at discharge. The features are admission and discharge information (admission type and source, discharge disposition), length of stay, counts of procedures, lab tests and medications, diagnosis categories, medication changes, glucose and A1C results, and the patient's number of outpatient, emergency and inpatient visits in the year before the encounter (per the dataset documentation). **[TODO: confirm the "year before" wording against the UCI/OpenML description.]**
- **Intended use (assumed):** flag patients for a follow-up intervention. The cost ratio `r` (cost of a missed readmission divided by the cost of an unnecessary intervention) is **assumed, not estimated**, because there is no cost data. Results are reported over a range of `r`.
- **Limits of the claim.** The results describe association and prediction on 1999 to 2008 US hospital data, not causes, and not performance on current patients (section 16).

---

## 1. Data source

- OpenML ID 4541, Diabetes 130-US hospitals, 1999 to 2008. Downloaded from OpenML in ARFF format.
- Original study: Strack et al. (2014), which asked whether HbA1c measurement affects readmission.
- License: **[TODO: verify license on the OpenML / UCI page and state it here]**.
- After cleaning: 99,320 encounters from 69,979 unique patients. Because patients have multiple encounters (1.42 per patient in the training split; 23.2% of training patients have two or more), a patient-level split is required (see section 4).
- Only the model-ready CSVs are used downstream. `Data/` is in `.gitignore`, and no patient-level rows go in the repository.

## 2. Cleaning (`Data_Loading_Processing.py`)

The ARFF file is split into data and meta, strings are decoded as UTF-8, the schema is saved from the meta, and `?` is replaced with NA. Missing counts were checked before and after cleaning. After cleaning, every nominal column was compared against the allowed values in the ARFF meta. The only differences are the values added deliberately during cleaning (`Unknown`, `Missing`, `None`).

| Column | Missing % | Decision | Reason |
|---|---|---|---|
| weight | 96.86% | Drop column | Too sparse to use |
| medical_specialty | 49.08% | Kept as `Missing` | Missingness may be informative, tested in an ablation (section 10) |
| payer_code | 39.56% | Kept as `Missing` | Missingness may be informative, tested in an ablation (section 10) |
| race | 2.23% | Assigned `Unknown` | Keeps rows for the fairness analysis |
| diag_1 | 0.02% | Drop rows | Negligible row loss |
| diag_2 | 0.35% | Assigned `None`, later `Missing` | A missing second diagnosis is informative |
| diag_3 | 1.40% | Assigned `None`, later `Missing` | A missing third diagnosis is informative |

Note that `None` in the raw data is read by pandas as NaN, so it is handled explicitly with `fillna('None')`.

## 3. Feature engineering (`Cleaned_Data_Encoding.py`)

### Target
`readmitted` has three values (`<30`, `>30`, `NO`). It becomes `readmit_30d`: 1 for `<30`, 0 for `>30` and `NO`.

### Rows removed
- Patients who died or went to hospice (discharge disposition ids 11, 13, 14, 19, 20, 21) are removed, about 2% of rows after cleaning. Predicting that outcome is outside the scope of the model and keeping them would corrupt the target.
- 3 rows with gender `Unknown/Invalid` are dropped.

### Drug columns
There are 23 drug columns. A near-zero-variance report flagged columns whose most frequent value exceeds 98.46%. Of the 23, 7 have real variation, 2 are constant and 14 are weak. All 23 are aggregated into two columns, `num_med_changes` (count of Up or Down) and `num_med_prescribed` (count not equal to No). The weak columns are dropped, and the 7 strong ones are kept and one-hot encoded: metformin, glimepiride, glipizide, glyburide, pioglitazone, rosiglitazone, insulin.

Disclosure: the 98.46% threshold was chosen while looking at readmission rates per column, so it does take a peek at the label. The threshold itself is based on the share of the dominant value, so the leakage risk is considered minimal. A train-only recomputation of this threshold and of the other category groupings has **not** been done yet (section 15 says exactly what has and has not been verified).

### Other encodings
- **age**: bin midpoint (for example `[10-20)` becomes 15), named `age_numeric`. This assumes a linear effect on the log-odds.
- **gender**: 0 female, 1 male. **change**: 1 if a medication change was made. **diabetesMed**: 1 if a diabetes medication was used.
- **Glucose and A1C**: each has a separate tested flag (`glu_tested`, `a1c_tested`) plus an ordinal result. `max_glu_serum`: None 0, Norm 0, >200 1, >300 2. `A1Cresult`: None 0, Norm 0, >7 1, >8 2. Untested and normal share level 0 because the tested flag separates them. A1C is the test the source study was built around.
- **ICD-9 (diag_1, diag_2, diag_3)**: grouped into Diabetes (250.xx), Circulatory (390 to 459, 785), Respiratory (460 to 519, 786), Digestive (520 to 579, 787), Genitourinary (580 to 629, 788), Musculoskeletal (710 to 739), Neoplasms (140 to 239), Injury (800 to 999 and E-codes), Other (including V-codes, which are supplementary factors; E-codes are external causes of injury), and Missing. Codes 785 to 788 are symptom codes, mapped to the organ system they point to. The grouping follows Strack et al. Mental health and blood disorders are large enough for their own categories but are left in Other to limit the category count. Splitting them out was not tested (section 17).
- **medical_specialty**: top 12 values (covering 93.47% of rows, out of 73 distinct) kept, the rest grouped as `Other`. This gives 10 named specialties plus `Missing` and `Other`.
- **payer_code**: codes under 1% of rows are merged into `OtherPayer`. `Missing` is kept as its own column. (A code with a single row in the whole dataset, FR, is what exposed the need for this merge.)
- **admission_source_id**: Referral (1, 2, 3), ER (7), Transfer (4, 5, 6, 10, 18, 22, 26), Unknown (everything else, such as NULL, not available and not mapped codes like 9, 17 and 20). **[TODO: add group counts from your printout]**
- **admission_type_id**: Emergency (1), Urgent (2), Elective (3), Unknown (everything else). **[TODO: add group counts]**
- **discharge_disposition_id**: Home (1), SNF/LTC (3, 4, 23, 24), HomeHealth (6, 8), Unknown (18, 25, 26), Rehab (22), AMA (7), OtherInstitution (all remaining codes). AMA is kept separate despite its size because it describes patient behaviour rather than a facility.

Group counts for discharge: Home 60,226; SNF/LTC 15,225; HomeHealth 13,006; Unknown 4,678; OtherInstitution 3,571; Rehab 1,992; AMA 622.

### One-hot encoding and reference categories
Ordered levels (glucose, A1C) are encoded as integers. Nominal variables are one-hot encoded, because a 0 to 3 coding would tell the model that 3 is greater than 2. One reference column is dropped from each group. Without that, the dummies of a group sum to 1 and become exactly collinear with the intercept, which makes the design matrix singular and breaks Newton's method. Every dummy is then read as "versus the reference".

| Group | Reference (dropped) |
|---|---|
| Discharge group | Home |
| Admission source | ER |
| Admission type | Emergency |
| Payer code | MC |
| Medical specialty | InternalMedicine |
| Each of the 7 strong drugs | No |
| Race (in `fair_*.csv` only) | AfricanAmerican |
| Diagnosis categories (diag_1, diag_2, diag_3) | Circulatory (first alphabetically) |

A row with all `race_*` columns at 0 is AfricanAmerican, which matters for the fairness slice (it appears as "Reference" in the fairness table). Home was chosen as the discharge reference because Ridge shrinks each dummy toward the reference group, so the shrinkage pulls toward a well-estimated baseline. Groupings were made by clinical meaning or frequency and never by outcome, to avoid using the label to design features.

### Final dataset
99,320 rows. **99 model features** in `X`. Race is excluded from `X` (see section 4) and kept for evaluation only. The identifiers `patient_nbr` and `encounter_id` and the target `readmit_30d` are never passed to the model.

## 4. Split and scaling (`DataSplit.py`)

- **Patient-level split**, implemented from scratch with a numpy RNG on unique patient numbers, `seed = 42`. Ratios 70/20/10 (train/test/val). No patient appears in more than one split, and all patients are covered. The split is not stratified.
- Patient IDs per split are saved in `patient_splits.csv`, for leakage tests.

| Split | Rows | Patients | Positive rate |
|---|---|---|---|
| Train | 69,396 | 48,985 | 11.35% |
| Test | 19,835 | 13,995 | 11.40% |
| Val | 10,089 | 6,999 | 11.63% |

- **Scaling**: 13 continuous columns (`time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`, `num_med_changes`, `num_med_prescribed`, `age_numeric`, `max_glu_serum`, `A1Cresult`) are standardised with the **train** mean and standard deviation (ddof=0), applied to test and val. Binary columns are not scaled. The check in section 15 rebuilds these from the raw file with train statistics only.
- **Race is excluded from the features.** Race columns, gender and age (before standardization) are saved separately in `fair_*.csv` so performance can be checked per group. Gender and age remain features. Proxies for race (for example payer code, admission source, specialty) remain in `X` and are listed as a limitation.
- Outputs: `X_*.csv`, `y_*.csv`, `fair_*.csv` for train, test and val. CSV was chosen because headers are easy to access, there are no dependencies, and inspection is easy.

## 5. Data checks

- Train design matrix with the intercept column: 100 columns. Rank checks, constant-column checks and the condition number are produced by `DataSplit.py` (the script asserts that no train column is constant).
- **Condition number of the design matrix with intercept: 154.19**, the ratio of the largest to the smallest singular value. The **condition number of the Hessian at the Newton solution is 24,808**, about the square of that. This is a modest conditioning: the unpenalized problem is well posed and Newton converges in 6 updates. It does not by itself decide whether Ridge helps prediction. That was decided by the validation sweep (section 7), where it did not.
- **[TODO: rerun and record on the 99-feature version: confirmed rank, no duplicate columns, no pair with absolute correlation above 0.9, no feature that separates the outcome perfectly, and the smallest binary column in train (the earlier figure of 60 ones was from the previous feature set).]**

## 6. Model: logistic regression from scratch (`ModelClass.py`, `model.py`)

The model is unregularized logistic regression: `p = sigmoid(w0 + w1 x1 + ... + w99 x99)`, fitted by minimising the mean negative log-likelihood. There are 100 weights, including the intercept.

Building blocks, written from scratch (the verified definitions are also collected in `model.py`, which has no side effects so later scripts can import it):
- `sigmoid(z)`, with z clipped to [-40, 40] to prevent overflow.
- `log_loss(y, p)`, with p clipped to [1e-15, 1 - 1e-15].
- `gradient(W, Xb, y) = Xbᵀ(p - y) / n`.
- `hessian(W, Xb) = Xbᵀ diag(p(1-p)) Xb / n`, computed by row scaling and never by building an n by n matrix.
- `fit_gd`: full-batch gradient descent (learning rate 0.1).
- `fit_newton`: Newton's method, with `np.linalg.solve(H, g)` and no explicit inverse.
- `ridge_gradient` / `ridge_hessian` / `fit_newton_ridge`: adds `lambda * W` to the gradient and `lambda * I` to the Hessian, with the **intercept unpenalized**.
- `predict_proba(W, Xb)` and `predict(W, Xb, threshold)`.
- `brier` and `auc` (rank formula with average ranks for ties).

All fit functions return the weights and three histories (train loss, validation loss, gradient norm) indexed by number of updates: index 0 is the starting point (loss ln 2 = 0.693147) and index k is the state after k updates.

### Verification

| Check | Result |
|---|---|
| `log_loss` against `sklearn.metrics.log_loss` on random data | Identical to all printed digits |
| `gradient` against central finite differences (h = 1e-5, 5 coordinates including the intercept) | Agree to about 1e-11 |
| `hessian` column against finite difference of the gradient | Max difference 9e-12; matrix symmetric |
| Newton against sklearn `lbfgs` (`penalty=None`) | Max coefficient difference 7e-5 (lbfgs stops early on flat directions) |
| Newton against sklearn `newton-cg`, `tol=1e-12` | Max coefficient difference 1.3e-12, intercept difference 2e-13 |
| Ridge with lambda = 0 against Newton | Identical |
| `ridge_gradient` against finite differences of the penalized objective (lambda = 0 and 1e-3) | Agree to about 1e-11; the intercept entry is unchanged by the penalty |
| Ridge (lambda = 1e-3) against sklearn `newton-cg` with `C = 1 / (lambda * n)` | Max coefficient difference 1.8e-12, intercept difference 1.3e-13 |
| `auc` and `brier` against `sklearn.metrics.roc_auc_score` and `brier_score_loss` (validation, 3 prediction sets) | Differences of 0.0 as printed; toy cases (0.75, 0.5, 0.0375) correct |
| Standard errors from the Hessian against `statsmodels` | Max difference 1.4e-15 |
| Cluster-robust standard errors against `statsmodels` (cluster by patient) | Max difference 1.4e-14 |
| Mean predicted probability against base rate, validation | 0.1149 against 0.1163 (the intercept fixes the mean on the training data, not on validation) |

## 7. Optimizers and Ridge

### Optimizers

| Method | Updates | Train loss | Val loss | Final gradient norm | Max abs coefficient gap to Newton |
|---|---|---|---|---|---|
| Newton | 6 | 0.336450 | 0.339326 | 5.4e-17 | 0 |
| Gradient descent, lr 0.1 | 5,000 | 0.3374 | 0.3401 | 1.8e-3 | 0.725 |
| Gradient descent, lr 0.1 | 50,000 | 0.3365 | 0.3393 | 7.1e-5 | 0.353 |

**[TODO: these gradient descent rows come from earlier runs. Rerun with `RUN_GD = True` in the final script, confirm they reproduce, and confirm `W_gd.npy` and `loss_history_gd.csv` exist.]**

Newton's gradient norm falls 0.69, 1.2e-1, 2.0e-2, 1.4e-3, 1.0e-5, 8.0e-10, 5.4e-17, roughly squaring the error near the end (quadratic convergence). Gradient descent has not converged after 50,000 iterations. Its remaining coefficient gap sits on the rare drug dummies (for example `rosiglitazone_Down`, `pioglitazone_Up`, `glipizide_Down`), where each gradient entry is tiny because the column is nonzero in few rows, and one learning rate moves them slowly. The losses already agree to 4 digits, so the gap costs almost nothing in predictions. Collinear directions (the intercept, `num_med_prescribed`) also lagged at 5,000 iterations but had caught up by 50,000; this is the likely explanation but was not tested separately. The Hessian condition number of 24,808 is consistent with gradient descent being slow on this problem.

For scale, a constant prediction of the training base rate (11.35%) has a validation log loss of 0.3595 (test 0.3548), so the model's log loss (0.3393 on validation) is only modestly better. Readmission within 30 days is hard to predict from these features, so calibration, AUC and the threshold analysis carry more of the story than log loss alone.

### Ridge sweep (validation used for selection, test untouched)

| lambda | Train loss | Val loss | norm(W[1:]) | rosiglitazone_Down | pioglitazone_Up | glipizide_Down |
|---|---|---|---|---|---|---|
| 0 | 0.336450 | 0.339326 | 2.0818 | -0.5377 | 0.2002 | 0.2904 |
| 1e-6 | 0.336450 | 0.339326 | 2.0772 | -0.5301 | 0.1968 | 0.2874 |
| 1e-5 | 0.336450 | 0.339329 | 2.0437 | -0.4687 | 0.1749 | 0.2685 |
| 1e-4 | 0.336464 | 0.339351 | 1.9010 | -0.2206 | 0.1065 | 0.2104 |
| 3e-4 | 0.336520 | 0.339415 | 1.7387 | -0.1033 | 0.0628 | 0.1611 |
| 1e-3 | 0.336836 | 0.339778 | 1.4044 | -0.0364 | 0.0250 | 0.0890 |
| 3e-3 | 0.337756 | 0.340805 | 0.9690 | -0.0130 | 0.0081 | 0.0372 |

**Finding: Ridge does not improve validation log loss.** Validation loss is lowest at lambda = 0 and rises as lambda grows. With 69k training rows and 99 features, the unpenalized model is not overfitting in a way that hurts predictions (train-to-val gap about 0.003 at every lambda). What Ridge does change is the weakly determined coefficients. At lambda = 1e-4, `rosiglitazone_Down` (60 rows, 4 positives) moves from -0.54 to -0.22 at a validation cost of about 2.5e-5. Section 14 shows that this move is about half a standard error, which is why Ridge cannot improve validation loss.

## 8. Evaluation protocol and model selection

- The split (section 4) was fixed before any model comparison. **Train** is used for fitting only. **Validation** is used for model selection, the Ridge sweep, threshold exploration, ablations and error analysis. **Test** is used once, at the end, for the frozen models only (section 11).
- **Metrics.** Validation log loss is the primary criterion, because it is what the model optimizes and it is a proper scoring rule. Brier score, AUC and calibration are descriptive. Accuracy is not used: always predicting "no readmission" has 88.4% validation accuracy, and Newton at threshold 0.5 reaches 88.5% while catching 2.6% of readmissions.
- **Uncertainty.** Percentile intervals (95%) from a paired bootstrap that resamples encounters (500 to 1,000 resamples; the same resamples are used for every model in a comparison).
- **Selection rule (declared while working on validation, before the test set).** Within the logistic family, validation log loss picks the unpenalized Newton fit (lambda = 0). Lambda = 1e-4 is kept as a secondary "stabilized" model, justified by coefficient stability and not by loss. Across all candidates, validation log loss favors gradient boosting (section 9). Both Newton (the from-scratch, interpretable and well-calibrated model) and gradient boosting (the model preferred by the log loss criterion) were carried to the test set. That decision was made after the validation comparison and before the test data were loaded.
- Tree baselines use settings fixed in advance and were not tuned: random forest `n_estimators=300, min_samples_leaf=50, random_state=0`; `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, early_stopping=False, random_state=0)`.
- sklearn logistic regression matches the from-scratch Newton fit to about 1e-12 (section 6), so it is a verification and not a separate baseline.

## 9. Validation results

### Metrics and baselines (validation, n = 10,089)

| Model | Log loss | AUC | Brier |
|---|---|---|---|
| Baseline (constant, training base rate 0.1135) | 0.359455 | 0.500000 | 0.102755 |
| Newton (lambda = 0) | 0.339326 | 0.669633 | 0.097906 |
| Ridge lambda = 1e-4 | 0.339351 | 0.669851 | 0.097910 |
| Random forest | 0.339661 | 0.679353 | 0.098221 |
| Gradient boosting | 0.336527 | 0.675602 | 0.096981 |

Bootstrap intervals (95%, encounters resampled):
- Newton AUC 0.6696 [0.6518, 0.6855]; Brier 0.0979 [0.0938, 0.1022].
- Brier gain of Newton over the baseline: 0.00485 [0.00365, 0.00606], about 4.7% of the baseline Brier score.
- Newton minus Ridge 1e-4, AUC: -0.00022 [-0.00062, 0.00019]. The two are indistinguishable.
- Gradient boosting minus Newton: log loss -0.00286 [-0.00488, -0.00091], AUC +0.0064 [-0.0018, 0.0150].
- Random forest minus Newton: log loss +0.0004 [-0.0017, 0.0024], AUC +0.0097 [0.0010, 0.0181].

Interpretation: the model is better than the base-rate predictor, but modestly. All three families land at an AUC of 0.67 to 0.68, which points to limited information in the features, not a modelling shortfall. Gradient boosting has a small but clear log-loss advantage. Random forest ranks slightly better but its probabilities are no better (and slightly worse calibrated, below).

### Confusion-matrix metrics (Newton, validation, thresholds for inspection only)

| Threshold | TP | FP | FN | TN | Flagged | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 30 | 21 | 1143 | 8895 | 0.5% | 0.026 | 0.998 | 0.588 |
| 0.30 | 90 | 146 | 1083 | 8770 | 2.3% | 0.077 | 0.984 | 0.381 |
| 0.20 | 246 | 595 | 927 | 8321 | 8.3% | 0.210 | 0.933 | 0.293 |
| 0.15 | 420 | 1338 | 753 | 7578 | 17.4% | 0.358 | 0.850 | 0.239 |
| 0.1163 (base rate) | 642 | 2669 | 531 | 6247 | 32.8% | 0.547 | 0.701 | 0.194 |
| 0.10 | 792 | 3857 | 381 | 5059 | 46.1% | 0.675 | 0.567 | 0.170 |
| 0.08 | 984 | 5942 | 189 | 2974 | 68.7% | 0.839 | 0.334 | 0.142 |

### Calibration

Calibration curves use 10 equal-count bins (`figures/calibration_val.png`, with 95% intervals on the observed rates). Newton's decile table tracks the diagonal: the top decile has a mean predicted risk of 0.274 against 0.286 observed. Gradient boosting is close to the diagonal with a few wobbles of about two standard errors. Random forest under-predicts at the top (about 0.21 predicted against about 0.285 observed, read from the figure). Calibration slopes on the test set are in section 11. **[TODO: print the validation calibration slopes with the updated `plots_val.py` and add them here.]**

### Cost-sensitive thresholds (validation)

A flagged patient receives an intervention. A missed readmission costs `r` times an unnecessary intervention. If probabilities are calibrated, flagging is worth it when `p * r > (1 - p)`, so the threshold is **`t = 1/(1+r)`**, with no tuning. Cost is reported per patient in units of one unnecessary intervention, `(r * FN + FP) / n`, and compared with the cheaper of two trivial policies: flag nobody (cost `r * prevalence`) or flag everybody (cost `1 - prevalence`, 0.884). Flagging everybody beats flagging nobody when `r` is below 7.6.

| r | Threshold | Newton cost | GB cost | Flag nobody | Flag everybody | Saving, Newton | Saving, GB |
|---|---|---|---|---|---|---|---|
| 1 | 0.500 | 0.1154 | 0.1159 | 0.1163 | 0.8837 | 0.0009 | 0.0004 |
| 2 | 0.333 | 0.2290 | 0.2257 | 0.2325 | 0.8837 | 0.0036 | 0.0068 |
| 3 | 0.250 | 0.3343 | 0.3280 | 0.3488 | 0.8837 | 0.0145 | 0.0208 |
| 5 | 0.167 | 0.5041 | 0.4941 | 0.5813 | 0.8837 | 0.0772 | 0.0872 |
| 10 | 0.091 | 0.7693 | 0.7532 | 1.1627 | 0.8837 | 0.1145 | 0.1305 |
| 20 | 0.048 | 0.8780 | 0.8818 | 2.3253 | 0.8837 | 0.0057 | 0.0020 |

The theory threshold costs within about 0.002 of the best threshold found by searching a grid on validation for `r` up to 5 (0.009 at `r` = 10) for Newton, so the rule works without tuning. The cost curve is flat near the optimum, so small differences between the two are noise. At `r` = 1 and `r` = 20 the model has no useful operating point: it flags almost nobody or almost everybody. The headline range is **`r` in {2, 3, 5, 10}**, with **`r` = 5 as the worked example (an assumption)**.

Bootstrap intervals for the saving over the best trivial policy (validation):

| r | Newton saving | GB saving | GB cheaper than Newton by |
|---|---|---|---|
| 2 | 0.0035 [-0.0001, 0.0072] | 0.0068 [0.0030, 0.0108] | 0.0033 [-0.0003, 0.0067] |
| 3 | 0.0146 [0.0072, 0.0229] | 0.0208 [0.0127, 0.0293] | 0.0062 [-0.0002, 0.0125] |
| 5 | 0.0777 [0.0596, 0.0965] | 0.0876 [0.0667, 0.1082] | 0.0099 [-0.0018, 0.0222] |
| 10 | 0.1144 [0.0792, 0.1481] | 0.1307 [0.0935, 0.1644] | 0.0163 [-0.0071, 0.0403] |

The model beats the trivial policies for `r` from 3 to 10 and is marginal at `r` = 2. Gradient boosting has the better point estimate at every `r`, but the cost difference between the two models is not statistically resolved on validation.

## 10. Ablations (validation only)

Six feature-set variations, each fitted with Newton on the training set and compared with the full 99-feature model by a paired bootstrap on validation (500 resamples). Differences are ablation minus baseline, so a negative log loss difference is better.

| Variant | Features | Val log loss | Val AUC | Δ log loss [95% CI] | Δ AUC [95% CI] |
|---|---|---|---|---|---|
| Baseline (all features) | 99 | 0.33933 | 0.66963 | | |
| Drop all payer columns | 89 | 0.33958 | 0.66937 | +0.00027 [-0.00019, 0.00069] | -0.00035 [-0.00290, 0.00250] |
| Drop all specialty columns | 87 | 0.33977 | 0.66795 | +0.00046 [0.00004, 0.00087] | -0.00173 [-0.00426, 0.00087] |
| Drop `payer_code_Missing` only | 98 | 0.33941 | 0.66985 | +0.00009 [-0.00020, 0.00036] | +0.00022 [-0.00140, 0.00199] |
| Drop specialty `Missing` only | 98 | 0.33933 | 0.66968 | +0.00001 [-0.00015, 0.00015] | +0.00003 [-0.00087, 0.00085] |
| `log1p` on the three visit counts | 99 | 0.33864 | 0.67229 | -0.00071 [-0.00181, 0.00032] | +0.00258 [-0.00072, 0.00615] |
| Add the 5 race dummies back | 104 | 0.33938 | 0.66940 | +0.00005 [-0.00011, 0.00020] | -0.00023 [-0.00113, 0.00068] |

- The model is robust to all of these choices: the largest change in log loss is under 0.001 and the largest in AUC about 0.003.
- With six comparisons, one interval that just excludes zero (dropping all specialty columns, +0.00046) is not treated as a finding. The effect is about 0.14% of the loss.
- `log1p` has the best point estimates, but both intervals include zero. It was not adopted: the model and feature set were frozen before the test set, and switching after seeing which variant looks best would be post-hoc selection.
- Adding race back changes nothing detectable, so excluding race costs no predictive performance in this data. That is not evidence that race has no influence on outcomes, and it does not make the model fair (section 12).
- The visit counts are standardized in `X`, so the raw counts were recovered exactly from the standardized columns (the gap between the two smallest distinct values is one raw count). The recovered counts have minimum 0 and maxima of 42, 42 and 21. After `log1p` the columns were re-standardized with train statistics only.
- Not run: splitting ICD-9 mental health and blood disorders into their own categories, and the admission group variants (section 17).

## 11. Frozen decisions and test-set results

**Frozen before the test data were loaded** (written to `frozen_config.json` by the test script):
1. Models: Newton (unpenalized, fitted on train only) and gradient boosting with the settings in section 8, both fitted on train only.
2. Threshold rule `t = 1/(1+r)`, headline `r` in {2, 3, 5, 10}, `r` = 5 as the worked example.
3. Test reporting: log loss, AUC, Brier, calibration slope, and cost and saving at the frozen thresholds with bootstrap intervals. Nothing on the test set changes the models, settings or thresholds.

The test script re-fits both models, asserts that they reproduce the validation log losses (0.339326 and 0.336527), and refuses to run again once `test_probs.csv` exists. The test set was scored once.

### Test metrics (n = 19,835, positive rate 11.40%)

| Model | Log loss [95% CI] | AUC [95% CI] | Brier [95% CI] |
|---|---|---|---|
| Baseline (constant) | 0.3548 [0.3463, 0.3649] | 0.5000 | 0.1010 [0.0978, 0.1048] |
| Newton | 0.3344 [0.3256, 0.3440] | 0.6749 [0.6644, 0.6862] | 0.0962 [0.0930, 0.0997] |
| Gradient boosting | 0.3318 [0.3231, 0.3411] | 0.6810 [0.6704, 0.6927] | 0.0954 [0.0923, 0.0988] |

- Gradient boosting minus Newton: log loss -0.0026 [-0.0040, -0.0013], AUC +0.0062 [0.0005, 0.0121]. A small but real edge for boosting, about 0.8% in log loss.
- Newton's Brier score is 4.7% better than the baseline, the same as on validation. Test AUC is within the sampling error of validation AUC.
- **Calibration slope** (logistic regression of the outcome on the logit of the predicted probability; 1 is perfect, above 1 means probabilities are too compressed toward the mean): Newton **1.010**, gradient boosting **1.133**. Newton is well calibrated. Gradient boosting's probabilities are somewhat too conservative, so the rule `1/(1+r)` is only approximately right for it.

### Cost at the frozen thresholds (test; per 1,000 patients)

The test set contains about 114 readmissions per 1,000 patients. Saving is the reduction in cost per patient compared with the cheaper trivial policy.

| r | Model | Threshold | Cost | Best trivial | Saving [95% CI] | Flagged | Caught | Missed | Unnecessary flags |
|---|---|---|---|---|---|---|---|---|---|
| 2 | Newton | 0.333 | 0.2231 | 0.2280 | 0.0049 [0.0022, 0.0076] | 18.9 | 7.9 | 106.1 | 10.9 |
| 2 | Gradient boosting | 0.333 | 0.2201 | 0.2280 | 0.0079 [0.0052, 0.0106] | 16.4 | 8.1 | 105.9 | 8.3 |
| 3 | Newton | 0.250 | 0.3259 | 0.3420 | 0.0161 [0.0108, 0.0217] | 44.0 | 15.0 | 99.0 | 29.0 |
| 3 | Gradient boosting | 0.250 | 0.3210 | 0.3420 | 0.0210 [0.0155, 0.0267] | 42.3 | 15.8 | 98.2 | 26.5 |
| 5 | Newton | 0.167 | 0.5031 | 0.5700 | 0.0669 [0.0543, 0.0801] | 131.3 | 33.0 | 81.0 | 98.3 |
| 5 | Gradient boosting | 0.167 | 0.4958 | 0.5700 | 0.0742 [0.0606, 0.0882] | 151.8 | 37.7 | 76.3 | 114.1 |
| 10 | Newton | 0.091 | 0.7372 | 0.8860 | 0.1488 [0.1249, 0.1726] | 556.2 | 87.2 | 26.8 | 469.0 |
| 10 | Gradient boosting | 0.091 | 0.7351 | 0.8860 | 0.1509 [0.1260, 0.1749] | 541.9 | 86.1 | 27.9 | 455.8 |

- Every saving interval excludes zero on the test set, including `r` = 2, which only touched zero on validation. In relative terms Newton saves about 2% of the best trivial policy's cost at `r` = 2, 4.7% at `r` = 3, 11.7% at `r` = 5 and 16.8% at `r` = 10.
- **Worked example, r = 5, Newton:** per 1,000 patients the model flags 131, catches 33 of the 114 readmissions, misses 81 and raises 98 unnecessary flags. About 25% of flagged patients are real readmissions (11.4% overall), and the model catches 2.2 times as many readmissions as flagging the same number of patients at random.
- At `r` = 10 the model catches about three quarters of readmissions but must flag more than half of all patients, with precision around 16%.
- Gradient boosting minus Newton in cost: cheaper by 0.0031 [0.0008, 0.0056] at `r` = 2 and 0.0049 [0.0005, 0.0091] at `r` = 3. At `r` = 5 (0.0073 [-0.0009, 0.0150]) and `r` = 10 (0.0025 [-0.0147, 0.0202]) the intervals include zero. Boosting slightly favored at low cost ratios, not distinguishable at high ones.
- Validation and test agree: Newton's `r` = 5 saving was 0.0777 on validation and 0.0669 on test.

## 12. Fairness audit (test set, frozen predictions)

Descriptive audit of the frozen Newton model at `r` = 5 (threshold 0.167), by race, gender and age bin, using `fair_test.csv`. Intervals on sensitivity are 95% Wilson intervals. No threshold was adjusted per group.

| Group | n | Readmissions | Prevalence | Mean predicted | AUC | Flagged | Sensitivity [95% CI] | False-positive rate |
|---|---|---|---|---|---|---|---|---|
| **All** | 19,835 | 2,261 | 0.114 | 0.115 | 0.675 | 0.131 | 0.290 [0.271, 0.309] | 0.111 |
| Caucasian | 14,873 | 1,730 | 0.116 | 0.116 | 0.679 | 0.133 | 0.295 [0.274, 0.317] | 0.111 |
| AfricanAmerican (reference) | 3,715 | 423 | 0.114 | 0.117 | 0.656 | 0.140 | 0.286 [0.245, 0.331] | 0.122 |
| Hispanic | 415 | 42 | 0.101 | 0.105 | 0.690 | 0.092 | 0.214 [0.117, 0.359] | 0.078 |
| Other | 306 | 30 | 0.098 | 0.107 | 0.572 | 0.108 | 0.233 [0.118, 0.409] | 0.094 |
| Unknown | 425 | 28 | 0.066 | 0.096 | 0.699 | 0.071 | 0.179 [0.079, 0.356] | 0.063 |
| Asian | 101 | 8 | 0.079 | 0.099 | 0.671 | 0.069 | 0.250 [0.071, 0.591] | 0.054 |
| Gender 0 (female) | 10,707 | 1,243 | 0.116 | 0.115 | 0.667 | 0.129 | 0.275 [0.251, 0.301] | 0.109 |
| Gender 1 (male) | 9,128 | 1,018 | 0.112 | 0.116 | 0.685 | 0.134 | 0.307 [0.280, 0.336] | 0.113 |
| Age under 50 | 3,110 | 352 | 0.113 | 0.112 | 0.734 | 0.119 | 0.358 [0.310, 0.409] | 0.088 |
| Age 50 to 59 | 3,481 | 346 | 0.099 | 0.110 | 0.727 | 0.124 | 0.332 [0.285, 0.384] | 0.100 |
| Age 60 to 69 | 4,351 | 445 | 0.102 | 0.109 | 0.635 | 0.112 | 0.218 [0.182, 0.259] | 0.100 |
| Age 70 to 79 | 5,059 | 639 | 0.126 | 0.118 | 0.665 | 0.135 | 0.271 [0.238, 0.306] | 0.116 |
| Age 80 and over | 3,834 | 479 | 0.125 | 0.127 | 0.619 | 0.165 | 0.301 [0.261, 0.343] | 0.146 |

Findings:
- **Age shows the clearest difference.** Sensitivity is 36% under 50 and 22% at 60 to 69; these two intervals do not overlap. Gradient boosting shows the same pattern (43% against 29%). Ranking quality (AUC) is highest under 60 (0.73) and lowest for ages 60 to 69 and 80 and over (0.62 to 0.64). Patients 80 and over also get the most unnecessary flags (false-positive rate 14.6% against 8.8% under 50). Part of the difference in flag rates follows real differences in prevalence, but the sensitivity gap is a performance difference between groups.
- **Race: no detectable difference** between the two large groups (sensitivity 29.5% and 28.6%). The Asian, Hispanic, Other and Unknown groups have only 8 to 42 readmissions each, so their intervals are too wide to support any claim. The model over-predicts risk somewhat for Unknown and Asian patients (mean predicted 0.096 and 0.099 against prevalence 0.066 and 0.079), which is a possible pattern and not a finding.
- **Gender: no clear difference.** Sensitivities of 27.5% and 30.7% have overlapping intervals, and false-positive rates are almost equal.
- Caveats: about 14 groups were compared at one threshold, with no multiple-comparison adjustment. Race is excluded from the features, but proxies remain. Calibration, flag rate, sensitivity and false-positive rate cannot all be equalized when prevalence differs between groups, so this audit does not show that the model is "fair", and no per-group thresholds were tried because the test set had already been used. Results for `r` = 3 and 10 are in `fairness_test.csv`.

## 13. Error analysis (validation, Newton, r = 5)

At threshold 1/6 on validation the model gives 355 true positives, 818 false negatives, 996 false positives and 7,920 true negatives.

- **Risk deciles.** The top decile has an observed readmission rate of 28.6% (2.5 times the base rate) and holds 24.6% of all readmissions; the top three deciles hold 51.7%. Readmissions are thinly spread: the bottom half of the scores still contains about 30% of them.
- **Errors resemble their neighbours.** Missed readmissions look like patients who were correctly left alone: 61% have no prior inpatient visit (74% of true negatives), with mean predicted risk 0.105 against 0.093. False alarms look like caught readmissions: 2.4 prior inpatient visits (3.0 for true positives), and rehab discharge in 14% (16%). Heavy prior utilization and rehab discharge drive the flags, and among flagged patients the features barely separate who actually returns (precision 26%).
- **The main limitation.** Patients with no prior inpatient visit are 66% of validation patients and 48% of readmissions, and the model catches 11% of them. Within that group, patients with a home discharge (4,231 patients) have 283 readmissions, 24% of all readmissions, a readmission rate of 6.7% and sensitivity 0.0. 6.7% is below the 16.7% break-even implied by `r` = 5 (flagging a whole group pays only when its rate exceeds `1/(1+r)`, which here would need `r` above about 14), so not flagging them is what the cost rule prescribes. The remaining features carry little signal for these patients.

| Prior inpatient visits | n | Mean predicted | Observed |
|---|---|---|---|
| 0 | 6,620 | 0.091 | 0.084 |
| 1 | 1,936 | 0.121 | 0.132 |
| 2 | 780 | 0.160 | 0.192 |
| 3 to 4 | 563 | 0.217 | 0.238 |
| 5 to 7 | 138 | 0.333 | 0.326 |
| 8 or more | 52 | 0.606 | 0.577 |

- Calibration holds across the range of prior visits, with small under-prediction at 1 to 4 visits (about two standard errors at 2 visits) and no sign of over-prediction for extreme utilizers. The 8-plus group is small, so its margin of error is wide.
- Individual high-risk false alarms (patients with 12 to 14 prior inpatient visits and predicted risk above 0.8) are cases where the readmission simply did not happen. Individual records are not published; the case-level printout was used only for analysis.
- These errors indicate limited information in the recorded features, not a fitting failure: gradient boosting and logistic regression behave similarly.

## 14. Coefficients and standard errors (training data)

Standard errors come from the Hessian at the Newton solution, `sqrt(diag(inv(H)) / n)`, and were verified against `statsmodels` (1.4e-15). Because 23.2% of training patients have more than one encounter, a **cluster-robust (sandwich) version, clustered by patient**, is the main table (verified against `statsmodels`, 1.4e-14). Continuous columns are standardized, so their odds ratios are per standard deviation. Dummy columns are "versus the reference". The intercept is a baseline log-odds, not an odds ratio. Odds ratios and intervals for all 100 coefficients are in `coef_se_cluster.csv`.

| Feature | Coefficient | Cluster SE | z | Odds ratio [95% CI] |
|---|---|---|---|---|
| number_inpatient | 0.3229 | 0.0104 | 31.2 | 1.38 [1.35, 1.41] |
| discharge_group_Rehab | 1.2616 | 0.0704 | 17.9 | 3.53 [3.08, 4.05] |
| discharge_group_OtherInstitution | 0.7831 | 0.0570 | 13.7 | 2.19 [1.96, 2.45] |
| discharge_group_SNF_LTC | 0.3501 | 0.0389 | 9.0 | 1.42 [1.32, 1.53] |
| number_diagnoses | 0.0903 | 0.0163 | 5.5 | 1.09 [1.06, 1.13] |
| diag_1_category_Respiratory | -0.2202 | 0.0443 | -5.0 | 0.80 [0.74, 0.88] |
| discharge_group_HomeHealth | 0.1875 | 0.0398 | 4.7 | 1.21 [1.12, 1.30] |
| diabetesMed | 0.2140 | 0.0467 | 4.6 | 1.24 [1.13, 1.36] |
| diag_2_category_Neoplasms | 0.3595 | 0.0818 | 4.4 | 1.43 [1.22, 1.68] |
| payer_code_Missing | 0.1319 | 0.0339 | 3.9 | 1.14 [1.07, 1.22] |
| discharge_group_AMA | 0.4474 | 0.1405 | 3.2 | 1.56 [1.19, 2.06] |
| intercept | -2.5648 | 0.1142 | -22.5 | (baseline) |

- 30 of the 100 coefficients have p < 0.05 with the naive standard errors and 29 with the cluster-robust ones. 12 (naive) and 11 (cluster) survive a Bonferroni correction for 100 tests.
- Cluster-robust standard errors are on median only 0.6% larger than the naive ones (ratio 1.006), with a maximum ratio of 1.255 for `number_emergency`. Repeat encounters from the same patient barely changed the uncertainty in this model.
- Prior inpatient visits are by far the strongest signal. Discharge destination also matters: relative to a home discharge, odds are about 3.5 times higher for rehab. This relies on the discharge disposition being known at prediction time (section 0).
- Statistical significance is not predictive value: `payer_code_Missing` has z = 3.9, yet dropping it changes validation log loss by only 0.00009 (section 10).
- **Rare columns and Ridge.** With naive standard errors, the rare drug dummies are poorly determined: `rosiglitazone_Down` has SE 0.58 (z = -0.9), `pioglitazone_Up` 0.35 (z = 0.6) and `glipizide_Down` 0.29 (z = 1.0). Ridge at lambda = 1e-4 moves them by 0.54, 0.27 and 0.28 standard errors, and the largest move across all coefficients (`discharge_group_Rehab`) is 0.67 standard errors. Ridge therefore shrinks weakly determined coefficients by less than their own uncertainty, which is consistent with it not improving validation loss.
- Coefficients describe association. A large coefficient does not mean that changing that feature would change readmission.

## 15. Leakage and preprocessing checks

### Encounter-level leakage demonstration (train and validation pooled; the test set is not used)

The experiment compares the actual patient-level validation result with a naive random split of encounters (same sizes, 5 seeds), where visits of the same patient can fall on both sides. "GB flexible" (`max_iter=300, learning_rate=0.1, max_leaf_nodes=63`) is a demo-only model that can memorize, and is not part of any result.

| Model | Evaluation | Train AUC | Eval log loss | Eval AUC | Eval Brier |
|---|---|---|---|---|---|
| Newton | patient-level (actual) | 0.6642 | 0.3393 | 0.6696 | 0.0979 |
| Newton | random encounters (mean of 5, SD) | 0.6661 | 0.3368 (0.0056) | 0.6556 (0.0058) | 0.0966 (0.0020) |
| GB (frozen settings) | patient-level (actual) | 0.7149 | 0.3365 | 0.6756 | 0.0970 |
| GB (frozen settings) | random encounters | 0.7145 | 0.3335 (0.0062) | 0.6639 (0.0091) | 0.0957 (0.0021) |
| GB flexible (demo only) | patient-level (actual) | 0.9335 | 0.3414 | 0.6674 | 0.0981 |
| GB flexible (demo only) | random encounters | 0.9331 | 0.3417 (0.0069) | 0.6470 (0.0085) | 0.0974 (0.0023) |

**No inflation from splitting by encounter was found.** Random-encounter AUC is lower, not higher, and log loss differences are about 0.003, within the spread across seeds. Even a model that fits the training set almost perfectly (train AUC 0.93) gains nothing. The most likely reason is that the features describe the encounter and nothing in them identifies the patient (an explanation that was not tested). The split by patient is correct by design for the real question, predicting for patients the model has not seen, but this experiment does not show that it changed the numbers here. The five random splits overlap, so their spread is only a rough guide.

### Train-only standardization check

The 13 standardized columns were rebuilt from the raw `Final_Data.csv` using the mean and standard deviation of the training patients only. The rebuilt values match `X_train`, `X_val` and `X_test` to a maximum error of 3.6e-15, and the dummy columns are unchanged. Validation and test columns have means slightly off zero (largest absolute column mean 0.029 and 0.025), as expected when training statistics are applied. Standardizing with all rows instead would shift single extreme cells by up to 5, 1.8 and 9 standard deviations (train, val, test) because the visit counts are heavy-tailed, so the check would detect a leak.

**Not yet verified:** the top-12 specialty list, the 1% rare-payer threshold and the 98.46% drug-variance threshold were computed on all rows before the split (without using labels, apart from the disclosure in section 3). A recomputation on the training patients only has not been done. **[TODO]**

## 16. Limitations

1. **Weak signal.** The best AUC is about 0.68. At `r` = 5 about 70% of readmissions are missed, and catching about three in four at `r` = 10 means flagging more than half of all patients. This is not a model a hospital could rely on by itself.
2. **Assumed costs.** The cost ratio `r` is assumed, and an intervention is treated as a perfect fix for a flagged readmission. Real costs, intervention capacity and intervention effectiveness would change the picture.
3. **Probability calibration for the rule.** The threshold `1/(1+r)` assumes calibrated probabilities. Newton is well calibrated (test slope 1.010) but gradient boosting is somewhat conservative (1.133), and random forest under-predicts at the top.
4. **Uncertainty is bootstrapped over encounters.** Repeat encounters of one patient were not resampled together. The comparison of cluster-robust and naive standard errors (section 14) suggests the effect is small, but this was not checked for the metric intervals.
5. **Fairness.** Race is not a model feature, but proxies for race remain in `X` (payer code, admission source, specialty and others). The audit is descriptive, covers about 14 groups at one threshold, has low power for small groups, and found a clear performance gap by age (lower sensitivity and higher false-positive rates for older patients). Caucasian is the largest group but AfricanAmerican is the reference category in the fairness files.
6. **Rare drug columns** (for example `rosiglitazone_Down`, 60 rows and 4 positives) have noisy coefficients and should not be interpreted one by one. Ridge shrinks them, but does not improve validation loss.
7. **Heavy-tailed visit counts** (up to about 50 standard deviations). `log1p` gave the best ablation point estimates, but the effect was not statistically clear and it was not adopted.
8. **Age** is encoded as bin midpoints, which assumes a linear effect on the log-odds.
9. **Preprocessing choices** (top-12 specialties, rare-payer threshold, drug variance threshold) were computed on the full data without labels, and the 98.46% threshold was chosen while viewing readmission rates; a train-only recomputation is still to do (section 15).
10. **Scope.** Patients who died or went to hospice are excluded, the data covers 1999 to 2008 at US hospitals, and results describe prediction and not causes.
11. **Model form.** Logistic regression assumes linear effects on the log-odds and no interactions. Random forest and gradient boosting are included for that reason, and perform only slightly better. The tree models used fixed, untuned settings.
12. **Standard errors** assume the model is correctly specified. A coefficient's size measures association only.
13. **The test set was used once**, for the two frozen models. Anything added after that, such as recalibrating gradient boosting or group-specific thresholds, would be a post-hoc analysis.

## 17. Remaining work

1. Build the website (static, GitHub Pages): sections for question, data, preprocessing, baseline and methods, evaluation, results, errors, limitations, conclusion and reproduction steps, plus a cost-ratio slider showing flagged, caught, missed and unnecessary flags per 1,000 patients, and the saving, from a precomputed `results.json` (aggregates only, no patient-level rows).
2. Plots: data understanding (class balance and a few key features), fairness by group with intervals, and the cost saving against `r`.
3. Train-only recomputation of the specialty top-12, rare-payer and drug-variance rules (section 15).
4. Optional ablations not yet run: ICD-9 mental health and blood disorder categories, and admission group variants.
5. Fill the **[TODO]** items in this file and finish the final README pass.
6. Viva preparation: each student should be able to explain their contribution, key choices, the evaluation design, a limitation, and the effect of a small change (for example a different `r`, or `log1p` on visit counts).

## 18. Files and reproducibility

Run in this order: `Data_Loading_Processing.py`, `Cleaned_Data_Encoding.py`, `DataSplit.py` (produces `Final_Data.csv`-based splits; **[TODO: confirm which script writes `Final_Data.csv`]**), `ModelClass.py` (set `RUN_GD = True` once for the gradient descent run), then the analysis scripts, which read the saved outputs and `model.py`:

| Script | Purpose | Uses test data? |
|---|---|---|
| `model.py` | Verified function definitions, no side effects | no |
| metrics script | AUC, Brier, confusion metrics, bootstrap, tree baselines, `val_probs.csv` | no |
| `plots_val.py` | ROC, calibration (with intervals and slopes), convergence figures | no |
| `cost_val.py` | Threshold table, cost summary, saving bootstrap | no |
| `ablations_val.py` | Six feature-set ablations | no |
| `errors_val.py` | Error analysis | no |
| `standard_errors.py`, `standard_errors_cluster.py` | Naive and cluster-robust coefficient uncertainty | no |
| `leakage_demo.py` | Encounter-level leakage demonstration | no |
| `preproc_check.py` | Train-only standardization check | reads raw split labels only |
| test-set script **[TODO: filename]** | Single test evaluation, writes `frozen_config.json` and `test_probs.csv`; refuses to run twice | yes, once |
| `fairness_test.py` | Fairness audit from `test_probs.csv` and `fair_test.csv` | yes (frozen predictions) |

All paths use `BASE = Path(r'E:\ML') / 'Data'`. Seeds: split 42; models `random_state=0`; bootstrap seeds are set in each script. Dependencies are in `requirements.txt`. **[TODO: add scikit-learn, scipy, matplotlib and statsmodels (the last only for the cross-checks); confirm pandas and numpy versions.]**

Saved outputs (aggregates and model weights, no patient-level rows should be committed): `W_nt.npy`, `W_gd.npy`, `W_ridge_1e-4.npy`, `W_ridge_best.npy`, `loss_history_newton.csv`, `loss_history_gd.csv`, `coefficients.csv`, `ridge_sweep.csv`, `metrics_val.csv`, `metrics_val_all.csv`, `confusion_val_newton.csv`, `threshold_table_val.csv`, `cost_summary_val.csv`, `ablations_val.csv`, `coef_se.csv`, `coef_se_cluster.csv`, `error_groups_val.csv`, `leakage_demo.csv`, `leakage_demo_raw.csv`, `frozen_config.json`, `test_metrics.csv`, `test_cost.csv`, `fairness_test.csv`, `figures/`. **[TODO: confirm each file exists.]** `val_probs.csv` and `test_probs.csv` contain one predicted probability per patient encounter and should stay out of the repository like the other row-level files.

## 19. AI use disclosure

Claude (Anthropic) was used for code review, debugging, explanation and planning, and to draft the analysis scripts after the model itself was working: the evaluation metrics, baselines, bootstrap, calibration, cost-threshold, ablation, fairness, error-analysis, standard-error, leakage and preprocessing-check scripts, and drafts of this README. The core code (preprocessing, sigmoid, loss, gradient, Hessian, Newton's method, Ridge) was written by the student, with review. Every drafted script was run by the student on the real data and its outputs were checked: the from-scratch metrics against scikit-learn, the standard errors against statsmodels, and the models against scikit-learn solvers (section 6). The student is responsible for every result and can explain any part of the work. Antigravity's AI features were used as applicable. **[TODO: state exactly how, and edit this paragraph so it is true to what you actually did.]**

## 20. References

- Strack, B., DeShazo, J. P., Gennings, C., et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*, 2014, Article ID 781670. **[TODO: confirm volume and article ID against the paper.]**
- OpenML dataset 4541, Diabetes 130-US hospitals for years 1999 to 2008.
