# Leakage-Free, Calibrated Probability-of-Default Modelling on Lending Club: An Out-of-Time Evaluation

**Bhanu Prakash** (2420030209) · **Krishita Myneni** (2420030443) · **Laasya Priya** (2420030002)
*Department of Data Science & Computer Science Engineering, Project Team 7*

> Revision 2. This version replaces the coursework draft (kept in `archive/coursework_v1/`).
> The earlier draft reported ROC-AUC 0.7524 for a "Hybrid Neuro-GBDT" ensemble. That figure
> came from a 100k-row sample whose features still included Lending Club's own pricing
> (`int_rate`, `grade`, `installment`), and the ensemble weights it described were fixed
> rather than optimised. Every number below is reproduced by `python run_pipeline.py` and
> the scripts named in each section.

---

### Abstract

Reported accuracy on the Lending Club dataset is often inflated by post-origination leakage,
by test periods that include loans that had not finished yet, or by features that encode the
lender's own risk grade. We build a probability-of-default (PD) model that avoids all three:
40 post-origination columns are removed, only 36-month loans that had fully matured by the
data snapshot are kept (454,788 loans issued 2013–2015), Lending Club's pricing signals are
excluded, and the model is evaluated on a strictly later out-of-time window. Seven model
families are compared. The champion, an isotonic-calibrated LightGBM chosen on a separate
calibration window, reaches **ROC-AUC 0.688** (95% CI 0.685–0.692) and **KS 0.270** on
136,712 test loans. All seven models land within 0.012 AUC of each other. Calibration
matters more than the choice of model for deployment: predicted and observed default rates
agree within 1 percentage point in every policy band (approve 6.7% vs 6.9%, review 15.3% vs
15.5%, decline 26.1% vs 27.1%). Adding Lending Club's own pricing signals raises AUC to
**0.703**, and adding three post-origination columns raises it to **0.997**,
which shows how easily headline numbers on this dataset come from information that is not
available at application time. A segment fairness audit finds even calibration across
income, region, housing and verification groups, but approval rates that fail the
four-fifths screening rule for low-income applicants and renters.

---

## I. Introduction

A PD model drives three things at a lender: whether to approve, how to price, and how much
capital to hold (`EL = PD × LGD × EAD` [2]). For the first two, ranking ability (AUC, KS) is
what matters. For the third, the probabilities themselves must be right. A model can rank
well and still be badly calibrated, especially after class re-weighting [5].

Public work on Lending Club often reports AUCs above 0.9. On this dataset that almost always
means leakage: columns like `total_pymnt` or `recoveries` describe what happened after the
loan was issued. Less obvious problems are (a) test sets that include immature loans, which
are labelled "good" only because they have not had time to default, and (b) using
`grade`/`int_rate`, which are the output of Lending Club's own underwriting model.

Our contributions are practical, not algorithmic:

1. A leakage-free pipeline restricted to matured loans, with every statistic (imputation
   medians, winsorisation bounds) learned on the training window only.
2. An honest seven-model comparison on a later time window, with the champion chosen on a
   separate calibration window.
3. Two measured ablations: the effect of Lending Club's pricing signals, and the effect of
   post-origination leakage.
4. A policy backtest of calibrated decision bands, plus a segment-level fairness audit.
5. A scoring API that reproduces training-time preprocessing exactly and returns
   per-applicant reason codes.

## II. Related Work

Logistic-regression scorecards with weight-of-evidence binning remain the industry baseline
[1], [3]. Large benchmarks find that ensembles, gradient boosting in particular, usually beat
them, but by small margins on application data [4]. GBDT implementations [6]–[8] are now
the standard non-linear benchmark for tabular credit data. Niculescu-Mizil and Caruana [5]
show that boosted trees and re-weighted models produce distorted probabilities that
isotonic or Platt calibration corrects. KS and decile lift are the conventional
credit-scoring summaries alongside AUC [1].

## III. Data

**Source.** Lending Club accepted loans 2007–2018Q4: 2,260,701 rows × 151 columns. We read
41 columns that are known at application time, plus the outcome.

**Cohort construction** (`src/cleaning.py`, logged to `artifacts/cleaning_log.json`):

| Step | Rows |
|---|---|
| Raw file | 2,260,701 |
| Resolved outcome only (Fully Paid / Charged Off / Default) | 1,348,099 |
| Issued 2013 or later | 1,252,197 |
| 36-month term only | 948,202 |
| Matured by 2018-09-01 (snapshot 2018-12 minus 3-month buffer) | 457,349 |
| Near-duplicates removed (9-column business key) | **454,788** |

The maturity filter is the step most often skipped. Without it, 2016–2018 loans that have not
reached the end of their term would be over-represented among "Fully Paid" (early payoffs)
and missing from "Charged Off" (defaults that have not happened yet). Our cohort therefore
ends with loans issued in September 2015.

**Target.** Charged Off / Default = 1, Fully Paid = 0 (including the "does not meet the
credit policy" variants).

**Leakage removal.** 40 columns that are populated after issuance are dropped before
anything else: payment totals, recoveries, last-payment fields, `last_fico_*`,
`last_credit_pull_d`, and all hardship and settlement fields (`config.LEAKAGE_COLS`).

**Data quality.** A weighted completeness / validity / consistency / uniqueness score moves
from 99.1 to 99.9 after cleaning. On this dataset the score is not very informative: the
file is already clean by these measures, and the meaningful data work is the cohort
definition above.

## IV. Features and Split

**Engineered features** (`src/features.py`): loan-to-income, average FICO, credit history
length in months, delinquency flag and severity index (2×delinquencies + 1.5×public
records + 3×bankruptcies), revolving utilisation flag (> 80%), inquiry velocity
(inquiries ÷ (years of history + 1)), balance-to-limit ratio, open-to-total account ratio,
and employment-length bands. The champion uses 26 numeric and 7 categorical features.

**Excluded by default:** `grade`, `sub_grade`, `int_rate`, `installment` and features
derived from them. They are Lending Club's own risk pricing (Section VI-C).

**Preprocessing inside the model.** Median imputation, winsorisation at the 0.5th/99.5th
percentiles, standardisation and one-hot encoding (categories under 1% frequency pooled) are
all pipeline steps fitted on the training window. A first version computed clipping
quantiles and medians in the cleaning step, over all periods at once, which leaks test-window
statistics into training. At scoring time it also made an applicant's features depend on the
other rows in the uploaded file. A unit test now guards against this.

**Out-of-time split** on non-overlapping months:

| Window | Issue dates | Loans | Default rate |
|---|---|---|---|
| Train | Jan 2013 – Dec 2014 | 261,849 | 13.2% |
| Calibration | Jan 2015 – Mar 2015 | 56,227 | 14.8% |
| Test | Apr 2015 – Sep 2015 | 136,712 | 14.9% |

The default rate rises across windows, so the test window is somewhat harder than the
training data, which is the realistic direction.

## V. Models

Seven candidates, each in the same preprocessing pipeline (`src/models.py`):
L2 logistic regression; random forest (250 trees, depth 12, min leaf 40); XGBoost (250
trees, depth 4, η 0.05); LightGBM (300 trees, 31 leaves, η 0.035); CatBoost (300
iterations, depth 5); an MLP (128-64, early stopping); and a soft-voting ensemble of
XGBoost, CatBoost and the MLP with fixed weights 0.40 / 0.35 / 0.25. Hyperparameters are
hand-set, regularised defaults. No search was run.

**Imbalance.** Class weighting (`class_weight="balanced"`, `scale_pos_weight` ≈ 6.6) rather
than SMOTE.

**Calibration.** Each fitted model is wrapped in isotonic regression fitted on the
calibration window. Isotonic regression is monotone, so ranking metrics are unchanged.

**Champion selection.** The champion is the model with the highest *calibration-window*
AUC among models with exact per-applicant attributions (boosters, random forest, logistic
regression). The test window is only used for reporting. The precision/recall operating
threshold is also tuned on the calibration window (F1-optimal).

## VI. Results

### A. Model comparison (test window, N = 136,712)

| Model | ROC-AUC [95% CI] | KS | PR-AUC | Brier | Train AUC |
|---|---|---|---|---|---|
| **LightGBM (champion)** | **0.6883** [0.6845, 0.6920] | 0.2701 | 0.2630 | 0.1198 | 0.7126 |
| Soft-voting ensemble | 0.6881 [0.6843, 0.6919] | 0.2715 | 0.2643 | 0.1197 | 0.6925 |
| XGBoost | 0.6862 [0.6824, 0.6900] | 0.2685 | 0.2622 | 0.1199 | 0.6949 |
| CatBoost | 0.6858 [0.6821, 0.6896] | 0.2698 | 0.2604 | 0.1199 | 0.6868 |
| MLP | 0.6856 [0.6819, 0.6895] | 0.2703 | 0.2625 | 0.1199 | 0.6860 |
| Logistic regression | 0.6783 [0.6746, 0.6821] | 0.2596 | 0.2531 | 0.1206 | 0.6702 |
| Random forest | 0.6765 [0.6728, 0.6805] | 0.2572 | 0.2492 | 0.1208 | 0.7635 |

CIs are 1,000-sample bootstraps. The four boosters, the MLP and the ensemble overlap with
each other. Their gain over logistic regression (~0.01 AUC) is outside the CIs, but it is
small. The ensemble is statistically indistinguishable from LightGBM, and it costs 4.7× the
training time and loses exact attributions. The random forest overfits the most (train–test
gap 0.087).

At the calibration-tuned threshold (PD ≥ 0.17), the champion flags 58.4% of defaulters with
24.5% precision. Accuracy is not a useful metric at a 15% base rate and is not used for any
decision.

**Calibration** reduced the champion's test Brier score from 0.226 to 0.120 with no change
in ranking.

### B. Decile lift (champion)

| Decile | Default rate | Avg predicted PD | Lift | Cumulative defaults |
|---|---|---|---|---|
| 1 (riskiest) | 32.1% | 31.1% | 2.15× | 21.5% |
| 2 | 24.4% | 23.2% | 1.64× | 37.8% |
| 3 | 20.4% | 19.8% | 1.36× | 51.5% |
| 5 | 15.0% | 14.7% | 1.01× | 72.8% |
| 8 | 8.3% | 8.2% | 0.56× | 93.8% |
| 10 (safest) | 3.3% | 3.1% | 0.22× | 100.0% |

Default rates fall monotonically across deciles. Predicted and actual rates agree within
1.3 points in every decile.

### C. Ablation: Lending Club's pricing signals

Re-running the same pipeline with `--with-lc-signals` adds `grade`, `int_rate`,
`installment` and three features derived from them. Champion test AUC moves from 0.688 to
**0.703** (KS 0.299). Lending Club's grade therefore adds about 0.015 AUC on top of the borrower's own
attributes, so most of what the grade knows is already in the application data. All seven models
are in `artifacts/ablation_lc_signals.json`. Even with these signals, the matured
cohort does not come close to the 0.75 reported in the coursework draft, and that figure
could not be reproduced.

### D. Ablation: post-origination leakage

`scripts/leakage_demo.py` trains the same XGBoost on the same split, once on
application-time features and once with three post-origination columns (`total_pymnt`,
`recoveries`, `last_pymnt_amnt`) added:

| Features | ROC-AUC | KS |
|---|---|---|
| Application-time only | 0.687 | 0.269 |
| + 3 post-origination columns | **0.997** | 0.968 |

The leaked model looks excellent and is useless: those columns do not exist when the credit
decision is made.

### E. Drivers

Mean absolute TreeSHAP contribution on 5,000 test loans, with one-hot columns summed back
into their source feature: average FICO (0.243), accounts opened in the past 24 months
(0.219), income (0.136), DTI (0.120), loan-to-income (0.110), revolving balance (0.092),
home ownership (0.079), loan purpose (0.075), inquiry velocity (0.075), share of bankcards
above 75% utilisation (0.075).

### F. Stability

Population stability index between the training and test windows is 0.006 for the score and
under 0.03 for every monitored feature, well inside the conventional 0.10 "stable" limit.

## VII. Decision Policy and Backtest

PDs map to a score by points-to-double-the-odds scaling (PD 20% → 660, +40 points per
doubling of the good:bad odds, bounded to 300–850) and to a decision by fixed cut-offs
defined once in `config.py`:

| Decision | PD band | Share of test loans | Predicted default | Actual default |
|---|---|---|---|---|
| APPROVE | ≤ 10% | 39.4% | 6.7% | 6.9% |
| REVIEW | 10–20% | 36.1% | 15.3% | 15.5% |
| DECLINE | > 20% | 24.5% | 26.1% | 27.1% |

Fixed cut-offs, rather than quantiles, keep a band's meaning constant as the applicant mix
changes. Expected loss uses a flat LGD of 0.60, the usual assumption for unsecured
consumer credit.

## VIII. Fairness Audit

Lending Club does not publish protected attributes, so we audit four available proxy
segments (`src/fairness.py`, `artifacts/fairness_report.csv`):

| Segment | Lowest-approved group (rate) | Highest (rate) | Min adverse impact ratio | Max calibration gap |
|---|---|---|---|---|
| Income quartile | Q1, 15.1% | Q4, 63.5% | 0.24 | 0.9 pp |
| Home ownership | Rent, 27.0% | Mortgage, 52.3% | 0.52 | 0.6 pp |
| Income verification | Verified, 25.6% | Not verified, 54.5% | 0.47 | 1.2 pp |
| Region | Midwest, 38.7% | Northeast, 40.6% | 0.95 | 1.0 pp |

The model is equally well calibrated in every group, and approved loans default at 6.4–7.8%
across all segments, so it is not systematically wrong for any group. Approval rates still
differ sharply and fail the four-fifths screening rule for income, housing and verification.
These factors are legitimate underwriting inputs and the groups really do default at
different rates (lowest income quartile 19.2% vs highest 11.1%). In a real deployment this
result would trigger a documented business-necessity review and a search for less
discriminatory alternatives, not automatic approval or rejection of the model. A known
Lending Club quirk is also visible: *verified* income goes with higher default, because
verification was requested for riskier applications.

## IX. System

A FastAPI service (`app.py`) accepts CSV uploads of up to 5 MB, validates each row
(invalid rows are reported with row number and reason while the rest are scored), refuses
to score 60-month loans (outside the model's validated scope), and applies the exact
training-time cleaning, feature and preprocessing code. Each applicant gets a PD, score,
band, decision, expected loss and the four largest attributions as plain-English reason
codes. Thresholds, LGD and benchmarks are read from the saved model bundle. Measured
in-process latency on a 12-core desktop: a single applicant takes 46 ms (median, p95 49 ms) and a 300-row batch 280 ms (p95 321 ms), including validation, features, scoring and reason codes.

## X. Limitations

- **Selection bias.** Only loans that Lending Club approved are observed. PDs for applicants
  outside that population are extrapolations. Reject inference using the rejected-applications
  file is the natural next step.
- **Single test window.** Six months of 2015. Walk-forward evaluation across several windows
  would give a more reliable read on stability.
- **No hyperparameter search.** Given how close all the models are, we expect search to add
  little, but we have not shown that.
- **Flat LGD** and **proxy-only fairness**, as discussed above.

## XI. Conclusion

On matured, leakage-free Lending Club data with the lender's own pricing removed,
application-time information supports an out-of-time AUC of about 0.69. Gradient boosting
adds about 0.01 over logistic regression, and more elaborate ensembles add nothing
measurable. The practically important result is calibration: after isotonic recalibration,
the predicted default rate in each policy band matches the observed rate within about a
point, which is what an approve/review/decline policy and an expected-loss estimate need.
Together, the ablations and the fairness audit show what the model can and cannot claim.

---

## References

[1] L. C. Thomas, D. B. Edelman, and J. N. Crook, *Credit Scoring and Its Applications*. SIAM, 2002.
[2] Basel Committee on Banking Supervision, "International Convergence of Capital Measurement and Capital Standards: A Revised Framework," BIS, 2006.
[3] D. J. Hand and W. E. Henley, "Statistical classification methods in consumer credit scoring: A review," *J. Royal Statistical Society A*, vol. 160, no. 3, pp. 523–541, 1997.
[4] S. Lessmann, B. Baesens, H.-V. Seow, and L. C. Thomas, "Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research," *European J. Operational Research*, vol. 247, no. 1, pp. 124–136, 2015.
[5] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in *Proc. ICML*, 2005, pp. 625–632.
[6] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. KDD*, 2016, pp. 785–794.
[7] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Proc. NeurIPS*, 2017, pp. 3146–3154.
[8] L. Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features," in *Proc. NeurIPS*, 2018, pp. 6638–6648.
