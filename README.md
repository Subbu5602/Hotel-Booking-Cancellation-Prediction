# Hotel Booking Cancellation Prediction

 
## Business Problem
 
Hotels lose revenue and struggle with staffing/inventory planning when bookings are canceled unpredictably. This project tries to predict **which upcoming hotel bookings are likely to be canceled**, so revenue management can adjust overbooking policy and staffing decisions beforehand.
 
A model that reliably flags high-risk bookings lets the hotel decide whether to require a deposit or follow up with a guest before the cancellation happens.


## Data Source
 
- **Dataset:** [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Kaggle)
- **Description:** 119,390 hotel bookings spanning a city hotel and a resort hotel, including booking lead time, stay length, guest composition, meal type, market segment, distribution channel, deposit type, and prior booking history.
  *(Note: some of these are combined descriptions of related raw columns — e.g. "stay length" comes from `stays_in_weekend_nights` + `stays_in_week_nights`, and "guest composition" from `adults`, `children`, and `babies`. These will be engineered into single features during the feature engineering step.)*
- **Origin:** Originally published by Antonio, Almeida & Nunes (2019) in *Data in Brief*; cleaned and redistributed via the TidyTuesday project.


## Target Variable
 
- **`is_canceled`** (binary: 1 = canceled, 0 = not canceled)
- **Note:** `reservation_status` and `reservation_status_date` will be dropped before modeling — they directly determine the outcome, and leaving them in would produce artificially perfect (and meaningless) results.


## Success Metric
 
Optimizing primarily for **F1** on the canceled class, balancing **recall** and **precision**. (Revised from an earlier recall-first framing — optimizing recall in isolation risks a model that over-flags cancellations to catch every one, so F1 better reflects the practical tradeoff between catching cancellations and avoiding unnecessary false alarms.)
 

## Roadmap
 
1] **EDA** — ✅ Done. Understand distributions, missing values (`children`, `country`, `agent`, `company`), and cancellation patterns across hotel type, lead time, and market segment.

2] **Feature engineering & model comparison** — ✅ Done. Encoded categorical columns, compared 2 models (RandomForest, XGBoost) with feature importance plots for interpretability. Found XGBoost's importance heavily concentrated on `deposit_type` (a known dataset quirk tied to bulk agency bookings) while RandomForest's is more distributed — see **02_Baseline_Model.ipynb**. Separately engineered `cancel_ratio` and `first_timer` from booking history columns (**03_Feature_Engineering.ipynb**); both changed scores by under 0.002 across all metrics — a negative result, not a bug.

3] **Hyperparameter tuning** — ✅ Done. Ran GridSearchCV and RandomizedSearchCV on RandomForest and XGBoost (80/20 stratified hold-out, refit on F1). RandomForest tuning barely moved the score (default settings were already near-optimal). XGBoost improved meaningfully: F1 0.778 → 0.820, recall 0.718 → 0.788. Selected model: RandomizedSearchCV-tuned XGBoost (F1 0.8199, recall 0.7879, precision 0.8546). See **04_HyperParameter_Tuning.ipynb** for the full results table and reasoning.

4] **Evaluation & write-up** — assess the tuned model with recall/precision/F1/ROC-AUC, and summarize findings with recommendations for what a hotel could do next with these predictions to minimize loss.


## What I'd do next

- **Time-based validation split** — the current train/test split is random, which mixes 2015–2017 bookings together. A real hotel predicts future bookings from past ones; a chronological split (train on earlier arrival dates, test on the latest) would be a stricter, more realistic check of how the model holds up over time.
- **Recall-weighted scoring** — try `fbeta_score` with `beta=2` (or similar) instead of plain F1, to more directly reflect that catching a cancellation matters more than avoiding a false alarm, without going as far as optimizing recall alone.
- **A finer third tuning pass** — narrow the search further around RandomizedSearchCV's winning region (the current best XGBoost parameters) to check whether additional gains are available nearby.
- **Imputation strategy consistency** — `02_Baseline_Model.ipynb` used median imputation for numeric columns; `src/preprocessing.py` (used from `03` onward) uses mean. Only `children` has missing values (4 rows), so this had no practical effect here, but it's worth standardizing on one strategy for consistency.
