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
 
Optimizing primarily for **recall** on the canceled class, with **precision** and **F1** as supporting metrics. A missed cancellation costs the business more than an unnecessary retention/deposit follow-up on a false positive — the model should err on the side of caution.
 

## Roadmap
 
1] **EDA** — understand distributions, missing values (`children`, `agent`, `company`), and cancellation patterns across hotel type, lead time, and market segment.

2] **Feature engineering & model comparison** — encode the categorical columns for the models, engineer features (e.g. `total_stay_length`, `total_guests`) for ease of reading for the model, and compare at least 2–3 models for the sake of sanity (e.g. logistic regression, random forest, gradient boosting).

3] **Evaluation & write-up** — assess with recall / precision / F1 / ROC-AUC, and summarize findings with recommendations for what a hotel could do next with these predictions to minimize loss.
