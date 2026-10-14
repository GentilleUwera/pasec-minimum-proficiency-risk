"""Leakage-free preprocessing pipeline (implemented in Phase 2).

Design fixed in Phase 1 (see phase1_strategy/Phase1_Strategy_Report.pdf):
1. Split first. Hold out a 20% test set with StratifiedGroupKFold
   (groups = school, i.e. ID_PAYS + ID_ECOLE; strata = country x label).
2. Every imputer, encoder and scaler lives inside a scikit-learn Pipeline,
   so it is fitted on X_train only and re-fitted inside each CV fold.
3. Steps shared by both models (ColumnTransformer by data type):
   - numeric (ses, qe62, qm22): within-country mean/median imputation with
     missing indicators.
   - binary and ordinal (qe621a, qe621b, qe68, qm5, qd31, qe620):
     within-country mode imputation with missing indicators, ordered
     integer codes.
   - nominal (ID_PAYS): one-hot encoding, unknown categories ignored.
4. Logistic regression only: winsorizing of the numeric features (z-score
   for ses, 1.5 x IQR for qe62 and qm22, bounds learned on the training
   fold), log1p for qm22, StandardScaler.
5. XGBoost: original values, no capping, log transform or scaling.
"""
