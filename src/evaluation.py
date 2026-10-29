"""Survey-weighted training weights, metrics and cross-validation helpers.

PASEC uses a stratified cluster sample, so each student stands for a different
number of students in the population (final weight rwgt0). Metrics computed
with rwgt0 estimate population-level performance; unweighted metrics describe
the sample.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def fit_weights(y, survey_weight):
    """Training weights: rwgt0 rescaled to mean 1, times a class-balance factor.

    The balance factor gives both classes the same total weight, so the loss
    is weighted by population share without being dominated by the majority class.
    """
    y = np.asarray(y)
    w = np.asarray(survey_weight, dtype=float)
    w = w / w.mean()
    positive_share = np.average(y, weights=w)
    return w * np.where(y == 1, 0.5 / positive_share, 0.5 / (1 - positive_share))


def classification_scores(y, prob, weight=None, threshold=0.5):
    """Recall (class 1), F1, balanced accuracy and ROC AUC, optionally weighted."""
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {
        "recall": recall_score(y, pred, sample_weight=weight),
        "f1": f1_score(y, pred, sample_weight=weight),
        "balanced_accuracy": balanced_accuracy_score(y, pred, sample_weight=weight),
        "roc_auc": roc_auc_score(y, prob, sample_weight=weight),
    }


def out_of_fold_proba(pipeline, X, y, groups, strata, survey_weight=None, n_splits=5, random_state=42,
                      fit_params=None):
    """Predicted probabilities for every training row from school-grouped CV.

    The whole pipeline is cloned and re-fitted inside each fold, so imputation
    values, capping bounds and scaling never see the validation fold.
    With `survey_weight`, the model is fitted with fit_weights(y, rwgt0).
    `fit_params` are passed to the pipeline's fit (for example CatBoost's cat_features).
    """
    prob = np.full(len(X), np.nan)
    fold = np.full(len(X), -1)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for k, (fit_idx, val_idx) in enumerate(cv.split(X, strata, groups)):
        model = clone(pipeline)
        params = dict(fit_params or {})
        if survey_weight is not None:
            params["model__sample_weight"] = fit_weights(y.iloc[fit_idx], survey_weight.iloc[fit_idx])
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx], **params)
        prob[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        fold[val_idx] = k
    return prob, fold


def scores_by_fold(y, prob, fold, weight=None):
    """classification_scores for each fold, plus the mean row."""
    rows = {}
    for k in np.unique(fold):
        mask = fold == k
        rows[f"fold {k + 1}"] = classification_scores(
            y[mask], prob[mask], None if weight is None else weight[mask])
    table = pd.DataFrame(rows).T
    table.loc["mean"] = table.mean()
    return table


def leave_one_country_out(pipeline, X, y, country, survey_weight):
    """Train on all countries but one, test on the held-out country, for each country.

    Returns survey-weighted balanced accuracy and ROC AUC per held-out country.
    """
    rows = {}
    for c in sorted(country.unique()):
        held_out = (country == c).values
        model = clone(pipeline)
        model.fit(X[~held_out], y[~held_out],
                  model__sample_weight=fit_weights(y[~held_out], survey_weight[~held_out]))
        prob = model.predict_proba(X[held_out])[:, 1]
        s = classification_scores(y[held_out], prob, survey_weight[held_out].values)
        rows[c] = {"balanced_accuracy": s["balanced_accuracy"], "roc_auc": s["roc_auc"]}
    return pd.DataFrame(rows).T
