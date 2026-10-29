"""Leakage-free preprocessing pipelines.

Design (reports/Phase1_Strategy_Report.pdf):
- the split comes first; every step below is fitted on X_train only and
  re-fitted inside each cross-validation fold;
- shared steps: within-country imputation with missing indicators, ordinal
  encoding of ordered answers, and a country representation;
- logistic regression only: winsorizing (before imputation, so the bounds come
  from observed values), log(1 + x) for class size, scaling;
- gradient boosting (XGBoost, CatBoost): original numeric values (tree splits are not
  distorted by extremes); CatBoost receives country as one categorical column.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from .transformers import (
    BINARY_FEATURES,
    CONTEXT_COLUMNS,
    COUNTRY,
    NUMERIC_FEATURES,
    ORDINAL_CATEGORIES,
    ORDINAL_FEATURES,
    CountryContext,
    CountryImputer,
    CountryIterativeImputer,
    Log1pColumns,
    Winsorizer,
)

INDICATOR_FEATURES = [f"{c}_missing" for c in NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES]
MODELS = ("logreg", "xgboost", "catboost")
COUNTRY_MODES = ("onehot", "context", "none")
IMPUTERS = ("country", "iterative", "none")


def _as_integer_codes(X):
    """Country codes as integers, the form CatBoost expects for a categorical column."""
    return X.astype(int)


def _numeric_branch(model):
    if model == "logreg":
        return Pipeline([
            ("log", Log1pColumns(columns=("qm22",))),
            ("scale", StandardScaler()),
        ])
    return "passthrough"


def _binary_branch(model):
    return StandardScaler() if model == "logreg" else "passthrough"


def _ordinal_branch(model, keep_missing=False):
    if keep_missing:
        # Gaps stay NaN so XGBoost can learn a default direction for them.
        encoder = OrdinalEncoder(categories=ORDINAL_CATEGORIES, handle_unknown="use_encoded_value",
                                 unknown_value=np.nan)
    else:
        encoder = OrdinalEncoder(categories=ORDINAL_CATEGORIES)
    steps = [("encode", encoder)]
    if model == "logreg":
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def build_preprocessor(model="logreg", country="onehot", external=None, imputer="country"):
    """Imputation followed by a ColumnTransformer that routes columns by type.

    model: "logreg" (capping, log and scaling), "xgboost" (no capping or scaling) or
    "catboost" (as xgboost, but country stays one categorical column for CatBoost).
    country: "onehot" (one column per country), "context" (country averages of the
    survey answers, plus the `external` indicators if given) or "none".
    imputer: "country" (within-country mean/median/mode, the default), "iterative"
    (MICE-style sensitivity check) or "none" (gaps left as NaN, XGBoost only).
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if country not in COUNTRY_MODES:
        raise ValueError(f"country must be one of {COUNTRY_MODES}")
    if imputer not in IMPUTERS or (imputer == "none" and model == "logreg"):
        raise ValueError(f"imputer must be one of {IMPUTERS}; 'none' only with the tree models")

    branches = [
        ("numeric", _numeric_branch(model), NUMERIC_FEATURES),
        ("binary", _binary_branch(model), BINARY_FEATURES),
        ("ordinal", _ordinal_branch(model, keep_missing=imputer == "none"), ORDINAL_FEATURES),
    ]
    if imputer != "none":
        branches.append(("missing_flags", "passthrough", INDICATOR_FEATURES))
    if country == "onehot" and model == "catboost":
        branches.append(("country", FunctionTransformer(_as_integer_codes, feature_names_out="one-to-one"), [COUNTRY]))
    elif country == "onehot":
        branches.append(("country", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [COUNTRY]))
    elif country == "context":
        context_columns = list(CONTEXT_COLUMNS) + ([] if external is None else list(external.columns))
        branches.append(("country", StandardScaler() if model == "logreg" else "passthrough", context_columns))

    steps = []
    if imputer == "country":
        steps.append(("impute", CountryImputer()))
    elif imputer == "iterative":
        steps.append(("impute", CountryIterativeImputer()))
    if country == "context":
        steps.append(("context", CountryContext(external=external)))
    steps.append(("columns", ColumnTransformer(branches, verbose_feature_names_out=False)))
    if model == "logreg":
        # Capping comes before imputation so the bounds are learned from real
        # answers only, not from the filled-in values.
        steps.insert(0, ("cap", Winsorizer()))
    return Pipeline(steps).set_output(transform="pandas")


def build_pipeline(estimator, model="logreg", country="onehot", external=None, imputer="country"):
    """Full pipeline: preprocessing plus the estimator, fitted as one object."""
    return Pipeline([("preprocess", build_preprocessor(model, country, external, imputer)), ("model", estimator)])


def train_test_split_by_school(X, y, groups, strata, test_fold=0, random_state=42):
    """80/20 split: one fold of a 5-fold StratifiedGroupKFold is held out as the test set.

    Schools (groups) never appear on both sides; strata (country x label) keep
    the label rate and country shares the same in train and test.
    """
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(splitter.split(X, strata, groups)):
        if i == test_fold:
            return train_idx, test_idx
