"""Data loading, missing-value cleaning and label construction for PASEC2019 Grade 6.

Every function here is deterministic and never modifies the raw file.
Sources:
- PASEC2019 data manual (CONFEMEN), Table 5.17, p. 66, for the proficiency cutoffs.
- AFLEARN PASEC2019_Grade6_EN_labels.R for the value-label sets and sentinel codes.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
RAW_ZIP = SRC_DIR.parent / "data" / "raw" / "PASEC2019_INT_STATA.zip"
RAW_MEMBER = "PASEC2019_GRADE6_INT.dta"

# PASEC2019 manual, Table 5.17 (end of primary, mathematics):
# 433.28 < PV <= 521.46 is the band just below the PASEC minimum proficiency level.
PASEC_MIN_MATHS_G6 = 521.46
N_PLAUSIBLE_VALUES = 5
MATHS_PV = [f"MATHS_PV{i}" for i in range(1, N_PLAUSIBLE_VALUES + 1)]
READING_PV = [f"LECT_PV{i}" for i in range(1, N_PLAUSIBLE_VALUES + 1)]

# Columns that must never enter the feature matrix.
TEST_ITEMS = [f"f{i}" for i in range(1, 99)] + [f"m{i}" for i in range(1, 85)]
DESIGN_WEIGHTS = [f"rwgt{i}" for i in range(0, 91)] + ["W_ECOLE", "JKZONE", "JKREP"]
IDENTIFIERS = ["ID_TOT", "ID_ECOLE", "ID_ELEVE", "ID_STRATE", "ID_ENSEIGNANT", "PAYS"]
LEAKAGE_COLUMNS = TEST_ITEMS + MATHS_PV + READING_PV

# Sentinel codes per AFLEARN value-label set (PASEC2019_Grade6_EN_labels.R).
AFLEARN_SENTINEL_SETS = {
    "agree": {6: 'Invalid', 9: 'Missing'},
    "freq": {6: 'Invalid', 7: 'Not applicable', 9: 'Missing'},
    "freq2": {6: 'Invalid', 9: 'Missing'},
    "gender": {6: 'Invalid', 9: 'Missing'},
    "goodbad": {6: 'Invalid', 9: 'Missing'},
    "missing2": {96: 'Invalid', 97: 'Not applicable', 99: 'Missing'},
    "missing3": {996: 'Invalid', 997: 'Not applicable', 999: 'Missing'},
    "missing4": {9996: 'Invalid', 9997: 'Not applicable', 9999: 'Missing'},
    "qd17": {6: 'Invalid', 9: 'Missing'},
    "qd28": {96: 'Invalid', 99: 'Missing'},
    "qd31": {6: 'Invalid', 9: 'Missing'},
    "qd34": {6: 'Invalid', 7: 'Not applicable', 9: 'Missing'},
    "qd43a": {6: 'Invalid', 7: 'Not applicable', 9: 'Missing'},
    "qd45a": {6: 'Invalid', 7: 'Not applicable', 9: 'Missing'},
    "qd59": {6: 'Invalid', 7: 'Not applicable', 9: 'Missing'},
    "qd67": {6: 'Invalid', 7: 'Not applicable', 9: 'Missing'},
    "qd68": {6: 'Invalid', 7: 'Not applicable', 9: 'Missing'},
    "qd84": {6: 'Invalid', 7: 'Not applicable', 9: 'Missing'},
    "qe615": {6: 'Invalid', 7: 'Not applicable', 9: 'Missing'},
    "qe632": {6: 'Invalid', 7: 'Not applicable', 9: 'Missing'},
    "qe634": {6: 'Invalid', 7: 'Not applicable', 9: 'Missing'},
    "qe639": {96: 'Invalid', 99: 'Missing'},
    "qm13": {96: 'Invalid', 99: 'Missing'},
    "qm17": {6: 'Invalid', 9: 'Missing'},
    "qm19": {6: 'Invalid', 9: 'Missing'},
    "qm25a": {96: 'Invalid', 97: 'Not applicable', 99: 'Missing'},
    "qm27": {6: 'Invalid', 9: 'Missing'},
    "qm30": {6: 'Invalid', 7: 'Not applicable', 9: 'Missing'},
    "qm33": {6: 'Invalid', 7: 'Not applicable', 9: 'Missing'},
    "qm38": {6: 'Invalid', 9: 'Missing'},
    "qm40": {96: 'Invalid', 99: 'Missing'},
    "qm5": {96: 'Invalid', 99: 'Missing'},
    "yesno": {6: 'Invalid', 7: 'Not applicable', 9: 'Missing'},
}

# Variables attached to each set in the AFLEARN file (f1-f98 and m1-m84 use "item").
AFLEARN_SET_VARIABLES = {
    "agree": [
        "qe640", "qe641", "qe642", "qe643",
    ],
    "freq": [
        "qe612", "qe613", "qe616", "qe617", "qe618", "qe619", "qe624",
    ],
    "freq2": [
        "qe620",
    ],
    "gender": [
        "qe63", "qd2",
    ],
    "goodbad": [
        "qm46a", "qm46b", "qm46c", "qm46d", "qm46e", "qm46f", "qm46g", "qm46h",
        "qm46i", "qm46j", "qm46k", "qm46l", "qm46m", "qm46n", "qm46o", "qm46p",
    ],
    "missing2": [
        "qe61d", "qe61m", "qe62", "qd1", "qd3", "qd4", "qd6", "qd8",
        "qd10", "qd20", "qd22", "qd24", "qd25", "qd26d", "qd26m", "qd27d",
        "qd27m", "qd30", "qd36", "qd41a", "qd41b", "qd41c", "qd69a", "qd69b",
        "qd76", "qd77a", "qd77b", "qd77c", "qd80a", "qd80b", "qd80c", "qd80d",
        "qm2", "qm6", "qm8", "qm10", "qm11cia", "qm11cpa", "qm11ce1a", "qm11ce2a",
        "qm11cm1a", "qm11cm2a", "qm12", "qm14",
    ],
    "missing3": [
        "qm21", "qm22", "qm22a", "qm22b", "qm23a", "qm23b", "qm29", "qm36",
    ],
    "missing4": [
        "qe61y",
    ],
    "qd17": [
        "qd17",
    ],
    "qd28": [
        "qd28",
    ],
    "qd31": [
        "qd31",
    ],
    "qd34": [
        "qd34",
    ],
    "qd43a": [
        "qd43a", "qd43b", "qd43c", "qd43d", "qd43e", "qd44a", "qd44b", "qd44c",
        "qd44d",
    ],
    "qd45a": [
        "qd45a", "qd45b", "qd45c", "qd45d", "qd45e",
    ],
    "qd59": [
        "qd59", "qd61",
    ],
    "qd67": [
        "qd67",
    ],
    "qd68": [
        "qd68",
    ],
    "qd84": [
        "qd84",
    ],
    "qe615": [
        "qe615",
    ],
    "qe632": [
        "qe632",
    ],
    "qe634": [
        "qe634",
    ],
    "qe639": [
        "qe639",
    ],
    "qm13": [
        "qm13",
    ],
    "qm17": [
        "qm17",
    ],
    "qm19": [
        "qm19",
    ],
    "qm25a": [
        "qm25a", "qm25b", "qm25c", "qm25d", "qm25e", "qm25f",
    ],
    "qm27": [
        "qm27",
    ],
    "qm30": [
        "qm30", "qm31",
    ],
    "qm33": [
        "qm33", "qm35",
    ],
    "qm38": [
        "qm38",
    ],
    "qm40": [
        "qm40",
    ],
    "qm5": [
        "qm5",
    ],
    "yesno": [
        "qe64a", "qe64b", "qe64c", "qe64d", "qe64e", "qe64f", "qe64g", "qe65",
        "qe66", "qe67a", "qe67b", "qe67c", "qe68", "qe69a", "qe69b", "qe69c",
        "qe69d", "qe69e", "qe69f", "qe610", "qe611a", "qe611b", "qe614", "qe621a",
        "qe621b", "qe621c", "qe621d", "qe621e", "qe621f", "qe621g", "qe622", "qe623",
        "qe625a", "qe625b", "qe625c", "qe625d", "qe625e", "qe625f", "qe625g", "qe626",
        "qe627", "qe628", "qe629", "qe630", "qe631", "qe633", "qe635a", "qe635b",
        "qe635c", "qe635d", "qe635e", "qe635f", "qe635g", "qe635h", "qe635i", "qe635j",
        "qe635k", "qe636a", "qe636b", "qe636c", "qe636d", "qe636e", "qe636f", "qe636g",
        "qe636h", "qe636i", "qe637a", "qe637b", "qe637c", "qe637d", "qe637e", "qe637f",
        "qe637g", "qe638", "qd5", "qd7", "qd9", "qd13", "qd14", "qd15a",
        "qd15b", "qd15c", "qd16a", "qd16b", "qd16c", "qd16d", "qd16e", "qd18",
        "qd19", "qd21", "qd23", "qd29", "qd32a", "qd32b", "qd32c", "qd32d",
        "qd32e", "qd32f", "qd32g", "qd32h", "qd32i", "qd32j", "qd32k", "qd33",
        "qd35", "qd37a", "qd37b", "qd37c", "qd37d", "qd37e", "qd37f", "qd37g",
        "qd37h", "qd38", "qd39", "qd40a", "qd40b", "qd40c", "qd40d", "qd42",
        "qd46a", "qd46b", "qd46c", "qd46d", "qd46e", "qd46f", "qd47a", "qd47b",
        "qd47c", "qd47d", "qd47e", "qd47f", "qd47g", "qd47h", "qd48a", "qd48b",
        "qd48c", "qd48d", "qd48e", "qd48f", "qd48g", "qd48h", "qd48i", "qd48j",
        "qd48k", "qd48l", "qd48m", "qd48n", "qd49a", "qd49b", "qd49c", "qd49d",
        "qd49e", "qd49f", "qd49g", "qd49h", "qd49i", "qd49j", "qd49k", "qd49l",
        "qd49m", "qd49n", "qd50a", "qd50b", "qd50c", "qd50d", "qd50e", "qd50f",
        "qd50g", "qd50h", "qd50i", "qd50j", "qd50k", "qd50l", "qd51", "qd52",
        "qd53", "qd54", "qd55", "qd56", "qd57", "qd58", "qd60", "qd62",
        "qd63", "qd64", "qd65", "qd66", "qd70", "qd71", "qd72", "qd74",
        "qd78a", "qd78b", "qd78c", "qd78d", "qd78e", "qd78f", "qd78g", "qd78h",
        "qd78i", "qd78j", "qd78k", "qd78l", "qd78m", "qd78n", "qd78o", "qd78p",
        "qd78q", "qd78r", "qd78s", "qd78t", "qd79a", "qd79b", "qd81", "qd82",
        "qd83", "qd85", "qd86a", "qd86b", "classedp", "classefp", "qm7", "qm9a",
        "qm9b", "qm9c", "qm9d", "qm9e", "qm11ci", "qm11cp", "qm11ce1", "qm11ce2",
        "qm11cm1", "qm11cm2", "qm15a", "qm15b", "qm15c", "qm15d", "qm15e", "qm15f",
        "qm15g", "qm15h", "qm15i", "qm15j", "qm15k", "qm15l", "qm15m", "qm15n",
        "qm16", "qm18a", "qm18b", "qm18c", "qm18d", "qm18e", "qm20ci", "qm20cp",
        "qm20ce1", "qm20ce2", "qm20cm1", "qm20cm2", "qm24", "qm26a", "qm26b", "qm26c",
        "qm26d", "qm26e", "qm26f", "qm26g", "qm26h", "qm26i", "qm26j", "qm26k",
        "qm26l", "qm26m", "qm26n", "qm26o", "qm26p", "qm26q", "qm26r", "qm28",
        "qm32", "qm34", "qm37a", "qm37b", "qm37c", "qm37d", "qm37e", "qm37f",
        "qm39a", "qm39b", "qm39c", "qm39d", "qm39e", "qm39f", "qm41", "qm42a",
        "qm42b", "qm42c", "qm43", "qm44a", "qm44b", "qm44c", "qm44d", "qm44e",
        "qm44f", "qm44g", "qm45a", "qm45b",
    ],
}

AFLEARN_SET_VARIABLES["item"] = TEST_ITEMS
AFLEARN_SENTINEL_SETS["item"] = {6: "Invalid", 7: "Missing by design", 8: "Not reached", 9: "Missing"}

# Codes the AFLEARN file does not cover, checked against the PASEC manual and the .dta.
# "manual": listed in the PASEC2019 manual. "inferred": present in the file but not
# listed for that variable; none of these variables is a model feature.
EXTRA_SENTINELS = {
    "qm1": ({6: "Invalid", 9: "Missing"}, "manual"),
    "qm4": ({99: "Missing"}, "manual"),
    "qd12": ({99: "Missing"}, "manual"),
    "qd75a": ({9999999996: "Invalid", 9999999997: "Not applicable", 9999999999: "Missing"}, "manual"),
    "qd75b": ({9999999996: "Invalid", 9999999997: "Not applicable", 9999999999: "Missing"}, "manual"),
    "qd75c": ({9999999996: "Invalid", 9999999997: "Not applicable", 9999999999: "Missing"}, "manual"),
    "qd11": ({96: "Invalid", 99: "Missing"}, "inferred"),
    "qm3": ({96: "Invalid", 99: "Missing"}, "inferred"),
    "qm4_96": ({96: "Invalid"}, "inferred"),
    "qd12_96": ({96: "Invalid"}, "inferred"),
    "se8": ({9999: "Missing"}, "inferred"),
    "qd73": ({9999999996: "Invalid", 9999999997: "Not applicable", 9999999999: "Missing"}, "inferred"),
}


def sentinel_map() -> dict[str, dict[float, tuple[str, str]]]:
    """Per variable: {code: (meaning, source)}."""
    out: dict[str, dict[float, tuple[str, str]]] = {}
    for set_name, variables in AFLEARN_SET_VARIABLES.items():
        for var in variables:
            out.setdefault(var, {}).update(
                {float(c): (lab, "aflearn") for c, lab in AFLEARN_SENTINEL_SETS[set_name].items()})
    for key, (codes, source) in EXTRA_SENTINELS.items():
        var = key.split("_96")[0]
        out.setdefault(var, {}).update({float(c): (lab, source) for c, lab in codes.items()})
    return out


def load_raw_grade6(zip_path: Path | str = RAW_ZIP) -> pd.DataFrame:
    """Read the Grade 6 .dta straight from the original zip (read-only).

    Numeric codes are kept (convert_categoricals=False) so sentinel codes
    are handled explicitly instead of through French text labels.
    """
    with zipfile.ZipFile(zip_path) as zf, zf.open(RAW_MEMBER) as fh:
        return pd.read_stata(fh, convert_categoricals=False)


def replace_sentinels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn variable-specific sentinel codes into NaN.

    Codes are mapped per variable because the same number means different
    things across questions (6 is "Invalid" for yes/no items but "River"
    in qe639; 9 is a real diploma code in qm4).

    Returns the cleaned frame and an audit table: per column, the cells that
    were blank in the file and the cells holding each type of sentinel code.
    """
    out = df.copy()
    smap = sentinel_map()
    rows = []
    for col in df.columns:
        values = df[col]
        if not pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_datetime64_any_dtype(values):
            # Text columns (IDs, date parts) store missing as an empty string.
            blank = values.astype("string").str.strip().eq("").fillna(False)
            out.loc[blank, col] = np.nan
            values = pd.to_numeric(values.where(~blank), errors="coerce")
        else:
            blank = pd.Series(False, index=df.index)
        counts = {"blank_in_file": int(df[col].isna().sum() + blank.sum()), "inferred_code_cells": 0}
        for code, (label, source) in smap.get(col, {}).items():
            mask = (values == code).fillna(False)
            n = int(mask.sum())
            counts[label] = counts.get(label, 0) + n
            if source == "inferred":
                counts["inferred_code_cells"] += n
            out.loc[mask, col] = np.nan
        rows.append({"column": col, **counts})
    audit = pd.DataFrame(rows).set_index("column").fillna(0).astype(int)
    code_cols = [c for c in audit.columns if c != "inferred_code_cells"]
    audit["total_missing"] = audit[code_cols].sum(axis=1)
    audit["pct_missing"] = audit["total_missing"] / len(df) * 100
    return out, audit


def school_key(df: pd.DataFrame) -> pd.Series:
    """Unique school identifier. The manual (Table 2.2) defines ID_ECOLE as a
    school within a country, so codes repeat across countries (1,173 distinct
    codes, 3,298 schools). ID_TOT digits 1-8 (country + school) give the same key."""
    return (df["ID_PAYS"].astype(int).astype(str).str.zfill(2) + "_"
            + df["ID_ECOLE"].astype(str)).rename("school_key")


def make_label(df: pd.DataFrame, cutoff: float = PASEC_MIN_MATHS_G6,
               min_votes: int = 3) -> pd.Series:
    """Binary target: 1 = below the PASEC minimum in mathematics.

    A student is labelled 1 when at least `min_votes` of the five maths
    plausible values are at or below the cutoff (majority of 5).
    """
    votes = (df[MATHS_PV] <= cutoff).sum(axis=1)
    return (votes >= min_votes).astype("int8").rename("below_min_maths")


def label_per_plausible_value(df: pd.DataFrame,
                              cutoff: float = PASEC_MIN_MATHS_G6) -> pd.DataFrame:
    """Five alternative labels, one per plausible value, for the evaluation
    sensitivity check (metrics are computed against each and averaged)."""
    return (df[MATHS_PV] <= cutoff).astype("int8").set_axis(
        [f"below_min_pv{i}" for i in range(1, N_PLAUSIBLE_VALUES + 1)], axis=1)
