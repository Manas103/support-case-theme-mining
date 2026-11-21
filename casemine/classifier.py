"""A fast supervised theme classifier, trained to reproduce the 12-theme
taxonomy from case text alone, evaluated on a held-out split. In a real
deployment the training label would come from a human review pass over
the NMF clusters topics.py discovers; here the simulator's own
`true_theme` substitutes honestly for that review step (the classifier
never sees NMF's component assignments, only text in, theme out), the same
role a "_true" column plays elsewhere in this portfolio as a stand-in for
a labeling process this project does not itself model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 0
TEST_SIZE = 0.2


def train_and_evaluate(cases: pd.DataFrame) -> dict:
    train_df, test_df = train_test_split(
        cases, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=cases["true_theme"]
    )

    vectorizer = TfidfVectorizer(max_df=0.6, min_df=5, stop_words="english")
    X_train = vectorizer.fit_transform(train_df["case_text"])
    X_test = vectorizer.transform(test_df["case_text"])

    model = LogisticRegression(max_iter=1000, C=5.0, random_state=RANDOM_STATE)
    model.fit(X_train, train_df["true_theme"])

    y_pred = model.predict(X_test)
    macro_f1 = f1_score(test_df["true_theme"], y_pred, average="macro")
    report = classification_report(test_df["true_theme"], y_pred, output_dict=True)

    return {
        "macro_f1": float(macro_f1),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "per_theme_f1": {k: v["f1-score"] for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")},
        "model": model,
        "vectorizer": vectorizer,
        "test_df": test_df,
        "y_pred": y_pred,
    }
