"""TF-IDF vectorizes the 40,000 case texts and decomposes them into 12
themes with NMF (Non-negative Matrix Factorization), an unsupervised
method appropriate for sparse, non-negative TF-IDF features. Each case is
assigned its dominant NMF component (argmax of the document-topic matrix).

NMF's component indices are arbitrary (component 7 might correspond to
"billing_invoice_dispute" on one run and something else on the next), so
oracle_alignment below solves the linear-sum-assignment problem between
NMF components and the 12 true themes (never used to fit NMF itself, only
to score it afterward) to measure how well the unsupervised decomposition
actually recovered the generating structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

N_THEMES = 12
RANDOM_STATE = 0


def vectorize(texts: pd.Series) -> tuple:
    vectorizer = TfidfVectorizer(max_df=0.6, min_df=5, stop_words="english")
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def fit_nmf(X, n_components: int = N_THEMES) -> tuple:
    model = NMF(n_components=n_components, init="nndsvda", random_state=RANDOM_STATE, max_iter=400)
    W = model.fit_transform(X)
    assigned_component = np.argmax(W, axis=1)
    return model, W, assigned_component


def top_terms_per_component(model: NMF, vectorizer: TfidfVectorizer, n_terms: int = 8) -> dict:
    terms = np.array(vectorizer.get_feature_names_out())
    out = {}
    for c in range(model.components_.shape[0]):
        top_idx = np.argsort(model.components_[c])[::-1][:n_terms]
        out[c] = list(terms[top_idx])
    return out


def oracle_alignment(assigned_component: np.ndarray, true_theme: np.ndarray, themes: list) -> dict:
    """Solves the assignment problem (maximize total overlap) between NMF
    component indices and true theme labels via a contingency matrix and
    scipy's linear_sum_assignment (Hungarian algorithm), the component of
    this pipeline analogous to an independent reference oracle: it is
    never consulted while fitting NMF, only afterward to score it.
    Returns {component_to_theme, purity, contingency}.
    """
    n_components = int(assigned_component.max()) + 1
    contingency = np.zeros((n_components, len(themes)), dtype=int)
    theme_idx = {t: i for i, t in enumerate(themes)}
    for comp, theme in zip(assigned_component, true_theme):
        contingency[comp, theme_idx[theme]] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)  # maximize overlap
    component_to_theme = {int(r): themes[c] for r, c in zip(row_ind, col_ind)}
    matched_total = contingency[row_ind, col_ind].sum()
    purity = matched_total / contingency.sum()
    return {
        "component_to_theme": component_to_theme,
        "purity": float(purity),
        "contingency": contingency,
    }
