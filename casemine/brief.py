"""Builds the one-page brief: a short list of sentences, each tagged with
exactly one named figure and the value it states. Keeping sentence and
figure one-to-one (rather than free prose an NLP parser would have to
reverse-engineer) is what lets brief_oracle.py recompute every figure
independently and flag any sentence whose claimed value has drifted from
what the underlying data now supports, the same role the AI-commentary
checker plays elsewhere in this portfolio for a drafted report sentence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import cases as cases_mod


def compute_figures(
    cases_df: pd.DataFrame,
    nmf_alignment: dict,
    classifier_result: dict,
    adoption_result: dict,
) -> dict:
    """Computes every figure the brief states, from raw inputs only (the
    fitted NMF alignment, the classifier's held-out result, and the
    matched-control adoption result). This is the single source both
    render_brief and brief_oracle.recompute_figures call, so there are not
    two independently-maintained copies of the same arithmetic to drift
    apart; what the oracle re-derives independently is the fitted
    models/results themselves (see scripts/run_brief_benchmark.py and
    test_brief_oracle.py), not this arithmetic layer.
    """
    return {
        "n_cases": int(len(cases_df)),
        "n_accounts": int(cases_df["account_id"].nunique()),
        "n_themes": int(len(cases_mod.THEMES)),
        "nmf_purity": float(nmf_alignment["purity"]),
        "macro_f1": float(classifier_result["macro_f1"]),
        "n_test": int(classifier_result["n_test"]),
        "lost_points": float(-adoption_result["mean_matched_difference"]),
        "n_matched_pairs": int(adoption_result["n_pairs"]),
    }


def render_brief(figures: dict) -> list[dict]:
    """Returns a list of {"sentence", "figure", "value"} dicts: the brief,
    sentence by sentence, each naming the one figure it claims."""
    top3 = ", ".join(t.replace("_", " ") for t in cases_mod.TOP3_RISK_THEMES)
    return [
        {
            "sentence": f"This brief covers {figures['n_cases']:,} simulated support cases "
                        f"from {figures['n_accounts']:,} accounts.",
            "figure": "n_cases",
            "value": figures["n_cases"],
        },
        {
            "sentence": f"TF-IDF plus NMF recovered {figures['n_themes']} themes, matching the "
                        f"generating theme at {100 * figures['nmf_purity']:.1f}% purity against an "
                        f"independent Hungarian-algorithm alignment.",
            "figure": "nmf_purity",
            "value": figures["nmf_purity"],
        },
        {
            "sentence": f"A supervised theme classifier reached {figures['macro_f1']:.2f} macro-F1 "
                        f"on a held-out split of {figures['n_test']:,} cases.",
            "figure": "macro_f1",
            "value": figures["macro_f1"],
        },
        {
            "sentence": f"Accounts that filed a case in one of the top 3 highest-risk themes "
                        f"({top3}) lost {figures['lost_points']:.1f} adoption points more than "
                        f"matched controls over the next quarter, estimated from "
                        f"{figures['n_matched_pairs']:,} nearest-neighbor matched pairs.",
            "figure": "lost_points",
            "value": figures["lost_points"],
        },
    ]


def render_brief_text(figures: dict) -> str:
    lines = ["# Support-Case Theme Mining: One-Page Brief", ""]
    for entry in render_brief(figures):
        lines.append(f"- {entry['sentence']}")
    return "\n".join(lines) + "\n"
