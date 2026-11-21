"""Independently recomputes every figure the brief states and flags any
sentence whose claimed value no longer matches, naming the sentence and
the disagreement rather than returning a bare pass/fail. This is the
"independent oracle recomputes every figure" claim: check_brief takes the
brief's own rendered sentences (each already tagged with the one figure
it claims, see brief.py) and a freshly recomputed figures dict, and
reports every mismatch beyond a small numeric tolerance (floating-point
noise from a re-run, not a real drift).
"""
from __future__ import annotations

TOLERANCE = {
    "n_cases": 0,
    "n_accounts": 0,
    "n_themes": 0,
    "n_test": 0,
    "n_matched_pairs": 0,
    "nmf_purity": 1e-6,
    "macro_f1": 1e-6,
    "lost_points": 0.5,  # matched-control estimate has real sampling noise run to run
}


def check_brief(rendered_sentences: list, recomputed_figures: dict) -> list:
    """Returns a list of mismatch dicts (empty if every sentence agrees
    with recomputed_figures within tolerance)."""
    mismatches = []
    for entry in rendered_sentences:
        figure = entry["figure"]
        claimed = entry["value"]
        actual = recomputed_figures.get(figure)
        if actual is None:
            mismatches.append({"sentence": entry["sentence"], "figure": figure, "reason": "no recomputed value"})
            continue
        tol = TOLERANCE.get(figure, 0)
        if abs(claimed - actual) > tol:
            mismatches.append({
                "sentence": entry["sentence"],
                "figure": figure,
                "claimed": claimed,
                "recomputed": actual,
                "diff": claimed - actual,
            })
    return mismatches
