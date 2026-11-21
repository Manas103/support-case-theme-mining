"""Simulates a quarterly adoption-score panel for the 1,500 accounts
cases.py generates support cases for, with a designed causal effect: an
account that files a case in one of TOP3_RISK_THEMES during a quarter
sees its adoption score drop more over the *next* quarter than a matched
control account that did not. This module both designs that effect (the
data-generating process) and measures it with a nearest-neighbor matched
comparison, the same "design, then independently measure" discipline the
no-brainer analysis calls for; the measured effect is reported honestly
even though, being drawn from the same designed process, it should and
does land close to the design parameter below.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

N_QUARTERS = 8
TRUE_EFFECT_POINTS = -18.0  # the designed drop; see module docstring
NOISE_SD = 9.0


def _quarter_of(date: pd.Timestamp) -> int:
    return (date.year - 2024) * 4 + (date.month - 1) // 3


def build_adoption_panel(cases: pd.DataFrame, top3_risk_themes: list, seed: int = 20261102) -> pd.DataFrame:
    """Returns one row per (account_id, quarter): adoption_score,
    hit_top3_theme (whether this account filed a case in one of
    top3_risk_themes during this quarter), and account-level baseline
    fields used for matching."""
    rng = np.random.default_rng(seed)
    account_ids = np.sort(cases["account_id"].unique())
    n_accounts = len(account_ids)
    quarters = list(range(N_QUARTERS))

    baseline_adoption = rng.normal(60.0, 12.0, size=n_accounts).clip(10, 95)
    account_baseline = dict(zip(account_ids, baseline_adoption))

    cases = cases.copy()
    cases["quarter"] = cases["case_date"].map(_quarter_of)

    con = duckdb.connect(":memory:")
    con.register("cases", cases)
    con.register("risk_themes", pd.DataFrame({"theme": top3_risk_themes}))
    hit = con.execute(
        """
        SELECT account_id, quarter, COUNT(*) AS n_risk_cases
        FROM cases
        WHERE true_theme IN (SELECT theme FROM risk_themes)
        GROUP BY account_id, quarter
        """
    ).df()
    con.close()
    hit_set = set(zip(hit["account_id"], hit["quarter"]))

    rows = []
    adoption = {a: account_baseline[a] for a in account_ids}
    for q in quarters:
        for a in account_ids:
            hit_this_q = (a, q) in hit_set
            # drift + noise every quarter, plus the designed effect applied
            # to the *next* quarter after a risk-theme hit (q-1)
            drift = rng.normal(0.3, 2.0)
            effect = 0.0
            if q > 0 and (a, q - 1) in hit_set:
                effect = TRUE_EFFECT_POINTS + rng.normal(0.0, NOISE_SD)
            adoption[a] = float(np.clip(adoption[a] + drift + effect, 0, 100))
            rows.append({
                "account_id": a, "quarter": q,
                "adoption_score": adoption[a],
                "hit_top3_theme": int(hit_this_q),
                "baseline_adoption": account_baseline[a],
            })

    return pd.DataFrame(rows)


def matched_control_effect(panel: pd.DataFrame) -> dict:
    """For every account's *first* quarter with hit_top3_theme == 1 (a
    cohort design: an account's second and later hits are excluded so a
    repeated-hit account's already-depressed adoption score cannot
    compound into a second match), finds the nearest-neighbor control
    account (by pre-quarter adoption_score, among accounts with
    hit_top3_theme == 0 that same quarter) and compares each treated
    account's adoption change into the next quarter against its matched
    control's. Returns the mean treated-minus-control delta (the "lost N
    adoption points" number) plus the component means and sample size.
    See README Findings for why the first version of this function (every
    hit, not just the first) measured a materially smaller effect.
    """
    panel = panel.sort_values(["account_id", "quarter"]).reset_index(drop=True)
    panel["next_adoption"] = panel.groupby("account_id")["adoption_score"].shift(-1)
    panel["adoption_change"] = panel["next_adoption"] - panel["adoption_score"]

    scored = panel.dropna(subset=["next_adoption"]).copy()
    hit_rows = scored.loc[scored["hit_top3_theme"] == 1]
    first_hit_quarter = hit_rows.groupby("account_id")["quarter"].min()
    scored["is_first_hit"] = (
        (scored["hit_top3_theme"] == 1)
        & (scored["quarter"] == scored["account_id"].map(first_hit_quarter))
    )

    treated_deltas = []
    control_deltas = []
    pairs = []
    for q, group in scored.groupby("quarter"):
        treated = group.loc[group["is_first_hit"]]
        controls = group.loc[group["hit_top3_theme"] == 0]
        if treated.empty or controls.empty:
            continue
        control_scores = controls["adoption_score"].values
        control_deltas_arr = controls["adoption_change"].values
        control_ids = controls["account_id"].values
        for _, t_row in treated.iterrows():
            idx = int(np.argmin(np.abs(control_scores - t_row["adoption_score"])))
            treated_deltas.append(t_row["adoption_change"])
            control_deltas.append(control_deltas_arr[idx])
            pairs.append((t_row["account_id"], control_ids[idx], q))

    treated_deltas = np.array(treated_deltas)
    control_deltas = np.array(control_deltas)
    diffs = treated_deltas - control_deltas
    return {
        "n_pairs": len(pairs),
        "mean_treated_adoption_change": float(treated_deltas.mean()),
        "mean_control_adoption_change": float(control_deltas.mean()),
        "mean_matched_difference": float(diffs.mean()),
        "std_matched_difference": float(diffs.std(ddof=1)),
        "pairs": pairs,
    }
