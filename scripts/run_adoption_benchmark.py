"""Builds the quarterly adoption panel and the matched-control estimate of
adoption loss for accounts hitting a top-3 risk theme. Source for
docs/adoption_output.txt."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casemine import adoption, cases


def main():
    df = cases.generate_cases()
    t0 = time.time()
    panel = adoption.build_adoption_panel(df, cases.TOP3_RISK_THEMES)
    t1 = time.time()
    result = adoption.matched_control_effect(panel)
    t2 = time.time()

    print("=== Matched-control adoption-loss estimate ===")
    print(f"accounts: {df['account_id'].nunique()}, quarters: {panel['quarter'].nunique()}")
    print(f"panel build: {t1 - t0:.2f}s, matched-control estimate: {t2 - t1:.2f}s")
    print(f"matched pairs (first risk-theme exposure cohort): {result['n_pairs']}")
    print(f"mean treated adoption change (quarter of exposure -> next quarter): {result['mean_treated_adoption_change']:.3f}")
    print(f"mean matched-control adoption change: {result['mean_control_adoption_change']:.3f}")
    print(f"mean matched difference (lost points): {result['mean_matched_difference']:.3f} "
          f"(std {result['std_matched_difference']:.3f})")


if __name__ == "__main__":
    main()
