"""Runs the full pipeline end to end, renders the one-page brief, and
confirms the independent oracle finds 0 mismatches against freshly
recomputed figures. Writes docs/brief.md and source for
docs/brief_output.txt."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casemine import adoption, brief, brief_oracle, cases, classifier, topics


def main():
    root = Path(__file__).resolve().parents[1]
    docs = root / "docs"
    docs.mkdir(exist_ok=True)

    t0 = time.time()
    df = cases.generate_cases()
    X, vectorizer = topics.vectorize(df["case_text"])
    model, W, assigned = topics.fit_nmf(X)
    alignment = topics.oracle_alignment(assigned, df["true_theme"].values, cases.THEMES)
    clf_result = classifier.train_and_evaluate(df)
    panel = adoption.build_adoption_panel(df, cases.TOP3_RISK_THEMES)
    adoption_result = adoption.matched_control_effect(panel)
    t1 = time.time()

    figures = brief.compute_figures(df, alignment, clf_result, adoption_result)
    sentences = brief.render_brief(figures)
    text = brief.render_brief_text(figures)
    (docs / "brief.md").write_text(text)

    # the oracle recomputes the same figures from the same fitted results
    # (never from the rendered text) and must find 0 mismatches
    recomputed = brief.compute_figures(df, alignment, clf_result, adoption_result)
    mismatches = brief_oracle.check_brief(sentences, recomputed)

    print("=== One-page brief ===")
    print(text)
    print(f"pipeline wall time: {t1 - t0:.2f}s")
    print(f"oracle mismatches on the real brief: {len(mismatches)}")
    for m in mismatches:
        print(f"  MISMATCH: {m}")

    # deliberately corrupt one sentence's claimed value to prove the
    # oracle can actually catch a drift, not just pass trivially
    corrupted = [dict(s) for s in sentences]
    corrupted[2]["value"] = 0.99
    corrupted_mismatches = brief_oracle.check_brief(corrupted, recomputed)
    print(f"oracle mismatches on a deliberately corrupted brief: {len(corrupted_mismatches)}")
    for m in corrupted_mismatches:
        print(f"  CAUGHT: {m['figure']} claimed={m['claimed']} recomputed={m['recomputed']}")


if __name__ == "__main__":
    main()
