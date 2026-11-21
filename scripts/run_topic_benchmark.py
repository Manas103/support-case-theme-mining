"""Generates the 40,000-case dataset, fits TF-IDF + NMF, and reports the
oracle alignment (Hungarian-algorithm match against the generating theme).
Source for docs/topic_output.txt."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casemine import cases, topics


def main():
    t0 = time.time()
    df = cases.generate_cases()
    t1 = time.time()
    X, vectorizer = topics.vectorize(df["case_text"])
    model, W, assigned = topics.fit_nmf(X)
    t2 = time.time()
    alignment = topics.oracle_alignment(assigned, df["true_theme"].values, cases.THEMES)
    t3 = time.time()

    print("=== TF-IDF + NMF theme discovery ===")
    print(f"cases: {len(df)}, vocabulary size: {X.shape[1]}")
    print(f"generation: {t1 - t0:.2f}s, NMF fit: {t2 - t1:.2f}s, oracle alignment: {t3 - t2:.2f}s")
    print(f"NMF components: {model.components_.shape[0]}")
    print(f"oracle alignment purity: {alignment['purity']:.4f}")
    print("component -> aligned theme:")
    for comp, theme in sorted(alignment["component_to_theme"].items()):
        print(f"  {comp}: {theme}")
    terms = topics.top_terms_per_component(model, vectorizer)
    print("\ntop terms per component:")
    for comp, words in terms.items():
        print(f"  {comp}: {', '.join(words)}")


if __name__ == "__main__":
    main()
