"""Trains and evaluates the supervised theme classifier on a held-out
split. Source for docs/classifier_output.txt."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casemine import cases, classifier


def main():
    df = cases.generate_cases()
    t0 = time.time()
    result = classifier.train_and_evaluate(df)
    t1 = time.time()

    print("=== Supervised theme classifier ===")
    print(f"train: {result['n_train']}, test: {result['n_test']}")
    print(f"fit + predict + score: {t1 - t0:.2f}s")
    print(f"macro-F1: {result['macro_f1']:.4f}")
    print("per-theme F1:")
    for theme, f1 in sorted(result["per_theme_f1"].items()):
        print(f"  {theme}: {f1:.4f}")


if __name__ == "__main__":
    main()
