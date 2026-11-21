from casemine import classifier


def test_macro_f1_is_measured_and_reported(classifier_result):
    """Does not assert the claimed 0.86; records what was genuinely
    measured after 2 attempts (see README Findings)."""
    f1 = classifier_result["macro_f1"]
    assert 0.5 < f1 < 1.0
    print(f"measured macro-F1: {f1:.4f}")


def test_macro_f1_is_not_suspiciously_perfect(classifier_result):
    """A macro-F1 of 1.0 on free text would mean the simulated cases are
    trivially separable, not a genuine classification problem; see
    README Findings for the contamination fix that brought this down
    from an initial 1.0."""
    assert classifier_result["macro_f1"] < 0.99


def test_held_out_split_is_80_20(case_df, classifier_result):
    total = classifier_result["n_train"] + classifier_result["n_test"]
    assert total == len(case_df)
    assert abs(classifier_result["n_test"] / total - classifier.TEST_SIZE) < 0.01


def test_every_theme_has_a_reported_f1(classifier_result):
    from casemine import cases
    assert set(classifier_result["per_theme_f1"].keys()) == set(cases.THEMES)
