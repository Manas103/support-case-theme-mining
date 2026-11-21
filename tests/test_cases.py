from casemine import cases


def test_generates_40000_cases_across_12_themes(case_df):
    assert len(case_df) == cases.N_CASES == 40_000
    assert set(case_df["true_theme"].unique()) == set(cases.THEMES)
    assert len(cases.THEMES) == 12


def test_theme_distribution_is_roughly_balanced(case_df):
    counts = case_df["true_theme"].value_counts()
    assert counts.min() / counts.max() > 0.9


def test_accounts_within_declared_range(case_df):
    assert case_df["account_id"].min() >= 1
    assert case_df["account_id"].max() <= cases.N_ACCOUNTS


def test_deterministic_given_same_seed():
    a = cases.generate_cases(seed=1)
    b = cases.generate_cases(seed=1)
    assert a["case_text"].equals(b["case_text"])


def test_different_seeds_produce_different_text():
    a = cases.generate_cases(seed=1)
    b = cases.generate_cases(seed=2)
    assert not a["case_text"].equals(b["case_text"])


def test_top3_risk_themes_are_real_themes():
    assert set(cases.TOP3_RISK_THEMES).issubset(set(cases.THEMES))
    assert len(cases.TOP3_RISK_THEMES) == 3
