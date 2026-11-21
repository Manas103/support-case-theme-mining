import numpy as np
from scipy.optimize import linear_sum_assignment

from casemine import cases, topics


def test_nmf_produces_exactly_12_components(nmf_fit):
    assert nmf_fit["model"].components_.shape[0] == topics.N_THEMES == 12


def test_every_case_assigned_to_a_component(case_df, nmf_fit):
    assert len(nmf_fit["assigned"]) == len(case_df)
    assert nmf_fit["assigned"].min() >= 0
    assert nmf_fit["assigned"].max() < 12


def test_oracle_alignment_purity_is_well_above_chance(nmf_alignment):
    # chance-level purity for a 12-way, roughly-balanced problem is ~1/12
    assert nmf_alignment["purity"] > 4.0 / 12.0


def test_oracle_alignment_is_a_valid_one_to_one_mapping(nmf_alignment):
    themes_assigned = list(nmf_alignment["component_to_theme"].values())
    assert len(themes_assigned) == 12
    assert len(set(themes_assigned)) == 12


def test_oracle_alignment_matches_independent_reimplementation(case_df, nmf_fit, nmf_alignment):
    """Reimplements the Hungarian-algorithm alignment from scratch (not
    calling topics.oracle_alignment) and asserts it reaches the same
    purity, the reference-oracle-for-the-oracle check."""
    assigned = nmf_fit["assigned"]
    true_theme = case_df["true_theme"].values
    theme_idx = {t: i for i, t in enumerate(cases.THEMES)}
    contingency = np.zeros((12, 12), dtype=int)
    for comp, theme in zip(assigned, true_theme):
        contingency[comp, theme_idx[theme]] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency)
    purity = contingency[row_ind, col_ind].sum() / contingency.sum()
    assert abs(purity - nmf_alignment["purity"]) < 1e-9
