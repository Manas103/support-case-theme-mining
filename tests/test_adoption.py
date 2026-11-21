from casemine import adoption, cases


def test_panel_covers_all_accounts_and_8_quarters(case_df, adoption_panel):
    assert adoption_panel["account_id"].nunique() == case_df["account_id"].nunique()
    assert adoption_panel["quarter"].nunique() == adoption.N_QUARTERS == 8


def test_adoption_score_stays_in_bounds(adoption_panel):
    assert adoption_panel["adoption_score"].between(0, 100).all()


def test_matched_control_difference_is_negative_and_measured(adoption_result):
    """Does not assert the claimed -18; records what was genuinely
    measured after 2 attempts (see README Findings: the first version
    measured a materially smaller effect due to repeated-hit floor
    clipping)."""
    diff = adoption_result["mean_matched_difference"]
    assert diff < 0
    print(f"measured mean matched difference: {diff:.3f}")


def test_matched_pairs_cover_most_accounts(case_df, adoption_result):
    assert adoption_result["n_pairs"] > 0.8 * case_df["account_id"].nunique()


def test_control_group_shows_no_designed_effect(adoption_result):
    """The matched controls, by construction, did not hit a top-3 risk
    theme the exposure quarter, so their mean adoption change should be
    close to the undisturbed drift (~0.3), not the -18 designed effect."""
    assert abs(adoption_result["mean_control_adoption_change"]) < 3.0


def test_recomputing_with_every_hit_not_just_first_measures_a_smaller_effect(case_df):
    """Pins the Findings regression: matching on every hit (not just an
    account's first) measures a materially smaller, more attenuated
    effect because repeated hits compound into the 0 floor."""
    panel = adoption.build_adoption_panel(case_df, cases.TOP3_RISK_THEMES)
    first_hit_result = adoption.matched_control_effect(panel)

    panel2 = panel.sort_values(["account_id", "quarter"]).reset_index(drop=True)
    panel2["next_adoption"] = panel2.groupby("account_id")["adoption_score"].shift(-1)
    panel2["adoption_change"] = panel2["next_adoption"] - panel2["adoption_score"]
    scored = panel2.dropna(subset=["next_adoption"])
    every_hit_diffs = []
    for q, group in scored.groupby("quarter"):
        treated = group.loc[group["hit_top3_theme"] == 1]
        controls = group.loc[group["hit_top3_theme"] == 0]
        if treated.empty or controls.empty:
            continue
        control_scores = controls["adoption_score"].values
        control_deltas = controls["adoption_change"].values
        for _, t_row in treated.iterrows():
            idx = int((abs(control_scores - t_row["adoption_score"])).argmin())
            every_hit_diffs.append(t_row["adoption_change"] - control_deltas[idx])

    import numpy as np
    every_hit_mean = float(np.mean(every_hit_diffs))
    assert every_hit_mean > first_hit_result["mean_matched_difference"], (
        "the every-hit measurement should be materially attenuated toward 0 "
        "relative to the first-hit cohort design, per README Findings"
    )
