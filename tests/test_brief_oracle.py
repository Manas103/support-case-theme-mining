from casemine import brief, brief_oracle


def test_real_brief_has_zero_oracle_mismatches(case_df, nmf_alignment, classifier_result, adoption_result):
    figures = brief.compute_figures(case_df, nmf_alignment, classifier_result, adoption_result)
    sentences = brief.render_brief(figures)
    recomputed = brief.compute_figures(case_df, nmf_alignment, classifier_result, adoption_result)
    mismatches = brief_oracle.check_brief(sentences, recomputed)
    assert mismatches == []


def test_oracle_catches_a_deliberately_corrupted_figure(case_df, nmf_alignment, classifier_result, adoption_result):
    figures = brief.compute_figures(case_df, nmf_alignment, classifier_result, adoption_result)
    sentences = brief.render_brief(figures)
    corrupted = [dict(s) for s in sentences]
    corrupted[2]["value"] = 0.99  # the real macro_f1 is ~0.86
    recomputed = brief.compute_figures(case_df, nmf_alignment, classifier_result, adoption_result)
    mismatches = brief_oracle.check_brief(corrupted, recomputed)
    assert len(mismatches) == 1
    assert mismatches[0]["figure"] == "macro_f1"


def test_oracle_does_not_falsely_flag_a_clean_sentence_when_another_is_corrupted(
    case_df, nmf_alignment, classifier_result, adoption_result
):
    figures = brief.compute_figures(case_df, nmf_alignment, classifier_result, adoption_result)
    sentences = brief.render_brief(figures)
    corrupted = [dict(s) for s in sentences]
    corrupted[1]["value"] = 0.0  # corrupt nmf_purity only
    recomputed = brief.compute_figures(case_df, nmf_alignment, classifier_result, adoption_result)
    mismatches = brief_oracle.check_brief(corrupted, recomputed)
    flagged_figures = {m["figure"] for m in mismatches}
    assert flagged_figures == {"nmf_purity"}


def test_every_rendered_sentence_names_exactly_one_known_figure(case_df, nmf_alignment, classifier_result, adoption_result):
    figures = brief.compute_figures(case_df, nmf_alignment, classifier_result, adoption_result)
    sentences = brief.render_brief(figures)
    assert len(sentences) == 4
    for entry in sentences:
        assert entry["figure"] in figures
