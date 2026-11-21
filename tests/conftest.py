import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casemine import adoption, cases, classifier, topics


@pytest.fixture(scope="session")
def case_df():
    return cases.generate_cases()


@pytest.fixture(scope="session")
def nmf_fit(case_df):
    X, vectorizer = topics.vectorize(case_df["case_text"])
    model, W, assigned = topics.fit_nmf(X)
    return {"X": X, "vectorizer": vectorizer, "model": model, "W": W, "assigned": assigned}


@pytest.fixture(scope="session")
def nmf_alignment(case_df, nmf_fit):
    return topics.oracle_alignment(nmf_fit["assigned"], case_df["true_theme"].values, cases.THEMES)


@pytest.fixture(scope="session")
def classifier_result(case_df):
    return classifier.train_and_evaluate(case_df)


@pytest.fixture(scope="session")
def adoption_panel(case_df):
    return adoption.build_adoption_panel(case_df, cases.TOP3_RISK_THEMES)


@pytest.fixture(scope="session")
def adoption_result(adoption_panel):
    return adoption.matched_control_effect(adoption_panel)
