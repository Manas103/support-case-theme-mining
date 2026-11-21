"""Simulates 40,000 free-text support cases, each secretly generated from
one of 12 true themes (see THEMES), assembled from a per-theme phrase bank
plus generic filler phrases shared across every theme (so the text is not
trivially separable on filler alone) and an account/product mention.
`true_theme` is carried only for scoring (topics.py's oracle alignment,
classifier.py's held-out F1, adoption.py's causal design); the TF-IDF/NMF
pipeline in topics.py never reads it. Nothing here is a real customer
complaint; every phrase is hand-written to be representative of a Cisco-
shaped install-base support queue (per the no-brainer's RMA/software/
install-base framing), not transcribed from any real ticket.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_CASES = 40_000
N_ACCOUNTS = 1_500
SIM_START = pd.Timestamp("2024-01-01")
SIM_DAYS = 730  # 2 years, 8 quarters

THEMES = [
    "login_authentication_failure",
    "billing_invoice_dispute",
    "feature_request",
    "performance_degradation",
    "integration_api_failure",
    "documentation_gap",
    "security_vulnerability_concern",
    "data_loss_corruption",
    "licensing_entitlement_confusion",
    "onboarding_setup_difficulty",
    "mobile_app_defect",
    "notification_alert_noise",
]

# The 3 themes this project's adoption analysis treats as high-risk
# (serious technical pain points plausible as adoption-damaging), used by
# adoption.py to design the causal effect it then measures. Declared here,
# next to THEMES, so topics.py and classifier.py can see the full list
# without this list being a secret the theme-mining pipeline depends on.
TOP3_RISK_THEMES = ["data_loss_corruption", "security_vulnerability_concern", "performance_degradation"]

PHRASE_BANK = {
    "login_authentication_failure": [
        "unable to log in after the latest password reset",
        "SSO redirect loops back to the login page",
        "multi-factor code never arrives by text or email",
        "account gets locked out after a single failed attempt",
        "session expires within a minute of signing in",
        "single sign-on certificate appears to have expired",
        "login page returns an invalid credentials error with the correct password",
    ],
    "billing_invoice_dispute": [
        "invoice amount does not match the quoted renewal price",
        "we were billed for seats that were removed last quarter",
        "a credit from last month never appeared on this statement",
        "purchase order number is missing from the invoice",
        "tax line item looks incorrect for our billing region",
        "duplicate charge appeared on this month's statement",
        "the proration on our mid-cycle upgrade looks wrong",
    ],
    "feature_request": [
        "it would help to export this report directly to a spreadsheet",
        "please consider adding a dark mode to the admin console",
        "a bulk edit option for tags would save our team real time",
        "could the dashboard support a custom date range",
        "we would like a webhook when a ticket changes status",
        "an audit log filter by user would be very useful",
        "requesting a way to schedule recurring reports",
    ],
    "performance_degradation": [
        "the dashboard has been noticeably slower since the last update",
        "search queries are timing out that used to return instantly",
        "bulk export now takes several times longer than it used to",
        "the console freezes for a few seconds when switching tabs",
        "API response times have climbed steadily over the past week",
        "report generation is now taking over ten minutes to finish",
        "the mobile sync job appears to be stalling under load",
    ],
    "integration_api_failure": [
        "the webhook endpoint is returning a 500 error on every call",
        "our nightly sync job fails with a rate-limit error every run",
        "the API key stopped authenticating after the latest rotation",
        "the Salesforce connector silently stopped updating records",
        "pagination in the list endpoint is skipping records",
        "the OAuth token refresh is failing intermittently",
        "the SDK's retry logic seems to be causing duplicate records",
    ],
    "documentation_gap": [
        "the setup guide references a settings page that no longer exists",
        "there is no documented way to configure the retry policy",
        "the API reference is missing the new webhook payload fields",
        "the migration guide skips a required database step",
        "code samples in the docs use a deprecated client library version",
        "the changelog does not mention this breaking change",
        "the permissions matrix in the docs is out of date",
    ],
    "security_vulnerability_concern": [
        "a security researcher flagged a potential XSS in the comment field",
        "we noticed an API token was logged in plaintext in a response",
        "a former employee's account was not deprovisioned automatically",
        "the audit log does not capture permission-level changes",
        "we need confirmation this release addresses the recent CVE",
        "a shared link exposed a report to users outside our tenant",
        "password reset emails use a predictable token format",
    ],
    "data_loss_corruption": [
        "several records disappeared after the overnight sync",
        "a bulk import silently overwrote existing customer data",
        "exported files are missing rows that are visible in the UI",
        "a rollback left the account in a partially migrated state",
        "duplicate merge accidentally deleted the older record's history",
        "an attachment upload corrupted the file on download",
        "a failed migration left two conflicting copies of the same record",
    ],
    "licensing_entitlement_confusion": [
        "we are not sure which seats still have the premium add-on",
        "the entitlement count in the admin panel does not match our contract",
        "a downgraded user can still access a premium-only report",
        "we cannot tell which license tier unlocks the new analytics module",
        "the trial expired but premium features are still showing as locked",
        "seat count shown in billing disagrees with the console's user list",
        "we need a breakdown of which modules our current plan includes",
    ],
    "onboarding_setup_difficulty": [
        "the initial setup wizard gets stuck on the domain verification step",
        "importing our existing user directory failed partway through",
        "we could not figure out how to assign the first admin role",
        "the welcome email never arrived for half of our new users",
        "the sample project does not match what the tutorial describes",
        "connecting our identity provider during setup kept timing out",
        "new team members are not seeing the onboarding checklist",
    ],
    "mobile_app_defect": [
        "the mobile app crashes when opening a report with charts",
        "push notifications stopped arriving after the last app update",
        "the offline mode does not sync changes back once reconnected",
        "the app's camera scanner fails to recognize the barcode",
        "switching accounts in the mobile app logs the user out entirely",
        "the mobile app shows a blank screen on older Android devices",
        "the tablet layout cuts off the save button on smaller screens",
    ],
    "notification_alert_noise": [
        "we are getting duplicate alerts for the same incident",
        "digest emails are arriving multiple times a day instead of once",
        "alert thresholds cannot be tuned per team, only account-wide",
        "we would like to mute notifications for a specific project",
        "the alert channel keeps paging us for already-resolved issues",
        "too many low-priority notifications are drowning out real alerts",
        "weekend notifications should be suppressed per our schedule",
    ],
}

FILLER_PHRASES = [
    "this has been going on for a few days now",
    "please let us know if you need more detail",
    "this is affecting several members of our team",
    "we would appreciate a quick update on this",
    "happy to hop on a call if that is easier",
    "this is becoming a priority on our end",
    "let me know what additional logs would help",
]

PRODUCTS = ["DNA-CENTER", "SECURE-ENDPOINT", "WEBEX-CALLING", "UMBRELLA", "ISE", "MERAKI-MX"]


def _rng_for(seed: int) -> np.random.default_rng:
    return np.random.default_rng(seed)


def generate_cases(seed: int = 20261101) -> pd.DataFrame:
    rng = _rng_for(seed)

    account_id = rng.integers(1, N_ACCOUNTS + 1, size=N_CASES)
    true_theme = rng.choice(THEMES, size=N_CASES)
    product = rng.choice(PRODUCTS, size=N_CASES)
    day_offset = rng.integers(0, SIM_DAYS, size=N_CASES)
    case_date = pd.to_datetime(SIM_START) + pd.to_timedelta(day_offset, unit="D")

    # Deliberate label-boundary noise: a minority of cases mix in a phrase
    # from a second, randomly chosen theme (a case that genuinely touches
    # two concerns, e.g. a performance complaint that also mentions a
    # mobile crash), and cases carry only 1-2 theme phrases rather than a
    # heavy majority, so the classifier is not solving a trivially
    # separable bag-of-words problem (see README Findings: the first
    # version used 2-3 theme phrases and 0 contamination, which measured
    # a suspicious 1.0 macro-F1).
    CONTAMINATION_RATE = 0.55

    texts = []
    for i in range(N_CASES):
        bank = PHRASE_BANK[true_theme[i]]
        n_theme_phrases = rng.integers(1, 3)
        theme_phrases = rng.choice(bank, size=min(n_theme_phrases, len(bank)), replace=False)
        parts = list(theme_phrases)

        if rng.random() < CONTAMINATION_RATE:
            other_themes = [t for t in THEMES if t != true_theme[i]]
            other_theme = rng.choice(other_themes)
            other_bank = PHRASE_BANK[other_theme]
            parts.append(rng.choice(other_bank))

        n_filler = rng.integers(0, 3)
        filler = rng.choice(FILLER_PHRASES, size=n_filler, replace=False) if n_filler else []
        parts = parts + list(filler)
        rng.shuffle(parts)
        text = f"Regarding {product[i]}: " + ". ".join(parts) + "."
        texts.append(text)

    df = pd.DataFrame({
        "case_id": [f"CASE-{i:06d}" for i in range(N_CASES)],
        "account_id": account_id,
        "product_id": product,
        "case_date": case_date,
        "case_text": texts,
        "true_theme": true_theme,
    })
    return df.sort_values("case_date").reset_index(drop=True)
