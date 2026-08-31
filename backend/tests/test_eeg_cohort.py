import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import eeg_cohort_service as cohort


client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_caches():
    cohort.reset_caches()
    yield
    cohort.reset_caches()


def test_cohort_index_returns_subjects() -> None:
    response = client.get("/eeg/cohort", params={"limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] > 0
    assert len(body["subjects"]) <= 5
    row = body["subjects"][0]
    for field in ("subject_id", "true_class", "site", "signal_quality",
                  "highest_risk_condition", "risk_scores", "confound_severity"):
        assert field in row


def test_cohort_filters_narrow_results() -> None:
    unfiltered = client.get("/eeg/cohort", params={"limit": 1}).json()["total"]
    filtered = client.get("/eeg/cohort", params={"true_class": "AD", "limit": 1}).json()
    assert 0 < filtered["total"] < unfiltered
    assert all(s["true_class"] == "AD" for s in
               client.get("/eeg/cohort", params={"true_class": "AD", "limit": 200})
               .json()["subjects"])


def test_cohort_rejects_unknown_filter_value() -> None:
    assert client.get("/eeg/cohort", params={"true_class": "NOPE"}).status_code == 422


def test_cohort_search_matches_partial_id_across_the_full_index() -> None:
    subjects = cohort.load_cohort_index()
    target = subjects[-1]
    fragment = target.subject_id[-8:].lower()
    body = client.get("/eeg/cohort", params={"search": fragment, "limit": 200}).json()
    assert target.subject_id in {subject["subject_id"] for subject in body["subjects"]}


@pytest.mark.parametrize(
    ("query", "expected_class"),
    [("alzheimer", "AD"), ("parkinson", "PD"), ("multiple sclerosis", "MS"), ("healthy", "HC")],
)
def test_cohort_search_matches_condition_names(query: str, expected_class: str) -> None:
    body = client.get("/eeg/cohort", params={"search": query, "limit": 200}).json()
    assert body["total"] > 0
    assert all(
        subject["true_class"] == expected_class
        or subject["highest_risk_condition"] == expected_class
        for subject in body["subjects"]
    )


def test_cohort_search_combines_terms_and_filters() -> None:
    subject = cohort.load_cohort_index()[0]
    body = client.get(
        "/eeg/cohort",
        params={
            "search": f"{subject.subject_id[:6]} {subject.signal_quality}",
            "site": subject.site,
            "limit": 200,
        },
    ).json()
    assert body["total"] > 0
    assert all(item["site"] == subject.site for item in body["subjects"])


def test_report_has_every_required_block() -> None:
    subject_id = client.get("/eeg/cohort", params={"limit": 1}).json()["subjects"][0]["subject_id"]
    body = client.get(f"/eeg/cohort/{subject_id}").json()
    for block in ("risk_scores", "risk_assessment", "signal_quality", "embedding",
                  "confound_disclosure", "clinical_disclaimer", "band_power_profile"):
        assert block in body, f"missing {block}"
    assert "not a clinical diagnosis" in body["clinical_disclaimer"]


def test_risk_scores_are_independent_not_a_distribution() -> None:
    """The single most likely regression: somebody "fixes" the scores to sum to 1.

    They are independent sigmoids. If every subject summed to 1.0 the model would
    have silently become a softmax and every risk meter in the UI would be wrong.
    """
    subjects = client.get("/eeg/cohort", params={"limit": 60}).json()["subjects"]
    sums = [sum(s["risk_scores"].values()) for s in subjects]
    assert any(abs(total - 1.0) > 0.05 for total in sums), (
        "every subject's risk scores sum to ~1.0 — these are supposed to be "
        "independent sigmoids, not a distribution"
    )


def test_report_declares_independence_flag() -> None:
    subject_id = client.get("/eeg/cohort", params={"limit": 1}).json()["subjects"][0]["subject_id"]
    body = client.get(f"/eeg/cohort/{subject_id}").json()
    assert body["risk_assessment"]["scores_are_independent"] is True
    assert "do not sum to 1" in body["risk_assessment"]["interpretation"]


def test_every_condition_carries_a_band_and_severity() -> None:
    subject_id = client.get("/eeg/cohort", params={"limit": 1}).json()["subjects"][0]["subject_id"]
    conditions = client.get(f"/eeg/cohort/{subject_id}").json()["risk_assessment"]["conditions"]
    assert set(conditions) == {"AD", "PD", "MS"}
    for name, entry in conditions.items():
        assert entry["risk_band"] in {"Low", "Medium", "High"}
        assert entry["confound_severity"]
        assert 0.0 <= entry["risk_score"] <= 1.0


def test_disclosure_is_merged_from_model_card_not_the_stored_report(monkeypatch) -> None:
    """Confound numbers describe the model, so a stale stored copy must not win."""
    subject_id = client.get("/eeg/cohort", params={"limit": 1}).json()["subjects"][0]["subject_id"]
    card = cohort.load_model_card()
    sentinel = "SENTINEL-DISCLOSURE-TEXT"
    patched = card.model_copy(deep=True)
    patched.confound_disclosure.statement = sentinel

    monkeypatch.setattr(cohort, "load_model_card", lambda: patched)
    body = cohort.get_report(subject_id)
    assert body.confound_disclosure.statement == sentinel


def test_disclosure_flags_the_confounded_condition() -> None:
    """MS is the confounded class on this cohort; the exact grade is measured.

    Severity is derived from evidence — site purity plus the within-negatives
    age correlation — so it moves between HIGH and CRITICAL as the data changes.
    What must hold is that MS is flagged at all and that `has_critical` agrees
    with the severity strings, because the UI keys its render-blocking banner
    off that flag.
    """
    disclosure = client.get("/eeg/model-card").json()["confound_disclosure"]
    severity = disclosure["severity_by_condition"]
    assert "MS" in severity
    assert severity["MS"].lower() != "low", "MS is single-site and must stay flagged"

    any_critical = any("CRITICAL" in value.upper() for value in severity.values())
    assert disclosure["has_critical"] == any_critical, (
        "has_critical must match the severity strings — the UI uses it to decide "
        "whether the disclosure banner is dismissible"
    )


def test_unknown_subject_returns_404() -> None:
    assert client.get("/eeg/cohort/does-not-exist").status_code == 404


def test_path_traversal_is_rejected() -> None:
    for attempt in ("..", "../../etc/passwd", "a/b"):
        assert client.get(f"/eeg/cohort/{attempt}").status_code in {404, 422}


def test_embedding_vector_is_unit_norm() -> None:
    subject_id = client.get("/eeg/cohort", params={"limit": 1}).json()["subjects"][0]["subject_id"]
    body = client.get(f"/eeg/embeddings/{subject_id}").json()
    assert body["dim"] == 256
    assert len(body["z_eeg"]) == 256
    assert abs(body["l2_norm"] - 1.0) < 1e-3
    assert body["availability_flag"] == 1


def test_projection_covers_the_cohort() -> None:
    """Every assessed subject must appear, not just the run's demo subjects.

    The training run exported z_eeg for four subjects, which made the scatter look
    complete while showing 3% of the cohort. `backfill_eeg_embeddings.py` recomputes
    the rest from the fold that held each subject out; this asserts the result is
    actually cohort-wide so a future bundle swap cannot silently regress to four.
    """
    body = client.get("/eeg/cohort/projection").json()
    assert body["method"] == "PCA"
    point = body["points"][0]
    assert {"subject_id", "x", "y", "true_class"} <= set(point)

    total = client.get("/eeg/cohort", params={"limit": 1}).json()["total"]
    plotted = {p["subject_id"] for p in body["points"]}
    assert len(plotted) == len(body["points"]), "duplicate subject in the projection"
    assert len(plotted) == total, (
        f"projection covers {len(plotted)} of {total} subjects — rerun "
        "scripts/backfill_eeg_embeddings.py"
    )
    assert {p["true_class"] for p in body["points"]} == {"HC", "AD", "PD", "MS"}


def test_every_subject_has_a_retrievable_embedding() -> None:
    subjects = client.get("/eeg/cohort", params={"limit": 200}).json()["subjects"]
    for subject in subjects[:12]:
        body = client.get(f"/eeg/embeddings/{subject['subject_id']}").json()
        assert body["dim"] == 256
        assert abs(body["l2_norm"] - 1.0) < 1e-3
        assert body["availability_flag"] == 1


def test_reports_carry_centroid_similarities() -> None:
    """The embedding panel is empty without these, which reads as 'not measured'."""
    subjects = client.get("/eeg/cohort", params={"limit": 200}).json()["subjects"]
    missing = []
    for subject in subjects[:12]:
        report = client.get(f"/eeg/cohort/{subject['subject_id']}").json()
        cosines = report["embedding"]["cosine_to_class_centroids"]
        if set(cosines) != {"HC", "AD", "PD", "MS"}:
            missing.append(subject["subject_id"])
        else:
            assert report["embedding"]["nearest_centroid"] in cosines
    assert not missing, f"no centroid similarities for: {missing}"


def test_band_reference_serves_every_condition_with_a_verdict() -> None:
    body = client.get("/eeg/band-reference").json()
    assert body["healthy"]["n"] > 0
    assert set(body["conditions"]) == {"AD", "PD", "MS"}
    for name, profile in body["conditions"].items():
        assert set(profile["auc_vs_hc"]) == set(body["bands"]), f"{name} missing a band"
        assert profile["note"], f"{name} has no verdict text"
        assert isinstance(profile["has_signature"], bool)


def test_band_reference_medians_match_the_cohort_reports() -> None:
    """The reference must describe the reports actually being served, not a stale run."""
    reference = client.get("/eeg/band-reference").json()
    subjects = client.get("/eeg/cohort", params={"true_class": "HC", "limit": 200}).json()
    assert subjects["total"] == reference["healthy"]["n"]


def test_band_reference_declines_to_claim_a_pattern_it_cannot_support() -> None:
    """At least one condition here has no band-power signature; it must say so.

    Guards the honesty gate end to end: if the API ever reports a signature for
    every condition, the panel it drives has stopped being evidence-led.
    """
    conditions = client.get("/eeg/band-reference").json()["conditions"]
    without = [name for name, p in conditions.items() if not p["has_signature"]]
    assert without, "expected at least one condition with no band-power signature"
    for name in without:
        assert "No pattern is claimed" in conditions[name]["note"]
