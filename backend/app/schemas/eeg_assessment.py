"""Response contracts for the EEG neurological risk module.

The central difference from the speech pipeline: `risk_scores` holds three
**independent** sigmoid probabilities that deliberately do not sum to 1. A subject
may show elevated risk for more than one condition at once, which a softmax over
classes cannot express. `scores_are_independent` is carried in the payload so a
client cannot mistake this for a distribution.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RiskCondition = Literal["AD", "PD", "MS"]
RiskBand = Literal["Low", "Medium", "High"]
JobStatus = Literal[
    "queued", "validating", "preprocessing", "inference", "completed", "failed"
]


class ConditionRisk(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_band: RiskBand
    label: str
    epoch_score_std: float = Field(default=0.0, ge=0.0, le=1.0)
    epoch_score_range: list[float] = Field(default_factory=list, max_length=2)
    confound_severity: str = "unknown"


class RiskAssessment(BaseModel):
    conditions: dict[str, ConditionRisk]
    highest_risk_condition: str
    scores_are_independent: Literal[True] = True
    interpretation: str
    risk_bands: dict[str, float]


class FourClassPrediction(BaseModel):
    """Auxiliary softmax head. This one *is* a distribution."""

    predicted_class: str
    class_probabilities: dict[str, float]
    note: str = (
        "Auxiliary head. Mutually exclusive by construction, so it cannot express "
        "co-elevated risk. The risk_scores block is the module's primary output."
    )


class IcaRejection(BaseModel):
    component: int
    criteria: list[str]
    kurtosis: float | None = None
    frontal_corr: float | None = None
    hf_power_ratio: float | None = None


class SignalQuality(BaseModel):
    epochs_used: int = Field(..., ge=0)
    total_epochs_generated: int = Field(default=0, ge=0)
    clean_epoch_ratio: float = Field(..., ge=0.0, le=1.0)
    grade: str
    ica_components_removed: int = Field(default=0, ge=0)
    ica_rejections: list[IcaRejection] = Field(default_factory=list)
    channels: int = Field(..., gt=0)
    sampling_rate_hz: float = Field(..., gt=0)
    source_kind: str = "continuous"
    warnings: list[str] = Field(default_factory=list)


class EmbeddingSummary(BaseModel):
    """The fusion contract, summarised. The full 256-D vector is a separate call."""

    dim: int = Field(..., gt=0)
    l2_norm: float = Field(..., ge=0.0)
    availability_flag: Literal[0, 1]
    consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    cosine_to_class_centroids: dict[str, float] = Field(default_factory=dict)
    nearest_centroid: str | None = None
    vector_url: str | None = None


class Explainability(BaseModel):
    scalp_region_importance: dict[str, float] = Field(default_factory=dict)
    band_importance: dict[str, float] = Field(default_factory=dict)
    method: str = (
        "occlusion — drop in predicted-condition probability when the input is zeroed"
    )


class ConfoundDisclosure(BaseModel):
    """Travels with every report so the caveat cannot be separated from the number."""

    age_probe_mae_years: float | None = None
    age_probe_improvement_over_baseline: float | None = None
    site_probe_balanced_accuracy: float | None = None
    risk_score_age_correlation: dict[str, float] = Field(default_factory=dict)
    severity_by_condition: dict[str, str] = Field(default_factory=dict)
    statement: str
    has_critical: bool = False


class EegRiskReport(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    subject_id: str
    source: Literal["cohort", "upload"]
    generated_at: str
    dataset: dict[str, str] = Field(default_factory=dict)

    risk_scores: dict[str, float]
    risk_assessment: RiskAssessment
    optional_four_class_prediction: FourClassPrediction | None = None

    signal_quality: SignalQuality
    band_power_profile: dict[str, float] = Field(default_factory=dict)
    embedding: EmbeddingSummary
    explainability: Explainability = Field(default_factory=Explainability)
    confound_disclosure: ConfoundDisclosure
    model_summary: dict[str, str | int | float | None] = Field(default_factory=dict)
    clinical_disclaimer: str


class CohortSubject(BaseModel):
    """One row of the cohort index — enough to render a list without the full report."""

    subject_id: str
    true_class: str
    site: str
    source_kind: str
    signal_quality: str
    epochs_used: int
    age: float | None = None
    highest_risk_condition: str
    highest_risk_score: float
    risk_scores: dict[str, float]
    confound_severity: str


class CohortPage(BaseModel):
    total: int
    offset: int
    limit: int
    subjects: list[CohortSubject]
    available_filters: dict[str, list[str]]


class ProjectionPoint(BaseModel):
    subject_id: str
    x: float
    y: float
    true_class: str
    site: str


class CohortProjection(BaseModel):
    method: str
    points: list[ProjectionPoint]
    explained_variance: list[float] | None = None
    # Fraction of subjects whose nearest neighbour in the FULL embedding space shares
    # their class, overall and per class. PC1 + PC2 of a 256-D space capture a small
    # slice of the variance, so a scatter can look unstructured while the geometry it
    # is drawn from separates cleanly. Without this a reader concludes the encoder
    # failed. Keys: "overall" plus one per class.
    neighbourhood_agreement: dict[str, float] = Field(default_factory=dict)
    note: str


class GroupBandSummary(BaseModel):
    """Band power across one diagnostic group — the comparison baseline."""

    n: int = Field(..., ge=0)
    medians: dict[str, float] = Field(default_factory=dict)
    q1: dict[str, float] = Field(default_factory=dict)
    q3: dict[str, float] = Field(default_factory=dict)


class ConditionBandProfile(BaseModel):
    """How one condition's band power sits against controls, and whether that holds.

    `has_signature` is the honesty gate: False means the cohort shows no band-power
    pattern for this condition, and the UI must decline to point at one rather than
    dressing up noise. See `services/eeg_band_statistics` for the rule.
    """

    condition: str
    n: int = Field(..., ge=0)
    medians: dict[str, float] = Field(default_factory=dict)
    auc_vs_hc: dict[str, float] = Field(default_factory=dict)
    direction_vs_hc: dict[str, str] = Field(default_factory=dict)
    separating_bands: list[str] = Field(default_factory=list)
    opposing_bands: list[str] = Field(default_factory=list)
    has_signature: bool = False
    note: str = ""


class BandReference(BaseModel):
    """Cohort-level band statistics. Descriptive, not an attribution of the score."""

    generated_at: str
    bands: list[str] = Field(default_factory=list)
    separation_margin: float = 0.15
    canonical_axis: str = "theta_alpha_ratio"
    slowing_direction: dict[str, str] = Field(default_factory=dict)
    healthy: GroupBandSummary
    conditions: dict[str, ConditionBandProfile] = Field(default_factory=dict)
    method: str = ""


class EmbeddingVector(BaseModel):
    subject_id: str
    dim: int
    l2_norm: float
    availability_flag: Literal[0, 1]
    z_eeg: list[float]


class ModelCard(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    generated_at: str | None = None
    architecture: str
    input_representation: str
    input_shape: list[int]
    embedding_dim: int
    risk_conditions: list[str]
    cohort: dict[str, object] = Field(default_factory=dict)
    performance: dict[str, object] = Field(default_factory=dict)
    confound_disclosure: ConfoundDisclosure
    intended_use: dict[str, object] = Field(default_factory=dict)
    inference_available: bool = Field(
        default=False,
        description="False when PyTorch is absent; cohort browsing still works.",
    )


class JobError(BaseModel):
    code: str
    message: str
    # Structured explanation of the failure — e.g. which channels were too noisy
    # and by how much — so the UI can be specific rather than saying "it failed".
    details: dict[str, object] = Field(default_factory=dict)


class EegJob(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(default=0, ge=0, le=100)
    stage_label: str = ""
    filename: str | None = None
    created_at: str
    updated_at: str
    report: EegRiskReport | None = None
    error: JobError | None = None
