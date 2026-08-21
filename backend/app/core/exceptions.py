class AppError(Exception):
    """Base application error."""


class ModelLoadError(AppError):
    """Raised when the model cannot be loaded."""


class FeatureValidationError(AppError):
    """Raised when request features do not match the trained model contract."""


class PredictionError(AppError):
    """Raised when prediction output is invalid or cannot be produced."""


class AudioFeatureExtractionError(AppError):
    """Raised when an uploaded audio sample cannot be analyzed safely."""


class EegBundleError(AppError):
    """Raised when the EEG model bundle or cohort store is missing or malformed."""


class EegSubjectNotFoundError(AppError):
    """Raised when a requested cohort subject does not exist."""


class EegIngestError(AppError):
    """Raised when an uploaded EEG recording cannot be read."""


class EegQualityError(AppError):
    """Raised when a recording yields too few clean epochs to assess.

    Carries a structured diagnostic so the caller can explain *why* the recording
    was rejected — which channels were noisy and by how much — instead of only
    reporting that it was.
    """

    def __init__(self, message: str, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class EegJobNotFoundError(AppError):
    """Raised when an assessment job id is unknown or has expired."""
