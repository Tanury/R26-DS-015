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
