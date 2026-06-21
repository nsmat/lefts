from typing import Literal, get_args

FitLogging = Literal["capture", "drop", "print"]
FitErrors = Literal["capture", "raise"]
PredictErrors = Literal["raise", "skip_unfit_models", "output_nan"]


def _check_literal(value, annotation, name: str) -> None:
    allowed = get_args(annotation)
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
