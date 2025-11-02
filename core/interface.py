from .interpreter import _Model, PomapNode, _fit
from dataclasses import dataclass
from polars import DataFrame

@dataclass
class Model(_Model):
    root: PomapNode
    models=None
    hyperparameters=None

    def fit(self, df: DataFrame):
        models, hyperparameters = _fit(self.root, df)
        self.models = models
        self.hyperparameters = hyperparameters
