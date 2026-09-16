"""Serializable policy designs, with an explicit source identity for custom code."""

import importlib
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .artifacts import sha256


@dataclass(frozen=True)
class Architecture:
    layers: tuple[int, ...] = (128, 128)
    activation: str = "Tanh"
    extractor: str | None = None

    def __post_init__(self):
        if not 1 <= len(self.layers) <= 4 or any(
            type(width) is not int or not 16 <= width <= 512 for width in self.layers
        ):
            raise ValueError("Use 1..4 hidden layers, each with 16..512 units")
        if self.activation not in ("Tanh", "ReLU", "ELU"):
            raise ValueError("Unsupported activation")

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "layers": tuple(value.get("layers", (128, 128)))})

    def policy_kwargs(self):
        result = {
            "net_arch": {"pi": list(self.layers), "vf": list(self.layers)},
            "activation_fn": getattr(torch.nn, self.activation),
        }
        if self.extractor:
            module_name, class_name = self.extractor.split(":")
            if not module_name.startswith("qwop_lab.candidates."):
                raise ValueError("Custom architectures must live in qwop_lab.candidates")
            extractor = getattr(importlib.import_module(module_name), class_name)
            if not issubclass(extractor, BaseFeaturesExtractor):
                raise ValueError("Custom extractor must inherit BaseFeaturesExtractor")
            result["features_extractor_class"] = extractor
        return result

    def identity(self):
        value = asdict(self)
        value["layers"] = list(self.layers)
        kwargs = self.policy_kwargs()
        if self.extractor:
            source = Path(inspect.getfile(kwargs["features_extractor_class"]))
            value["source_sha256"] = sha256(source)
            value["source_file"] = str(source.resolve())
        return value
