from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class KVCache:
    """Key-Value cache for storing attention keys and values."""

    k: Tensor
    v: Tensor
    limit: int

    def __init__(self, limit: int = 1024):
        """Initialize an empty cache with a bounded retained history."""

        self.k = torch.zeros(1, 1, 1, 1)
        self.v = torch.zeros(1, 1, 1, 1)
        self.limit = limit
        self._placeholder = True

    def append(self, k: Tensor, v: Tensor) -> None:
        """Append new keys/values, replacing the placeholders on first use or shape change."""

        if self._placeholder or not self._compatible(self.k, k) or not self._compatible(self.v, v):
            self.k = k
            self.v = v
            self._placeholder = False
        else:
            self.k = torch.cat((self.k, k), dim=-2)
            self.v = torch.cat((self.v, v), dim=-2)

        self.k = self.k[..., -self.limit :, :]
        self.v = self.v[..., -self.limit :, :]

    @staticmethod
    def _compatible(current: Tensor, new: Tensor) -> bool:
        """Whether `new` can be concatenated to `current` along the sequence dimension."""

        return current.shape[:-2] == new.shape[:-2] and current.shape[-1] == new.shape[-1]

    def detach(self) -> None:
        """End gradient history at an optimizer-step boundary."""

        self.k = self.k.detach()
        self.v = self.v.detach()

    @property
    def length(self) -> int:
        """Return the number of real entries, excluding the initial placeholder."""

        return 0 if self._placeholder else self.k.shape[-2]
