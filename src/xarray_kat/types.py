import copy
from dataclasses import InitVar, dataclass, field
from typing import Any, Literal, Tuple

import numpy as np
import numpy.typing as npt

VanVleckLiteralType = Literal["off", "autocorr"]


@dataclass(eq=True, unsafe_hash=True, slots=True, repr=True)
class ArchiveArrayMetadata:
  """Holds metadata about arrays stored on the MeerKAT archive"""

  name: str
  default: Any
  dim_labels: Tuple[str, ...]
  prefix: str
  # Variables used to intialise the object, but
  # not stored on the object itself
  dask_chunks: InitVar[Tuple[Tuple[int, ...], ...]]
  dtype_str: InitVar[str]
  # Fields initialised from InitVars
  shape: Tuple[int, ...] = field(init=False)
  chunks: Tuple[int, ...] = field(init=False)
  dtype: npt.DTypeLike = field(init=False)

  def __post_init__(self, dask_chunks: Tuple[Tuple[int, ...], ...], dtype_str: str):
    for d, dim_chunks in enumerate(dask_chunks):
      if not all(dc == dim_chunks[0] for dc in dim_chunks[1:-1]):
        raise ValueError(
          f"Array {self.name} chunks {dim_chunks} are not homogenous in dimension {d}"
        )

    self.shape = tuple(map(sum, dask_chunks))
    self.chunks = tuple(dc[0] for dc in dask_chunks)
    self.dtype = np.dtype(dtype_str)
    if len(self.shape) != len(self.dim_labels):
      raise ValueError(
        f"{self.name} shape {self.shape} does not "
        f"match dimension labels {self.dim_labels}"
      )

  def copy(self, **kw):
    """Makes a copy of this object overridden by any key-values in kw"""
    obj = copy.copy(self)

    for k, v in kw.items():
      if k not in obj.__slots__:
        raise ValueError(f"{k} is not a valid {self.__class__.__name__} attribute")

      setattr(obj, k, v)

    return obj

  @property
  def ndim(self) -> int:
    """Returns the rank of the underlying array"""
    return len(self.shape)

  rank = ndim
