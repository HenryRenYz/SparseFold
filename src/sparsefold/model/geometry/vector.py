# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Vec3Array Class."""

from typing import Final, TypeAlias

import numpy as np
import torch

from sparsefold.runtime import tree
from sparsefold.runtime import util
from sparsefold.model.geometry import utils

Float: TypeAlias = float | torch.Tensor

VERSION: Final[str] = '0.1'


@tree.dataclass(mappable_dataclass=False, frozen=True)
class Vec3Array:
  """Vec3Array in 3 dimensional Space implemented as struct of arrays.

  This is done in order to improve performance and precision.
  On TPU small matrix multiplications are very suboptimal and will waste large
  compute ressources, furthermore any matrix multiplication on TPU happens in
  mixed bfloat16/float32 precision, which is often undesirable when handling
  physical coordinates.

  In most cases this will also be faster on CPUs/GPUs since it allows for easier
  use of vector instructions.
  """

  x: torch.Tensor
  y: torch.Tensor
  z: torch.Tensor

  # def __init__(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
  #   assert x.dtype == y.dtype == z.dtype
  #   assert x.shape == y.shape == z.shape
  #   self.x = x
  #   self.y = y
  #   self.z = z

  @property
  def dtype(self) -> torch.dtype:
    return self.x.dtype

  @property
  def shape(self) -> torch.Size:
    return self.x.shape

  def __len__(self) -> int:
    return self.x.shape[0]

  def __getitem__(self, key: int | slice | torch.Tensor) -> 'Vec3Array':
    return Vec3Array(
      self.x[key], self.y[key], self.z[key]
    )

  def __add__(self, other: 'Vec3Array') -> 'Vec3Array':
    return Vec3Array(
      self.x + other.x,
      self.y + other.y,
      self.z + other.z
    )

  def __sub__(self, other: 'Vec3Array') -> 'Vec3Array':
    return Vec3Array(
      self.x - other.x,
      self.y - other.y,
      self.z - other.z
    )

  def __mul__(self, other: Float) -> 'Vec3Array':
    return Vec3Array(
      self.x * other,
      self.y * other,
      self.z * other
    )

  def __rmul__(self, other: Float) -> 'Vec3Array':
    return self * other

  def __truediv__(self, other: Float) -> 'Vec3Array':
    return Vec3Array(
      self.x / other,
      self.y / other,
      self.z / other
    )

  def __neg__(self) -> 'Vec3Array':
    return Vec3Array(-self.x, -self.y, -self.z)

  def __pos__(self) -> 'Vec3Array':
    return Vec3Array(self.x, self.y, self.z)

  def cross(self, other: 'Vec3Array') -> 'Vec3Array':
    """Compute cross product between 'self' and 'other'."""
    new_x = self.y * other.z - self.z * other.y
    new_y = self.z * other.x - self.x * other.z
    new_z = self.x * other.y - self.y * other.x
    return Vec3Array(new_x, new_y, new_z)

  def dot(self, other: 'Vec3Array') -> Float:
    """Compute dot product between 'self' and 'other'."""
    return self.x * other.x + self.y * other.y + self.z * other.z

  def norm(self, epsilon: float = 1e-6) -> Float:
    """Compute Norm of Vec3Array, clipped to epsilon."""
    # To avoid NaN on the backward pass, we must use maximum before the sqrt
    norm2 = self.dot(self)
    if epsilon:
      norm2 = torch.clamp(norm2, min=epsilon ** 2)
    return torch.sqrt(norm2)

  def norm2(self):
    return self.dot(self)

  def normalized(self, epsilon: float = 1e-6) -> 'Vec3Array':
    """Return unit vector with optional clipping."""
    return self.__truediv__(self.norm(epsilon))
    # return self / self.norm(epsilon)

  @classmethod
  def zeros(cls, shape, dtype=torch.float32):
    """Return Vec3Array corresponding to zeros of given shape."""
    return cls(
      torch.zeros(shape, dtype=dtype, device=util.DEFAULT_DEVICE),
      torch.zeros(shape, dtype=dtype, device=util.DEFAULT_DEVICE),
      torch.zeros(shape, dtype=dtype, device=util.DEFAULT_DEVICE),
    )  # pytype: disable=wrong-arg-count  # trace-all-classes

  def to_array(self) -> torch.Tensor:
    return torch.stack([self.x, self.y, self.z], dim=-1)

  @classmethod
  def from_array(cls, array):
    return cls(*utils.unstack(array))

  def __getstate__(self):
    return (
      VERSION,
      [np.asarray(self.x), np.asarray(self.y), np.asarray(self.z)],
    )

  def __setstate__(self, state):
    version, state = state
    del version
    for i, letter in enumerate('xyz'):
      object.__setattr__(self, letter, state[i])


def square_euclidean_distance(
    vec1: Vec3Array, vec2: Vec3Array, epsilon: float = 1e-6
) -> Float:
  """Computes square of euclidean distance between 'vec1' and 'vec2'.

  Args:
    vec1: Vec3Array to compute  distance to
    vec2: Vec3Array to compute  distance from, should be broadcast compatible
      with 'vec1'
    epsilon: distance is clipped from below to be at least epsilon

  Returns:
    Array of square euclidean distances;
    shape will be result of broadcasting 'vec1' and 'vec2'
  """
  difference = vec1 - vec2
  distance = difference.dot(difference)
  if epsilon:
    distance = torch.clamp(distance, min=epsilon)
  return distance


def dot(vector1: Vec3Array, vector2: Vec3Array) -> Float:
  return vector1.dot(vector2)


def cross(vector1: Vec3Array, vector2: Vec3Array) -> Float:
  return vector1.cross(vector2)


def norm(vector: Vec3Array, epsilon: float = 1e-6) -> Float:
  return vector.norm(epsilon)


def normalized(vector: Vec3Array, epsilon: float = 1e-6) -> Vec3Array:
  return vector.normalized(epsilon)


def euclidean_distance(
    vec1: Vec3Array, vec2: Vec3Array, epsilon: float = 1e-6
) -> Float:
  """Computes euclidean distance between 'vec1' and 'vec2'.

  Args:
    vec1: Vec3Array to compute euclidean distance to
    vec2: Vec3Array to compute euclidean distance from, should be broadcast
      compatible with 'vec1'
    epsilon: distance is clipped from below to be at least epsilon

  Returns:
    Array of euclidean distances;
    shape will be result of broadcasting 'vec1' and 'vec2'
  """
  distance_sq = square_euclidean_distance(vec1, vec2, epsilon ** 2)
  distance = torch.sqrt(distance_sq)
  return distance


def dihedral_angle(
    a: Vec3Array, b: Vec3Array, c: Vec3Array, d: Vec3Array
) -> Float:
  """Computes torsion angle for a quadruple of points.

  For points (a, b, c, d), this is the angle between the planes defined by
  points (a, b, c) and (b, c, d). It is also known as the dihedral angle.

  Arguments:
    a: A Vec3Array of coordinates.
    b: A Vec3Array of coordinates.
    c: A Vec3Array of coordinates.
    d: A Vec3Array of coordinates.

  Returns:
    A tensor of angles in radians: [-pi, pi].
  """
  v1 = a - b
  v2 = b - c
  v3 = d - c

  c1 = v1.cross(v2)
  c2 = v3.cross(v2)
  c3 = c2.cross(c1)

  v2_mag = v2.norm()
  return torch.arctan2(c3.dot(v2), v2_mag * c1.dot(c2))


def random_gaussian_vector(shape, dtype=torch.float32) -> Vec3Array:
  vec_array = torch.randn(shape + (3,), dtype=dtype, device=util.DEFAULT_DEVICE)
  return Vec3Array.from_array(vec_array)
