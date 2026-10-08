# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Rot3Array Matrix Class."""

from typing import Final

import numpy as np
import torch

from sparsefold.runtime import tree
from sparsefold.runtime import util
from sparsefold.model.geometry import utils
from sparsefold.model.geometry import vector

COMPONENTS: Final[tuple[str, ...]] = (
  *('xx', 'xy', 'xz'),
  *('yx', 'yy', 'yz'),
  *('zx', 'zy', 'zz'),
)
VERSION: Final[str] = '0.1'


def make_matrix_svd_factors() -> np.ndarray:
  """Generates factors for converting 3x3 matrix to symmetric 4x4 matrix."""
  factors = np.zeros((16, 9), dtype=np.float32)

  factors[0, [0, 4, 8]] = 1.0

  factors[[1, 4], 5] = 1.0
  factors[[1, 4], 7] = -1.0

  factors[[2, 8], 6] = 1.0
  factors[[2, 8], 2] = -1.0

  factors[[3, 12], 1] = 1.0
  factors[[3, 12], 3] = -1.0

  factors[5, 0] = 1.0
  factors[5, [4, 8]] = -1.0

  factors[[6, 9], 1] = 1.0
  factors[[6, 9], 3] = 1.0

  factors[[7, 13], 2] = 1.0
  factors[[7, 13], 6] = 1.0

  factors[10, 4] = 1.0
  factors[10, [0, 8]] = -1.0

  factors[[11, 14], 5] = 1.0
  factors[[11, 14], 7] = 1.0

  factors[15, 8] = 1.0
  factors[15, [0, 4]] = -1.0

  return factors


class LargestEvec(torch.autograd.Function):
  @staticmethod
  def forward(ctx, m):
    eigvals, eigvecs = torch.linalg.eigh(m)
    ctx.save_for_backward(eigvals, eigvecs)
    return eigvecs[..., -1]

  @staticmethod
  def backward(ctx, grad_output):
    eigvals, eigvecs = ctx.saved_tensors
    large_eigval = eigvals[..., -1]
    large_eigvec = eigvecs[..., -1]
    other_eigvals = eigvals[..., :-1]
    other_eigvecs = eigvecs[..., :-1]

    other_ev_times_tangent = torch.einsum(
      '...aj,...ab->...bj', other_eigvecs, grad_output
    )
    nominator = torch.einsum(
      '...bj,...b->...j', other_ev_times_tangent, large_eigvec
    )
    prefactor = nominator / torch.clamp(
      large_eigval[..., None] - other_eigvals, min=1e-6
    )
    grad = torch.sum(prefactor[..., None, :] * other_eigvecs, dim=-1)

    return grad


def largest_evec(m):
  return LargestEvec.apply(m)


MATRIX_SVD_QUAT_FACTORS = torch.as_tensor(make_matrix_svd_factors())


@tree.dataclass(mappable_dataclass=False, frozen=True)
class Rot3Array:
  """Rot3Array Matrix in 3 dimensional Space implemented as struct of arrays."""

  xx: torch.Tensor
  xy: torch.Tensor
  xz: torch.Tensor
  yx: torch.Tensor
  yy: torch.Tensor
  yz: torch.Tensor
  zx: torch.Tensor
  zy: torch.Tensor
  zz: torch.Tensor

  # def __init__(self, xx: torch.Tensor, xy: torch.Tensor, 
  #              xz: torch.Tensor, yx: torch.Tensor, yy: torch.Tensor, 
  #              yz: torch.Tensor, zx: torch.Tensor, zy: torch.Tensor, 
  #              zz: torch.Tensor):
  #   assert xx.dtype == xy.dtype == xz.dtype == yx.dtype == yy.dtype == yz.dtype == zx.dtype == zy.dtype == zz.dtype
  #   assert xx.shape == xy.shape == xz.shape == yx.shape == yy.shape == yz.shape == zx.shape == zy.shape == zz.shape
  #   self.xx = xx
  #   self.xy = xy
  #   self.xz = xz
  #   self.yx = yx
  #   self.yy = yy
  #   self.yz = yz
  #   self.zx = zx
  #   self.zy = zy
  #   self.zz = zz

  @property
  def dtype(self) -> torch.dtype:
    return self.xx.dtype

  @property
  def shape(self) -> torch.Size:
    return self.xx.shape

  def __len__(self) -> int:
    return self.xx.shape[0]

  def __getitem__(self, key: int | slice | torch.Tensor) -> 'Rot3Array':
    return Rot3Array(
      self.xx[key], self.xy[key], self.xz[key],
      self.yx[key], self.yy[key], self.yz[key],
      self.zx[key], self.zy[key], self.zz[key],
    )

  def inverse(self) -> 'Rot3Array':
    """Returns inverse of Rot3Array."""
    return Rot3Array(
      self.xx, self.yx, self.zx,
      self.xy, self.yy, self.zy,
      self.xz, self.yz, self.zz,
    )

  def apply_to_point(self, point: vector.Vec3Array) -> vector.Vec3Array:
    """Applies Rot3Array to point."""
    return vector.Vec3Array(
      self.xx * point.x + self.xy * point.y + self.xz * point.z,
      self.yx * point.x + self.yy * point.y + self.yz * point.z,
      self.zx * point.x + self.zy * point.y + self.zz * point.z,
    )

  def apply_inverse_to_point(self, point: vector.Vec3Array) -> vector.Vec3Array:
    """Applies inverse Rot3Array to point."""
    return self.inverse().apply_to_point(point)

  def __matmul__(self, other: 'Rot3Array') -> 'Rot3Array':
    """Composes two Rot3Arrays."""
    c0 = self.apply_to_point(vector.Vec3Array(other.xx, other.yx, other.zx))
    c1 = self.apply_to_point(vector.Vec3Array(other.xy, other.yy, other.zy))
    c2 = self.apply_to_point(vector.Vec3Array(other.xz, other.yz, other.zz))
    return Rot3Array(c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z)

  @classmethod
  def identity(cls, shape: torch.Size, dtype: torch.dtype = torch.float32) -> 'Rot3Array':
    """Returns identity of given shape."""
    ones = torch.ones(shape, dtype=dtype, device=util.DEFAULT_DEVICE)
    zeros = torch.zeros(shape, dtype=dtype, device=util.DEFAULT_DEVICE)
    return cls(ones, zeros, zeros, zeros, ones, zeros, zeros, zeros,
               ones)  # pytype: disable=wrong-arg-count  # trace-all-classes

  @classmethod
  def from_two_vectors(cls, e0: vector.Vec3Array, e1: vector.Vec3Array) -> 'Rot3Array':
    """Construct Rot3Array from two Vectors.

    Rot3Array is constructed such that in the corresponding frame 'e0' lies on
    the positive x-Axis and 'e1' lies in the xy plane with positive sign of y.

    Args:
      e0: Vector
      e1: Vector

    Returns:
      Rot3Array
    """
    # Normalize the unit vector for the x-axis, e0.
    e0 = e0.normalized()
    # make e1 perpendicular to e0.
    c = e1.dot(e0)
    e1 = (e1 - e0.__mul__(c)).normalized()
    # Compute e2 as cross product of e0 and e1.
    e2 = e0.cross(e1)
    return cls(e0.x, e1.x, e2.x, e0.y, e1.y, e2.y, e0.z, e1.z,
               e2.z)  # pytype: disable=wrong-arg-count  # trace-all-classes

  @classmethod
  def from_array(cls, array: torch.Tensor) -> 'Rot3Array':
    """Construct Rot3Array Matrix from array of shape. [..., 3, 3]."""
    unstacked = utils.unstack(array, dim=-2)
    unstacked = sum([utils.unstack(x, dim=-1) for x in unstacked], [])
    return cls(*unstacked)

  def to_array(self) -> torch.Tensor:
    """Convert Rot3Array to array of shape [..., 3, 3]."""
    return torch.stack(
      [
        torch.stack([self.xx, self.xy, self.xz], dim=-1),
        torch.stack([self.yx, self.yy, self.yz], dim=-1),
        torch.stack([self.zx, self.zy, self.zz], dim=-1),
      ],
      dim=-2,
    )

  @classmethod
  def from_quaternion(
      cls,
      w: torch.Tensor,
      x: torch.Tensor,
      y: torch.Tensor,
      z: torch.Tensor,
      normalize: bool = True,
      epsilon: float = 1e-6,
  ) -> 'Rot3Array':
    """Construct Rot3Array from components of quaternion."""
    if normalize:
      inv_norm = torch.rsqrt(torch.clamp(w ** 2 + x ** 2 + y ** 2 + z ** 2, min=epsilon))
      w *= inv_norm
      x *= inv_norm
      y *= inv_norm
      z *= inv_norm
    xx = 1 - 2 * (torch.square(y) + torch.square(z))
    xy = 2 * (x * y - w * z)
    xz = 2 * (x * z + w * y)
    yx = 2 * (x * y + w * z)
    yy = 1 - 2 * (torch.square(x) + torch.square(z))
    yz = 2 * (y * z - w * x)
    zx = 2 * (x * z - w * y)
    zy = 2 * (y * z + w * x)
    zz = 1 - 2 * (torch.square(x) + torch.square(y))
    return cls(xx, xy, xz, yx, yy, yz, zx, zy, zz)  # pytype: disable=wrong-arg-count  # trace-all-classes

  @classmethod
  def from_svd(cls, mat: torch.Tensor, use_quat_formula: bool = True) -> 'Rot3Array':
    """Constructs Rot3Array from arbitrary array of shape [3 * 3] using SVD.

    The case when 'use_quat_formula' is False rephrases the problem of
    projecting the matrix to a rotation matrix as a problem of finding the
    largest eigenvector of a certain 4x4 matrix. This has the advantage of
    having fewer numerical issues.
    This approach follows:
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.65.971&rep=rep1&type=pdf
    In the other case we construct it via svd following
    https://arxiv.org/pdf/2006.14616.pdf
    In that case [∂L/∂M] is large if the two smallest singular values are close
    to each other, or if they are close to 0.

    Args:
      mat: Array of shape [..., 3 * 3]
      use_quat_formula: Whether to construct matrix via 4x4 eigenvalue problem.

    Returns:
      Rot3Array of shape [...]
    """
    assert mat.shape[-1] == 9
    if mat.dtype not in {torch.float32, torch.float64}:
      raise ValueError(
          "Rot3Array.from_svd requires float32 or float64 for eigh/SVD; "
          "cast lower precision inputs explicitly before calling"
      )
    if use_quat_formula:
      mat_svd_quat_factor = MATRIX_SVD_QUAT_FACTORS.to(device=mat.device, dtype=mat.dtype)
      symmetric_4by4 = torch.einsum(
        'ji, ...i -> ...j',
        mat_svd_quat_factor,
        mat,
      )
      symmetric_4by4 = torch.reshape(symmetric_4by4, mat.shape[:-1] + (4, 4))
      largest_eigvec = largest_evec(symmetric_4by4)
      return cls.from_quaternion(
        *utils.unstack(largest_eigvec, dim=-1)
      ).inverse()

    else:
      mat = torch.reshape(mat, mat.shape[:-1] + (3, 3))
      u, _, v_t = torch.linalg.svd(mat, full_matrices=False)
      det_uv_t = torch.linalg.det(
        torch.matmul(u, v_t)
      )
      ones = torch.ones_like(det_uv_t)
      diag_array = torch.stack((ones, ones, det_uv_t), dim=-1)
      # This is equivalent to making diag_array into a diagonal array and matrix
      # multiplying
      diag_times_v_t = diag_array[..., None] * v_t
      out = torch.matmul(u, diag_times_v_t)
      return cls.from_array(out)

  @classmethod
  def random_uniform(cls, shape: torch.Size, dtype=torch.float32) -> 'Rot3Array':
    """Samples uniform random Rot3Array according to Haar Measure."""
    quat_array = torch.randn(tuple(shape) + (4,), dtype=dtype, device=util.DEFAULT_DEVICE)
    quats = utils.unstack(quat_array)
    return cls.from_quaternion(*quats)

  def __getstate__(self):
    return (VERSION, [np.asarray(getattr(self, field)) for field in COMPONENTS])

  def __setstate__(self, state):
    version, state = state
    del version
    for i, field in enumerate(COMPONENTS):
      object.__setattr__(self, field, state[i])
