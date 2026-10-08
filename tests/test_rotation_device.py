# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Small rotation projection checks for constant device/dtype and eigensolver inputs."""
import pytest
import torch

from sparsefold.model.geometry.rotation_matrix import Rot3Array


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("quaternion", [True, False])
def test_rotation_projection_preserves_known_rotations(device, dtype, quaternion):
  if device == "cuda" and not torch.cuda.is_available():
    pytest.skip("CUDA unavailable")
  rotations = torch.tensor([
      [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
      [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
      [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
  ], device=device, dtype=dtype)
  projected = Rot3Array.from_svd(rotations.reshape(3, 9), use_quat_formula=quaternion).to_array()
  assert projected.device == rotations.device and projected.dtype == dtype
  torch.testing.assert_close(projected, rotations, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("quaternion", [True, False])
def test_rotation_projection_rejects_implicit_low_precision_eigensolve(dtype, quaternion):
  with pytest.raises(ValueError, match="cast lower precision inputs explicitly"):
    Rot3Array.from_svd(torch.eye(3, dtype=dtype).reshape(9), use_quat_formula=quaternion)



@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_geometry_weighted_mean_uses_value_device_and_dtype(device, dtype):
  if device == "cuda" and not torch.cuda.is_available():
    pytest.skip("CUDA unavailable")
  from sparsefold.model.geometry import utils
  values = torch.tensor([[1., 4., 8.], [2., 3., 9.]], device=device, dtype=dtype)
  # Host/list weights must align to CUDA values without changing the calculation.
  result = utils.weighted_mean([1., 2., 0.], values, dim=-1)
  expected = torch.tensor([3., 8. / 3.], device=device, dtype=dtype)
  assert result.device == values.device and result.dtype == dtype
  torch.testing.assert_close(result, expected)
  scalar = utils.weighted_mean(1., values, dim=-1)
  torch.testing.assert_close(scalar, values.mean(dim=-1))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("weight_mode", ["default", "host_list", "host_tensor"])
def test_point_alignment_recovers_known_rigid_transform(device, dtype, weight_mode):
  if device == "cuda" and not torch.cuda.is_available():
    pytest.skip("CUDA unavailable")
  from sparsefold.model.geometry.rigid_matrix_vector import Rigid3Array
  from sparsefold.model.geometry.vector import Vec3Array
  points = torch.tensor([
      [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
      [-1., 2., 0.5], [0.5, -1., 2.],
  ], device=device, dtype=dtype)
  rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device=device, dtype=dtype)
  translation = torch.tensor([2., -1., 0.5], device=device, dtype=dtype)
  target = points @ rotation.T + translation
  weights = None
  if weight_mode == "host_list":
    weights = [1., 2., 1., 2., 1., 3.]
  elif weight_mode == "host_tensor":
    weights = torch.tensor([1., 2., 1., 2., 1., 3.], dtype=torch.float64, device="cpu")
  transform = Rigid3Array.from_point_alignment(Vec3Array.from_array(target), Vec3Array.from_array(points), weights=weights)
  assert transform.rotation.to_array().device == points.device
  assert transform.translation.to_array().dtype == dtype
  torch.testing.assert_close(transform.rotation.to_array(), rotation, rtol=1e-5, atol=5e-6)
  torch.testing.assert_close(transform.translation.to_array(), translation, rtol=1e-5, atol=5e-6)
  mapped = transform.apply_to_point(Vec3Array.from_array(points)).to_array()
  torch.testing.assert_close(mapped, target, rtol=1e-5, atol=5e-6)
