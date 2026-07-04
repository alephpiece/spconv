import os
from typing import Any, Optional, Tuple

import torch


_TRUE_VALUES = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
_CACHE_PREFIX = "__spconv_rocm_flexgemm_subm__"
_FLEXGEMM_SPConv = None
_CURRENT_ALGORITHM = None


def _enabled() -> bool:
    return os.environ.get("SPCONV_ROCM_FLEXGEMM_SUBM", "1") in _TRUE_VALUES


def _debug() -> bool:
    return os.environ.get("SPCONV_ROCM_FLEXGEMM_SUBM_DEBUG", "0") in _TRUE_VALUES


def _log(message: str) -> None:
    if _debug():
        print(f"[spconv-rocm flexgemm] {message}", flush=True)


def _get_flexgemm_spconv():
    global _FLEXGEMM_SPConv
    if _FLEXGEMM_SPConv is None:
        from flex_gemm.ops import spconv as flex_spconv

        _FLEXGEMM_SPConv = flex_spconv
    return _FLEXGEMM_SPConv


def _algorithm_name() -> str:
    return os.environ.get(
        "SPCONV_ROCM_FLEXGEMM_SUBM_ALGO",
        "masked_implicit_gemm_splitk",
    )


def _set_algorithm(flex_spconv) -> str:
    global _CURRENT_ALGORITHM
    name = _algorithm_name()
    choices = {
        flex_spconv.Algorithm.MASKED_IMPLICIT_GEMM,
        flex_spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK,
    }
    if name not in choices:
        raise ValueError(
            "SPCONV_ROCM_FLEXGEMM_SUBM_ALGO must be one of "
            f"{sorted(choices)}, got {name!r}"
        )
    if name != _CURRENT_ALGORITHM:
        flex_spconv.set_algorithm(name)
        _CURRENT_ALGORITHM = name
    return name


def _kernel_volume(module) -> int:
    volume = 1
    for size in module.kernel_size:
        volume *= int(size)
    return volume


def _supported(module, input_tensor) -> Tuple[bool, str]:
    if not _enabled():
        return False, "disabled"
    if not module.subm:
        return False, "not subm"
    if module.ndim != 3:
        return False, "not 3d"
    if module.inverse or module.transposed:
        return False, "inverse/transposed"
    if module.conv1x1:
        return False, "conv1x1"
    if list(module.stride) != [1, 1, 1]:
        return False, "stride"
    if any(int(size) % 2 == 0 for size in module.kernel_size):
        return False, "even kernel"
    if int(module.weight.ndim) != 3:
        return False, "weight layout"
    kernel_volume = _kernel_volume(module)
    if int(module.weight.shape[0]) != kernel_volume:
        return False, "kernel volume"
    if int(module.weight.shape[1]) != int(module.in_channels):
        return False, "input channels"
    if int(module.weight.shape[2]) != int(module.out_channels):
        return False, "output channels"
    if input_tensor.indices.ndim != 2 or input_tensor.indices.shape[1] != 4:
        return False, "indices shape"
    if input_tensor.indices.dtype != torch.int32:
        return False, "indices dtype"
    if input_tensor.features.dtype != torch.float32 or module.weight.dtype != torch.float32:
        return False, "dtype"
    if not input_tensor.features.is_cuda or not input_tensor.indices.is_cuda or not module.weight.is_cuda:
        return False, "device"
    if module.bias is not None and module.bias.dtype != torch.float32:
        return False, "bias dtype"
    if module.bias is not None and not module.bias.is_cuda:
        return False, "bias device"
    if kernel_volume > 32:
        return False, "kernel volume > 32"
    return True, "ok"


def _cache_key(module, input_tensor, algorithm: str, needs_grad: bool) -> Optional[Tuple[Any, ...]]:
    if module.indice_key is None:
        return None
    return (
        _CACHE_PREFIX,
        module.indice_key,
        tuple(module.kernel_size),
        tuple(module.dilation),
        tuple(input_tensor.spatial_shape),
        int(input_tensor.batch_size),
        int(input_tensor.indices.data_ptr()),
        int(input_tensor.indices.shape[0]),
        algorithm,
        bool(needs_grad),
    )


def _coords_bxyz(input_tensor) -> torch.Tensor:
    # spconv stores 3D coordinates as [batch, z, y, x]; FlexGEMM uses [batch, x, y, z].
    return input_tensor.indices[:, [0, 3, 2, 1]].contiguous()


def _shape_ncwxd(module, input_tensor) -> torch.Size:
    z, y, x = [int(v) for v in input_tensor.spatial_shape]
    return torch.Size([int(input_tensor.batch_size), int(module.in_channels), x, y, z])


def _weight_co_kw_kh_kd_ci(module) -> torch.Tensor:
    kd, kh, kw = [int(v) for v in module.kernel_size]
    weight = module.weight.view(kd, kh, kw, int(module.in_channels), int(module.out_channels))
    return weight.permute(4, 2, 1, 0, 3).contiguous()


def _bias(module, features: torch.Tensor) -> torch.Tensor:
    if module.bias is not None:
        return module.bias
    return torch.zeros((int(module.out_channels),), dtype=features.dtype, device=features.device)


def try_flexgemm_subm_conv(module, input_tensor):
    ok, reason = _supported(module, input_tensor)
    if not ok:
        _log(f"fallback: {reason}")
        return None

    flex_spconv = _get_flexgemm_spconv()
    algorithm = _set_algorithm(flex_spconv)
    needs_grad = (
        bool(input_tensor.features.requires_grad)
        or bool(module.weight.requires_grad)
        or bool(module.bias is not None and module.bias.requires_grad)
    )
    key = _cache_key(module, input_tensor, algorithm, needs_grad)
    cache = input_tensor.indice_dict.get(key) if key is not None else None

    coords = _coords_bxyz(input_tensor)
    shape = _shape_ncwxd(module, input_tensor)
    weight = _weight_co_kw_kh_kd_ci(module)
    bias = _bias(module, input_tensor.features)
    dilation = (int(module.dilation[2]), int(module.dilation[1]), int(module.dilation[0]))

    out_features, neighbor_cache = flex_spconv.sparse_submanifold_conv3d(
        input_tensor.features,
        coords,
        shape,
        weight,
        bias=bias,
        neighbor_cache=cache,
        dilation=dilation,
    )
    if key is not None and cache is None:
        input_tensor.indice_dict[key] = neighbor_cache

    return input_tensor.replace_feature(out_features)
