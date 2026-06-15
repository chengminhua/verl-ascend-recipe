import os

import torch
import torch_npu
from vllm.logger import logger
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant

try:
    import batch_invariant_ops  # type: ignore[import-not-found] # noqa: F401

    HAS_ASCENDC_BATCH_INVARIANT_OPS = True
except ImportError:
    HAS_ASCENDC_BATCH_INVARIANT_OPS = False


_ORIGIN_TORCH_SUM = torch.sum
_BATCH_INVARIANT_ATEN_LIB = None

_NON_FLOAT_SUM_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
)


def batch_invariant_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    merged = x + residual
    output, _ = torch_npu.npu_rms_norm(merged, weight, eps)
    return output, None, merged


def fallback_sum(
    x: torch.Tensor,
    dim=None,
    keepdim: bool = False,
    dtype=None,
):
    return _ORIGIN_TORCH_SUM(x, dim=dim, keepdim=keepdim, dtype=dtype)


class BatchInvariantSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, keepdim, dtype, axis):
        if axis is not None and dim is not None:
            raise ValueError("Cannot specify both 'dim' and 'axis'. Use only 'dim'.")

        if axis is not None:
            dim = axis

        ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.original_shape = x.shape

        if dtype is not None and dtype != x.dtype:
            x = x.to(dtype)

        if x.dtype in _NON_FLOAT_SUM_DTYPES:
            return fallback_sum(x, dim, keepdim)

        if x.device.type != "npu" or dim is None:
            return fallback_sum(x, dim, keepdim)

        ndim = x.dim()
        if dim < 0:
            dim += ndim

        if dim == ndim - 1:
            return torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(x, dim, keepdim)

        permute_order = list(range(ndim))
        permute_order.remove(dim)
        permute_order.append(dim)

        x_permuted = x.permute(permute_order)
        result = torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(x_permuted, -1, keepdim)

        if keepdim:
            restore_order = [0] * len(permute_order)
            for index, axis_id in enumerate(permute_order):
                restore_order[axis_id] = index
            result = result.permute(restore_order)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        dim = ctx.dim
        keepdim = ctx.keepdim
        original_shape = ctx.original_shape

        if dim is None:
            return grad_output.expand_as(input_tensor), None, None, None, None

        if dim < 0:
            dim += len(original_shape)

        if not keepdim:
            grad_output = grad_output.unsqueeze(dim)

        grad_input = grad_output.expand_as(input_tensor)
        return grad_input, None, None, None, None


def batch_invariant_sum(
    x: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    dtype=None,
    axis=None,
) -> torch.Tensor:
    return BatchInvariantSumFunction.apply(x, dim, keepdim, dtype, axis)


class BatchInvariantLogSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        output = torch.ops.batch_invariant_ops.npu_log_softmax_batch_invariant(x, dim)
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        dim = ctx.dim

        exp_output = torch.exp(output)
        grad_sum = grad_output.sum(dim=dim, keepdim=True)
        grad_input = grad_output - exp_output * grad_sum

        return grad_input, None


def batch_invariant_log_softmax(x, dim, dtype=None):
    if dtype is not None and dtype != x.dtype:
        x = x.to(dtype)

    return BatchInvariantLogSoftmaxFunction.apply(x, dim)


class BatchInvariantSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        output = torch.ops.batch_invariant_ops.npu_softmax_batch_invariant(x, dim)
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        dim = ctx.dim

        grad_dot = (grad_output * output).sum(dim=dim, keepdim=True)
        grad_input = output * (grad_output - grad_dot)

        return grad_input, None


def batch_invariant_softmax(x, dim, dtype=None):
    if dtype is not None and dtype != x.dtype:
        x = x.to(dtype)

    return BatchInvariantSoftmaxFunction.apply(x, dim)


def setup_batch_invariant_envs():
    os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"
    os.environ["HCCL_DETERMINISTIC"] = "strict"
    os.environ["LCCL_DETERMINISTIC"] = "1"


def register_batch_invariant_ops():
    global _BATCH_INVARIANT_ATEN_LIB

    _BATCH_INVARIANT_ATEN_LIB = torch.library.Library("aten", "IMPL")

    _BATCH_INVARIANT_ATEN_LIB.impl(
        "aten::mm",
        torch.ops.batch_invariant_ops.npu_mm_batch_invariant,
        "NPU",
    )
    _BATCH_INVARIANT_ATEN_LIB.impl(
        "aten::matmul",
        torch.ops.batch_invariant_ops.npu_matmul_batch_invariant,
        "NPU",
    )
    _BATCH_INVARIANT_ATEN_LIB.impl("aten::sum", batch_invariant_sum, "NPU")
    _BATCH_INVARIANT_ATEN_LIB.impl("aten::softmax", batch_invariant_softmax, "NPU")
    _BATCH_INVARIANT_ATEN_LIB.impl("aten::_softmax", batch_invariant_softmax, "NPU")
    _BATCH_INVARIANT_ATEN_LIB.impl(
        "aten::log_softmax",
        batch_invariant_log_softmax,
        "NPU",
    )
    _BATCH_INVARIANT_ATEN_LIB.impl(
        "aten::_log_softmax",
        batch_invariant_log_softmax,
        "NPU",
    )

    torch_npu.npu_add_rms_norm = batch_invariant_add_rms_norm

    torch.sum = batch_invariant_sum
    torch.Tensor.sum = batch_invariant_sum


def init_batch_invariance_replace():
    if not vllm_is_batch_invariant():
        return

    if not HAS_ASCENDC_BATCH_INVARIANT_OPS:
        logger.warning(
            "Batch-invariant mode requested but AscendC batch-invariant ops "
            "are not available. Skipping batch-invariant initialization."
        )
        return

    logger.info("Enabling batch-invariant mode for vLLM on Ascend NPU.")
    setup_batch_invariant_envs()
    register_batch_invariant_ops()
