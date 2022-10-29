"""Operator implementations."""

from numbers import Number
from typing import Optional, List

import needle
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        result = out_grad * rhs, out_grad * lhs
        return result


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        result = (self.scalar * PowerScalar(self.scalar - 1)(node.inputs[0])) * out_grad
        return result
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        assert a.shape == b.shape
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad * 1.0 / rhs, out_grad * lhs * -1.0/rhs**2
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return (a / self.scalar).astype(array_api.float32)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * 1.0 /self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            return a.swapaxes(self.axes[0], self.axes[1])
        return a.swapaxes(-1, -2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Transpose(self.axes)(out_grad)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp = node.inputs[0]
        return Reshape(inp.shape)(out_grad)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        src_shape = list((node.inputs[0].shape))
        trg_shape = list((self.shape))
        src_shape = [1]*(len(trg_shape) - len(src_shape)) + src_shape
        summation_indices = []
        for i in range(len(src_shape)):
            if src_shape[i] == trg_shape[i]:
                continue
            else:
                summation_indices.append(i)
        result = Reshape(node.inputs[0].shape)(Summation(tuple(summation_indices))(out_grad))
        return  result
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp_shape = node.inputs[0].shape
        reduced_shape = list(inp_shape)
        if self.axes is None:
            result = BroadcastTo(inp_shape)(out_grad)
            return result
        for axis in self.axes:
            reduced_shape[axis] = 1
        result = BroadcastTo(inp_shape)(Reshape(reduced_shape)(out_grad))
        return result
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        res1 = MatMul()(out_grad, Transpose()(rhs))
        res2 = MatMul()(Transpose()(lhs), out_grad)
        summation_indices = [i for i in range(0, len(res1.shape) - len(lhs.shape))]
        if summation_indices:
            res1 = Summation(tuple(summation_indices))(res1)

        summation_indices = [i for i in range(0, len(res2.shape) - len(rhs.shape))]
        if summation_indices:
            res2 = Summation(tuple(summation_indices))(res2)
        return res1, res2
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Negate()(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp = node.inputs[0]
        result = out_grad * 1.0/inp
        return result
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp = node.inputs[0]
        return Exp()(inp) * out_grad
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return (a > 0) * a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp = node.inputs[0].realize_cached_data()
        return Tensor(inp > 0) * out_grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
# [1,2,3,4]
# 0, 2 => 2, 4

def logsumexp(x, axes):
    max_val = array_api.max(x.numpy(), axis=axes)
    requried_shape = list(x.shape)
    if axes is not None:
        for axis in axes:
            requried_shape[axis] = 1
        max_val = max_val.reshape(tuple(requried_shape))
    max_val = needle.Tensor(max_val)
    result = log(summation(exp(x + (-1) * max_val.broadcast_to(x.shape)), axes))
    return result + max_val.reshape(result.shape)

