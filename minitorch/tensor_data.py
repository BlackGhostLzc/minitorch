from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """
    # shape (2,3,4)       strides = [1,4,12]
    # [1][2][3]
    assert len(index) == len(strides)
    position = 0
    for i in range(len(index)):
        position += index[i] * strides[i]
    return position


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # position -> index
    # shape (2,3,4)    strides = (1,4,12)    
    # position = 17    
    for i in range(len(shape) - 1, -1, -1):
        dim_size = shape[i]
        out_index[i] = ordinal % dim_size
        ordinal = ordinal // dim_size



def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None
    """
    # 大张量的维度不能少于小张量
    assert len(big_shape) >= len(shape)

    # 检查索引长度是否匹配对应的形状
    assert len(big_index) == len(big_shape)
    assert len(out_index) == len(shape)

    # 检查广播规则兼容性，右对齐
    # big_shape (3,4,5)     shape (1,5)
    # big_index (2,3,2)    
    # return [0,2]
    offset = len(big_shape) - len(shape)
    for i in range(len(shape)):
        big_dim = big_shape[i + offset]
        small_dim = shape[i]
        # 规则：小维度必须是 1，或者两者必须相等
        assert small_dim == 1 or small_dim == big_dim, \
            f"Broadcasting mismatch at dim {i} (original) / {i+offset} (big): " \
            f"shape {shape} cannot be broadcast to {big_shape}. " \
            f"Dimension {small_dim} is not 1 and does not match {big_dim}."

    for i in range(len(shape)):
        if shape[i] == 1:
            # 原始维度是 1 (被广播拉伸了)
            # 无论大张量的索引是多少，回到原始数据里，它都在位置 0
            out_index[i] = 0
        else:
            # 原始维度 > 1 (没有被拉伸)
            # 直接取大张量对应位置的索引
            out_index[i] = big_index[i + offset]




def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    a, b = shape1, shape2
    m = max(len(a), len(b))
    
    # 2. 左侧补 1 (Padding) 以便对齐
    # (2, 3) -> (1, 2, 3)
    if len(a) < m:
        a = (1,) * (m - len(a)) + a
    if len(b) < m:
        b = (1,) * (m - len(b)) + b
    
    # 3. 逐位比较并构建结果形状
    out_shape = []
    for i in range(m):
        d1 = a[i]
        d2 = b[i]
        
        if d1 == d2:
            out_shape.append(d1)
        elif d1 == 1:
            out_shape.append(d2)
        elif d2 == 1:
            out_shape.append(d1)
        else:
            # 如果既不相等，也没人是 1，说明冲突了
            raise IndexingError(f"Shapes {shape1} and {shape2} are not broadcastable")
            
    return tuple(out_shape)


def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # 只需要改变一下 shape 和 stride，不需要修改 storage
        new_shape = tuple(self.shape[i] for i in order)
        new_strides = tuple(self.strides[i] for i in order)
        return TensorData(self._storage, new_shape, new_strides)
       

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
