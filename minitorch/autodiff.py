from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals1 = list(vals)
    vals2 = list(vals)
    vals1[arg] += epsilon
    vals2[arg] -= epsilon
    
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)    


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # 按计算图进行拓扑排序，返回Iterable[Variable]
    # 参数 variable 是子节点，要向上追溯(通过variable.back ScalarHistory)
    # is_leaf(): 判断是否是叶子节点
    # is_constant(): 判断是否是常数  constant也会以scalar的形式添加进ScalarHistory中
    
    topological_order = []
    visited = set()

    def visit(var: Variable):
        if var.unique_id in visited:
            return
        if var.is_constant():
            return
        
        visited.add(var.unique_id)

        if var.history is not None:
            for input_var in var.parents:
                visit(input_var)

        # 父母节点都添加进去了，才把自己加进去
        topological_order.append(var)

    visit(variable)

    return reversed(topological_order)



    


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    order = topological_sort(variable)
    # 创建一个字典用来记录每一个 scalar variable 的梯度
    grads = {}
    # 为根节点记录梯度，一般是1.0
    grads[variable.unique_id] = deriv
    for var in order:
        if var.unique_id not in grads:
            grads[var.unique_id] = 0.0

        d = grads[var.unique_id]
        # 叶子节点需要把梯度保留下来（需要训练进行更新的参数）
        if var.is_leaf():
            var.accumulate_derivative(d)

        # 中间节点
        else:
            if var.history is not None:
                for v, grad in var.chain_rule(d):
                    if v.is_constant():
                        continue
                    
                    if v.unique_id not in grads:
                        grads[v.unique_id] = 0.0

                    grads[v.unique_id] += grad

    

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
