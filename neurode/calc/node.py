from typing import Callable, Any
from builtins import min as min_, max as max_


class CalcNode:
    def __call__(self, data: dict[str, Any]):
        raise NotImplementedError("this method must be implemented in subclasses")

    def __add__(self, other):
        other = self.process_other(other)
        return Operator(self, other, lambda a, b: a + b)

    def __sub__(self, other):
        other = self.process_other(other)
        return Operator(self, other, lambda a, b: a - b)

    def __mul__(self, other):
        other = self.process_other(other)
        return Operator(self, other, lambda a, b: a * b)

    def __truediv__(self, other):
        other = self.process_other(other)
        return Operator(self, other, lambda a, b: a / b)

    def __floordiv__(self, other):
        other = self.process_other(other)
        return Operator(self, other, lambda a, b: a // b)

    def __mod__(self, other):
        other = self.process_other(other)
        return Operator(self, other, lambda a, b: a % b)

    def __pow__(self, other):
        other = self.process_other(other)
        return Operator(self, other, lambda a, b: a**b)

    def __radd__(self, other):
        other = self.process_other(other)
        return Operator(other, self, lambda a, b: a + b)

    def __rsub__(self, other):
        other = self.process_other(other)
        return Operator(other, self, lambda a, b: a - b)

    def __rmul__(self, other):
        other = self.process_other(other)
        return Operator(other, self, lambda a, b: a * b)

    def __rtruediv__(self, other):
        other = self.process_other(other)
        return Operator(other, self, lambda a, b: a / b)

    def __rfloordiv__(self, other):
        other = self.process_other(other)
        return Operator(other, self, lambda a, b: a // b)

    def __rmod__(self, other):
        other = self.process_other(other)
        return Operator(other, self, lambda a, b: a % b)

    def __rpow__(self, other):
        other = self.process_other(other)
        return Operator(other, self, lambda a, b: a**b)

    @staticmethod
    def process_other(other):
        return other if isinstance(other, CalcNode) else Value(other)

    @property
    def placeholders(self) -> list["Placeholder"]:
        raise NotImplementedError("this method must be implemented in subclasses")


def min(a: CalcNode, b: CalcNode):
    a = CalcNode.process_other(a)
    b = CalcNode.process_other(b)
    return Operator(a, b, lambda a, b: min_(a, b))


def max(a: CalcNode, b: CalcNode):
    a = CalcNode.process_other(a)
    b = CalcNode.process_other(b)
    return Operator(a, b, lambda a, b: max_(a, b))


class Value(CalcNode):
    def __init__(self, val) -> None:
        super().__init__()
        self.val = val

    def __call__(self, data: dict[str, Any]):
        return self.val

    @property
    def placeholders(self) -> list["Placeholder"]:
        return []


class Placeholder(CalcNode):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __call__(self, data: dict[str, Any]):
        return data[self.name]

    @property
    def placeholders(self) -> list["Placeholder"]:
        return [self]


class Operator(CalcNode):
    def __init__(
        self, left: CalcNode, right: CalcNode, operate_fn: Callable[[Any, Any], Any]
    ) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.operate_fn = operate_fn

    def __call__(self, data: dict[str, Any]):
        left_val = self.left(data)
        right_val = self.right(data)
        return self.operate_fn(left_val, right_val)

    @property
    def placeholders(self) -> list[Placeholder]:
        return self.left.placeholders + self.right.placeholders
