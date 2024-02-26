from typing import Callable, Any


class CalcNode:
    def __call__(self, data: dict[str, Any]):
        raise NotImplementedError

    def __add__(self, other):
        other = self.__process_other(other)
        return Operator(self, other, lambda a, b: a + b)

    def __sub__(self, other):
        other = self.__process_other(other)
        return Operator(self, other, lambda a, b: a - b)

    def __mul__(self, other):
        other = self.__process_other(other)
        return Operator(self, other, lambda a, b: a * b)

    def __truediv__(self, other):
        other = self.__process_other(other)
        return Operator(self, other, lambda a, b: a / b)

    def __radd__(self, other):
        other = self.__process_other(other)
        return Operator(other, self, lambda a, b: a + b)

    def __rsub__(self, other):
        other = self.__process_other(other)
        return Operator(other, self, lambda a, b: a - b)

    def __rmul__(self, other):
        other = self.__process_other(other)
        return Operator(other, self, lambda a, b: a * b)

    def __rtruediv__(self, other):
        other = self.__process_other(other)
        return Operator(other, self, lambda a, b: a / b)

    @staticmethod
    def __process_other(other):
        return other if isinstance(other, CalcNode) else lambda _: other

    @property
    def placeholders(self):
        raise NotImplementedError


class Placeholder(CalcNode):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __call__(self, data: dict[str, Any]):
        return data[self.name]

    @property
    def placeholders(self):
        raise [self]


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
    def placeholders(self):
        return self.left.placeholders + self.right.placeholders
