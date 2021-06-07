from typing import Protocol, TypeVar, Type

T = TypeVar("T")

class A(Protocol[T]):
    pass


def f(x: T) -> Type[A[T]]:
    ...

a_class = f(1)
a_instance: A[int] = reveal_type(a_class())

