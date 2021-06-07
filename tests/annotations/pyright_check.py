from typing import Protocol, TypeVar, Type

T1 = TypeVar("T1", covariant=True)
T2 = TypeVar("T2")


class A(Protocol[T1]):
    def __init__(self) -> None:
        ...

def f(x: T2) -> Type[A[T2]]:
    ...

a_class = f(1)
a_instance: A[int] = reveal_type(a_class())

