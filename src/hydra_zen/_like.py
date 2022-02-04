from typing import TypeVar, cast

T = TypeVar("T")


class _Tracker:
    def __init__(self, origin, tracks=None):
        self.origin = origin

        self.tracker = [] if tracks is None else tracks.copy()

    def __repr__(self) -> str:
        base = f"Like({repr(self.origin)})"
        for item in self.tracker:
            if isinstance(item, tuple):
                _, args, kwargs = item
                contents = ""
                if args:
                    contents += ", ".join(repr(x) for x in args)
                if kwargs:
                    if args:
                        contents += ", "
                    contents += ", ".join((f"{k}={v}" for k, v in kwargs.items()))

                base += f"({contents})"
            else:
                base += f".{item}"
        return base

    def __call__(self, *args, **kwargs):
        return _Tracker(self.origin, self.tracker + [("__call__", args, kwargs)])

    def __getattr__(self, name):
        # IPython will make attribute calls on objects; we don't want to track these.
        if name != "_ipython_canary_method_should_not_exist_" and name != "__wrapped__":
            return _Tracker(self.origin, self.tracker + [name])
        return self


def like(obj: T) -> T:
    return cast(T, _Tracker(obj))
