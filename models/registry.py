"""
Credit for the initial implementation of the @Model decorating system to Narayan Schuetz, University of Bern
"""

MODEL_REGISTRY = {}

def Model(*args, **kwargs):
    """Decorator function, makes model definition a bit more obvious than relying on python's underscore variant"""

    if len(args) == 1 and callable(args[0]):
        cls = args[0]
        name = cls.__name__
        MODEL_REGISTRY[name] = cls
        return cls

    else:
        name = kwargs.get("name")

        if name is None:
            raise ValueError("Invalid argument, requires keyword argument 'name' if argument is given!")

        def wrapped_decorator(cls):
            MODEL_REGISTRY[name] = cls
            return cls

        return wrapped_decorator