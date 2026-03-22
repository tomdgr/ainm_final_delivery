_model_registry = {}
_model_meta = {}


def register_model(name):
    def decorator(cls):
        _model_registry[name] = cls
        _model_meta[name] = {}
        return cls
    return decorator


def get_model(name):
    if name not in _model_registry:
        raise ValueError(f"Model '{name}' not found. Available: {list(_model_registry.keys())}")
    return _model_registry[name]


def list_models():
    return list(_model_registry.keys())
