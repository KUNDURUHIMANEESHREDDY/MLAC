"""Read-only inspector for an already-instantiated, trusted PyTorch module."""
from collections import Counter
from mlac.ir import Architecture, Component, TensorSpec

def inspect_torch_model(model, *, name="pytorch-model"):
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("Install PyTorch with: pip install 'mlac[torch]'") from exc
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    counts, components, tensors = Counter(), [], []
    total = trainable = 0
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        kind = type(module).__name__
        counts[kind] += 1
        config = {}
        for attr in ("in_features", "out_features", "num_features", "num_heads",
                     "hidden_size", "intermediate_size", "p", "kernel_size", "stride"):
            value = getattr(module, attr, None)
            if isinstance(value, (int, float, str, tuple)):
                config[attr] = value
        components.append(Component(module_name, kind, config))
    for param_name, param in model.named_parameters():
        total += param.numel()
        trainable += param.numel() if param.requires_grad else 0
        tensors.append(TensorSpec(param_name, tuple(int(x) for x in param.shape), str(param.dtype)))
    return Architecture(name, "pytorch_module", framework="pytorch",
        dimensions={"parameter_count": total, "trainable_parameter_count": trainable,
                    "module_count": len(components)},
        components=tuple(components), tensors=tuple(tensors),
        metadata={"module_types": dict(counts)})
