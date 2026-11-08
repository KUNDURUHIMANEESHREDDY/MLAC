"""Explicit allow-listed IR metadata mutation; not a live weight rewrite."""
from dataclasses import replace

def set_dropout_probability(architecture, *, component_name, probability):
    if not 0.0 <= probability < 1.0:
        raise ValueError("probability must be in [0, 1)")
    found, updated = False, []
    for component in architecture.components:
        if component.name == component_name:
            if component.kind not in {"Dropout", "Dropout1d", "Dropout2d", "Dropout3d"}:
                raise ValueError(f"{component_name} is not a supported dropout component")
            config = dict(component.config)
            config["p"] = probability
            updated.append(replace(component, config=config))
            found = True
        else:
            updated.append(component)
    if not found:
        raise KeyError(f"Dropout component not found: {component_name}")
    return replace(architecture, version=architecture.version + 1, components=tuple(updated))
