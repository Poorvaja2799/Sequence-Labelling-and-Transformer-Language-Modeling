import os
import json
import torch
from project3_src.model import TransformerLM


def save_model(model: torch.nn.Module, config: dict, path: str, filename="transformer_lm"):
    """
    Save the model and config to a file.

    Args:
        model: The model to save.
        config: The config to save.
        path: The path to save the model and config to.
        filename: The filename to save the model and config to.
    """
    config_path = os.path.join(path, f"{filename}_config.json")
    model_path = os.path.join(path, f"{filename}.pt")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    torch.save(model.state_dict(), model_path)


def load_model(path: str, filename="transformer_lm", device: str = 'cpu'):
    """
    Load the model and config from a file.

    Args:
        path: The path to load the model and config from.
        filename: The filename to load the model and config from.
        device: The device to load the model and config on.
    Returns:
        The model and config.
    """
    config_path = os.path.join(path, f"{filename}_config.json")
    model_path = os.path.join(path, f"{filename}.pt")
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = TransformerLM(**config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, config