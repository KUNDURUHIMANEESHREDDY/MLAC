"""Small example model for the optional PyTorch inspector."""
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Dropout(.1), nn.Linear(32, 4))
    def forward(self, inputs):
        return self.network(inputs)

if __name__ == "__main__":
    from mlac.inspector import inspect_torch_model
    print(inspect_torch_model(TinyMLP(), name="tiny-mlp").to_dict())
