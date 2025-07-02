# /JANUS-CORE/janus/architecture.py

import torch.nn as nn

class ResonanceActuator(nn.Module):
    """
    The digital Basal Ganglia. Takes an 'interference pattern' vector
    and outputs a 'Metacognitive Gain' score to arbitrate between
    synthesis and self-reflection.
    """
    def __init__(self, input_dim=384, hidden_dim_1=128, hidden_dim_2=64, output_dim=1):
        super(ResonanceActuator, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim_1)
        self.layer_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.layer_3 = nn.Linear(hidden_dim_2, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.sigmoid(self.layer_3(x))
        return x