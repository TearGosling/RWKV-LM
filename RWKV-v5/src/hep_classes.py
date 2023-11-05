"""Classes for Hephaestus."""
import math
import torch
import torch.nn.functional as F

from torch import nn

########################
# Activation functions #
########################
# Not all of these will be used, they're here for testing purposes

class DecayingSineUnit(nn.Module):
    """
    The Decaying Sine Unit from "Biologically Inspired Oscillating Activation Functions Can Bridge the Performance Gap between Biological and Artificial Neurons"
    https://arxiv.org/abs/2111.04020
    """
    def __init__(self, args) -> None:
        super().__init__()

    def forward(self, x):
        return (math.pi / 2) * (torch.sinc(x - math.pi) - torch.sinc(x + math.pi))

class ElephantActivation(nn.Module):
    def __init__(self, args) -> None:
        """
        Elephant activation function, from Elephant Neural Networks: Born to Be a Continual Learner
        https://arxiv.org/abs/2310.01365

        Args:
        a (float): the width of the function. Hyperparameter
        d (float): the slope of the function. Hyperparameter with a default of 4 (as in the paper)
        """
        super().__init__()
        self.a = args.elephant_a
        self.d = args.elephant_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.abs(x / self.a).pow(self.d))
    
class GeluActivation(nn.Module):
    def __init__(self, args) -> None:
        """
        Wrapper around the GELU activation function.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)
    
class GrowingCosineUnit(nn.Module):
    def __init__(self, args) -> None:
        """
        The Growing Cosine Unit from "Biologically Inspired Oscillating Activation Functions Can Bridge the Performance Gap between Biological and Artificial Neurons"
        https://arxiv.org/abs/2111.04020
        """
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.cos()
    
class MishActivation(nn.Module):
    def __init__(self, args) -> None:
        """
        Wrapper around the Mish activation function.
        """
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.mish(x)
    
class ReluSquaredActivation(nn.Module):
    def __init__(self, args) -> None:
        """
        Wrapper around the ReLU^2 activation function.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) ** 2

class RotaryActivation(nn.Module):
    def __init__(self, args) -> None:
        """
        Gated activation function based on rotary embeddings.
        Idea for dropping half: instead of every other value being always
        dropped, add nn.Dropout() so model doesn't learn to assign
        dropped values to just nothing.
        """
        super().__init__()
        self.drop_half = args.rotary_act_drop_half

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates half the hidden dims of the input.
        Implementation taken from modeling_llama.py:
        https://github.com/huggingface/transformers/blob/21dc5859421cf0d7d82d374b10f533611745a8c5/src/transformers/models/llama/modeling_llama.py#L200
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_half:
            # Drop every other value of cos and sin
            cos = (x * x.cos())[..., 0::2]
            sin = (self.rotate_half(x) * x.sin())[..., 1::2]
        else:
            # Chunk into two and operate on the halves
            x1, x2 = torch.chunk(x, 2, dim=-1)
            cos = x1 * x1.cos()
            sin = self.rotate_half(x2) * x2.sin()
        return torch.cat((cos, sin), dim=-1)
    
class SigmoidActivation(nn.Module):
    def __init__(self, args) -> None:
        """
        Wrapper around the sigmoid activation function.
        """
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.sigmoid(x)
    
class SwishActivation(nn.Module):
    def __init__(self, args) -> None:
        """
        Wrapper around the SiLU activation function (also known as swish).
        """
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)
    
# Mapping for activations
ACT2CLS = {
    "dsu": DecayingSineUnit,
    "elephant": ElephantActivation,
    "gelu": GeluActivation,
    "gcu": GrowingCosineUnit,
    "mish": MishActivation,
    "relu2": ReluSquaredActivation,
    "rotary": RotaryActivation,
    "sigmoid": SigmoidActivation,
    # Use both alias for swish to reduce silly errors
    "silu": SwishActivation,
    "swish": SwishActivation,
}
