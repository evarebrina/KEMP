"""
Base model interface for KEMP
All implementations must follow this interface.
"""
from abc import ABC, abstractmethod


class BaseKempModel(ABC):
    """Abstract base class for the model"""

    @abstractmethod
    def forward(self, input_tokens: list[int]) -> tuple[list[float], dict[str, any], any]:
        
        """Forward pass - returns logits for next token prediction"""
        pass

    @abstractmethod
    def predict(self, last_embedding: list[float]) -> list[float]:
        """
        Predict next token ID given input the dense 
        representation of the max_len-1 tokens in its last 
        embedding
        """
        pass