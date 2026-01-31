"""
Docstring for tokenizer.base
"""
from abc import ABC, abstractmethod


class BaseKempModel(ABC):
    """Abstract base class for KEMP models"""

    @abstractmethod
    def forward(self, input_tokens):
        """Forward pass to get model predictions"""
        pass

    @abstractmethod
    def train_step(self, input_tokens, target_token, learning_rate: float) -> float:
        """Perform a single training step and return the loss"""
        pass

    @abstractmethod
    def predict(self, input_tokens):
        """Predict the next token given input tokens"""
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save the model weights to a file"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str):
        """Load model weights from a file and return an instance"""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the size of the vocabulary"""
        pass

    @property
    @abstractmethod
    def emb_dim(self) -> int:
        """Return the embedding dimension"""
        pass

    @property
    @abstractmethod
    def max_len(self) -> int:
        """Return the maximum sequence length"""
        pass    

