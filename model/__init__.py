"""
Model implementations for KEMP
"""
from .base import BaseKempModel

def create_model(vocab_size: int, config: dict) -> BaseKempModel:
    """Factory function to create a KEMP model instance based on config"""
    # For now, always return the pure Python implementation
    model_type = config.get('model_type', 'pure_python')

    if model_type == 'pure_python':
        # Delayed import to avoid circular dependency
        from .pure_python import SimpleKEMPModel
        return SimpleKEMPModel(
            vocab_size=vocab_size,
            config=config
        )
    elif model_type == 'pytorch':
        # Delayed import to avoid circular dependency
        from .pytorch import PyTorchKemp
        return PyTorchKemp(
            vocab_size=vocab_size,
            emb_dim=config['emb_dim'],
            max_len=config['max_len']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
