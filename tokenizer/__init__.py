from .base import BaseKempModel
from .tokenizergpt import TokenizerGPT35
from .toy import ToyTokenizer

def create_tokenizer(config: dict, *args, **kwargs):
    """Factory function to create a tokenizer instance based on config"""
    tokenizer_type = config.get('tokenizer_type', 'gpt3.5')
    if tokenizer_type == 'toy':
        return ToyTokenizer(*args, **kwargs)
    elif tokenizer_type == 'gpt3.5':
        return TokenizerGPT35('r50k_base', *args, **kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")