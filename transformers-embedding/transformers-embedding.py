import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    # codes
    # pytorch is N(0, 1), we have dims
    return nn.Embedding(vocab_size, d_model)

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # codes
    # look up raw embedding for given token
    # shape of embedding (...., d_model)
    embedded = embedding(tokens)

    # scale by sqrt of d_model
    scaled_embeddings = embedded * math.sqrt(d_model)

    return scaled_embeddings
