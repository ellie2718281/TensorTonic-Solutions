import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    position_encoding = np.zeros((seq_length, d_model))
    for pos in range(seq_length):
        for i in range(0, d_model, 2):
            angle = pos/10000**(i/d_model)
            position_encoding[pos, i] = np.sin(angle)
            position_encoding[pos, i+1] = np.cos(angle)
    return position_encoding