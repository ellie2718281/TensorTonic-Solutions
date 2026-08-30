import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]

    scores = Q @ K.swapaxes(-2, -1)
    scores = scores / np.sqrt(d_k)

    weights = softmax(scores, axis=-1)

    return weights @ V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    d_model = Q.shape[-1]
    d_k = d_model // num_heads
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    heads = []
    for i in range(0, d_model, d_k):
        Q_head = Q_proj[..., i:i+d_k]
        K_head = K_proj[..., i:i+d_k]
        V_head = V_proj[..., i:i+d_k]
        head_output = scaled_dot_product_attention(Q_head, K_head, V_head)
        heads.append(head_output)
    concat = np.concatenate(heads, axis = -1)
    return concat @ W_o
        