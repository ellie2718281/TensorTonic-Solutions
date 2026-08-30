import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, gamma: np.ndarray,
               beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    normalized = (x - mean) / np.sqrt(var + eps)

    return gamma * normalized + beta


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:

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

        scores = Q_head @ K_head.swapaxes(-2, -1)
        scores = scores / np.sqrt(d_k)

        weights = softmax(scores, axis=-1)

        head_output = weights @ V_head
        heads.append(head_output)

    concat = np.concatenate(heads, axis=-1)

    return concat @ W_o


def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:

    hidden = x @ W1 + b1

    hidden = np.maximum(0, hidden)  # ReLU

    return hidden @ W2 + b2


def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:

    # 1. Self-attention
    attention_output = multi_head_attention(
        x, x, x,
        W_q, W_k, W_v, W_o,
        num_heads
    )

    # 2. Residual connection + LayerNorm
    x = layer_norm(
        x + attention_output,
        gamma1,
        beta1
    )

    # 3. Feed-forward network
    ff_output = feed_forward(
        x,
        W1, b1,
        W2, b2
    )

    # 4. Residual connection + LayerNorm
    x = layer_norm(
        x + ff_output,
        gamma2,
        beta2
    )

    return x