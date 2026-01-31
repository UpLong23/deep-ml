# %% 
import numpy as np

# %% 
def batch_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray = None,
    running_var: np.ndarray = None,
    momentum: float = 0.1,
    epsilon: float = 1e-5,
    training: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    https://www.deep-ml.com/problems/115
    Perform Batch Normalization on BCHW input.

    Args:
        X: Input array of shape (B, C, H, W)
        gamma: Scale parameter of shape (1, C, 1, 1)
        beta: Shift parameter of shape (1, C, 1, 1)
        running_mean: Running mean for inference, shape (1, C, 1, 1)
        running_var: Running variance for inference, shape (1, C, 1, 1)
        momentum: Momentum for updating running statistics
        epsilon: Small constant for numerical stability
        training: If True, use batch statistics; if False, use running statistics

    Returns:
        Tuple of (normalized_output, updated_running_mean, updated_running_var)
    """
    if (running_mean is None):
        running_mean = np.zeros((1,X.shape[1],1,1))
    if (running_var is None):
        running_var = np.ones((1,X.shape[1],1,1))

    if training:
        # calculate statistics
        mean_per_channel = X.mean(axis=(0,2,3), keepdims=True)
        var_per_channel = X.var(axis=(0,2,3), keepdims=True)
        # normalize
        X_hat = (X - mean_per_channel) / np.sqrt((var_per_channel+epsilon))

        running_mean = (momentum * running_mean) + ((1-momentum) * mean_per_channel)
        running_var = (momentum * running_var) + ((1-momentum) * var_per_channel)

    else: 
        X_hat = (X - running_mean) /np.sqrt((running_var+epsilon))

    # scale and shift
    y = (gamma*X_hat) + beta

    return (y, running_mean, running_var)



# %%
B, C, H, W = 2, 2, 2, 2 
np.random.seed(42) 
X = np.random.randn(B, C, H, W) 
gamma = np.ones(C).reshape(1, C, 1, 1) 
beta = np.zeros(C).reshape(1, C, 1, 1) 
running_mean = np.zeros((1, C, 1, 1)) 
running_var = np.ones((1, C, 1, 1)) 
output, rm, rv = batch_normalization(X, gamma, beta, running_mean, running_var, momentum=0.1, training=True) 
# print(np.round(output, 5)) 
print(np.round(rm, 5)) 
print(np.round(rv, 5))