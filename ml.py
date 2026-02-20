# %%
import numpy as np
from stats import mean_variance
from linalg import vector_dot_vector, norm_2, matrix_mean
import math
import matplotlib.pyplot as plt
from typing import List, Tuple

# %%
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    '''
    y = X @ theta => X.T @ y = (X.T @ X) @ theta
    => theta = (X.T @ X) ^ -1 @ X.T @ y 
    which is the normal equation
    '''
    X = np.array(X)
    y = np.array(y)
    theta = np.round(np.linalg.inv(X.T @ X) @ X.T @ y, 4)
    return theta

# %%
def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error between two arrays.

    MAE is a measure of the average magnitude of errors between predicted and actual values

    Parameters:
        y_true (numpy.ndarray): Array of true values
        y_pred (numpy.ndarray): Array of predicted values

    Returns:
        float: Mean Absolute Error
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return (1/(len(y_true))) * np.sum(np.abs(y_true-y_pred))

# %%
def mse(y_true, y_pred):
    '''
    MSE can be thought as the variance of the residuals.

    `MSE = resid.var() + resid.mean()**2`
    '''
    return (1/(len(y_true))) * np.sum((y_true-y_pred)**2)  # replace 2 * (len(y_true))) to make the grad easier

# %%
def rmse(y_true, y_pred):
    '''
    RMSE is a commonly used metric for evaluating the accuracy of regression models, 
    providing insight into the standard deviation of residuals.

    It is the standard deviation of the residuals if the residuals have zero mean i.e.

    `RMSE = sqrt(resid.var() + resid.mean()**2)`
    '''
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if y_true.shape != y_pred.shape:
        raise ValueError("Mismatched array shapes.")
    if y_true.size == 0:
        raise ValueError("Empty arrays.")
    if not np.issubdtype(y_true.dtype, np.number):
        raise TypeError("Invalid input types.")

    return np.sqrt(mse(y_true, y_pred))


# %%
def ridge_loss(
        X: np.ndarray,
        w: np.ndarray,
        y_true: np.ndarray,
        alpha: float
) -> float:
    '''
    Ridge Regression is a linear regression method with a regularization term
    to prevent overfitting **by controlling the size of the coefficients**.

    **Regularization:** Adds a penalty to the loss function to discourage large coefficients, 
    helping to generalize the model.

    **Penalty Term:** The sum of the squared coefficients, scaled by the regularization parameter λ, 
    which controls the strength of the regularization.

    In simple terms, the RIDGE LOSS is MSE with a regularization term.
    '''
    y_pred = X @ w 
    return mse(y_true, y_pred) + (alpha * sum(w**2))

# %% 
def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    https://www.deep-ml.com/problems/283
    Compute the average hinge loss for SVM classification.

    Args:
        y_true: Array of true labels (-1 or +1)
        y_pred: Array of predicted scores (raw SVM scores)

    Returns:
        Average hinge loss rounded to 4 decimal places
    """
    val = np.mean(np.maximum(0, 1-(y_pred*y_true)))
    return round(val, 4)



# %%
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # Initialize weights to zeros
    losses = []

    # Your code here: implement gradient descent
    for i in range(iterations):
        # forward 
        y_pred = X @ theta # (m,n) * (n,1) = (m,1)   
        loss = mse(y, y_pred)
        if i%100 ==0: 
            print(loss)
        losses.append(loss) 
        # backprop
        grad = 1/m * (X.T @ (y_pred-y))
        # optimization step
        theta = theta - alpha * grad 
    return theta.flatten()

# %%
def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ''' 
    Standardization rescales the feature to have a mean of 0 and a standard deviation of 1. 

    Min-max normalization rescales the feature to a range of [0, 1], where the minimum feature
    value maps to 0 and the maximum to 1.
    '''
    standardized_data = (data - data.mean(axis=0)[None,:]) / data.std(axis=0)
    normalized_data = (data - data.min(axis=0)[None,:]) / (data.max(axis=0)[None,:] - data.min(axis=0)[None,:])
    return standardized_data, normalized_data

# %%
def sigmoid(z: float|np.ndarray) -> np.ndarray:
    z = np.array(z)[None]
    result = np.array([1 / (1+np.exp(-x)) for x in z])
    return result.flatten()

# %% 
def tanh(x: float) -> float:
    """
    https://www.deep-ml.com/problems/264
    Implements the Tanh (hyperbolic tangent) activation function.

    Args:
        x (float): Input value

    Returns:
        float: The tanh of the input, rounded to 4 decimal places
    """
    val = ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))) if x != 0. else 0.
    return round(val, 4)

# %% 
def hard_sigmoid(x: float) -> float:
    """
    Implements the Hard Sigmoid activation function.

    Args:
        x (float): Input value

    Returns:
        float: The Hard Sigmoid of the input
    """
    if x<2.5 and x>-2.5:
        return .2*x+0.5
    return 0. if x <= -2.5 else 1.

# %%

# Visualize the difference between hard and regular sigmoid
# x = np.linspace(-5,5,1000)
# y = []
# y_ = []
# for p in x: 
#     y.append(hard_sigmoid(p))
#     y_.append(sigmoid(p))

# plt.plot(x,y, label='hard_sigmoid') 
# plt.plot(x,y_, label='sigmoid')
# plt.legend()
# plt.show()


# %%
def softmax(scores: list[float]) -> list[float]:
    '''
    The softmax function converts a list of values into a probability distribution. 
    The probabilities are proportional to the exponential of each element divided by 
    the sum of the exponentials of all elements in the list.
    '''
    result = [math.exp(z-max(scores))/sum([math.exp(z_-max(scores)) for z_ in scores]) for z in scores]

    return result

# %%
def log_softmax(scores: list) -> np.ndarray:
    '''
    Directly applying the logarithm to the softmax function can lead 
    to numerical instability, especially when dealing with large numbers.
    
    This formulation helps to avoid overflow issues that can occur when 
    exponentiating large numbers.

    The log-softmax function is particularly useful in machine learning for 
    calculating probabilities in a stable manner, especially when used with 
    cross-entropy loss functions. IT DOESN'T YIELD A PROBABILITY DISTR. ANY MORE
    '''
    return [x - max(scores) -np.log(sum([np.exp(y - max(scores)) for y in scores])) for x in scores]

# %%
def relu(z: float) -> float:
    return max([0., z])

# %% 
def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
    '''
    It addresses the "dying ReLU" problem by allowing a small, 
    non-zero gradient when the input is negative.

    It prevents neurons from becoming inactive.
    '''
    return z if z>0 else alpha * z

# %%
def single_neuron_model(
        features: list[list[float]],
        labels: list[int],
        weights: list[float],
        bias: float
) -> tuple[list[float], float]:
    # Your code here
    features = np.array(features)
    labels = np.array(labels)
    weights = np.array(weights)
    # forward
    probabilities = sigmoid((features @ weights) + bias)

    # preds = np.array([1 if probability > 0.5 else 0 for probability in probabilities]) # this is wrong apparently
    
    # loss calculation
    mse = mse(labels, probabilities)

    return probabilities, mse

# %% 
def shuffle_data(X, y, seed=None):
    '''
    We will use a random permutation to permute the indices of X and y
    '''
    np.random.seed(seed=seed)
    permutation = np.random.choice(X.shape[0], X.shape[0], replace=False)
    return (X[permutation],y[permutation])

# %%
def batch_iterator(X, y=None, batch_size=64):
    np.ceil(X.shape[0]/batch_size)
    batch_X = np.array_split(X,np.ceil(X.shape[0]/batch_size))
    if not (y is None):
        batch_y = np.array_split(y,np.ceil(X.shape[0]/batch_size))
        return list(zip(batch_X, batch_y))
    return batch_X


# %%
def accuracy_score(y_true, y_pred):
    correct = [y_true[i]==y_pred[i] for i in range(len(y_true))]
    return sum(correct) / len(y_true)

# %%
def tp_fp_tn_fn(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true==1) & (y_pred==1))
    fp = np.sum((y_true==0) & (y_pred==1))
    tn = np.sum((y_true==0) & (y_pred==0))
    fn = np.sum((y_true==1) & (y_pred==0))
    return tp, fp, tn, fn    


# %%
def precision(y_true, y_pred):
    '''
    Accuracy of the positive predictions made by the model.

    How good is the model at finding the positives?
    i.e. how many of the positive predictions were actually 
    positive

    `Precision = TP / TP + FP`
    '''
    tp = tp_fp_tn_fn(y_true, y_pred)[0]
    fp = tp_fp_tn_fn(y_true, y_pred)[1]

    return tp / (tp+fp) if (tp+fp) else 0.

# %%
def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.

    How many positives did I get out of all the positives?
    i.e. how many positives did I find out of the predicted 
    positives (tp) and the false negatives (fn)

    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)

    Returns:
        Recall value as a float
    """
    tp = tp_fp_tn_fn(y_true, y_pred)[0]
    fn = tp_fp_tn_fn(y_true, y_pred)[3]
    return tp / (tp+fn) if (tp+fn) else 0.

# %%
def specificity(y_true, y_pred):
    """
    Calculate the specificity metric for binary classification.

    How well are we labeling the negative elements correctly? 
    i.e. ` TN / (TN + FP) `

    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)

    Returns:
        Recall value as a float
    """
    tn = tp_fp_tn_fn(y_true, y_pred)[2]
    fp = tp_fp_tn_fn(y_true, y_pred)[1]
    return tn / (tn+fp) if (tn+fp) else 0.

# %%
def negative_predictive_value(y_true, y_pred):
    """
    Calculate the negative predictive value metric for binary classification.

    How many elements labeled as negative are actually negative?
    i.e. ` TN / (TN + FN) `

    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)

    Returns:
        Recall value as a float
    """
    tn = tp_fp_tn_fn(y_true, y_pred)[2]
    fn = tp_fp_tn_fn(y_true, y_pred)[3]
    return tn / (tn+fn) if (tn+fn) else 0.

# %%
def f1_score(y_true, y_pred):
    """
    Calculate the F1 score based on true and predicted labels.

    Args:
        y_true (list): True labels (ground truth).
        y_pred (list): Predicted labels.

    Returns:
        float: The F1 score rounded to three decimal places.

    This is equivalent two the dice score, which can also be 
    computed with another way.
    """
    pr = precision(y_true, y_pred)
    re = recall(y_true, y_pred)
    f1 =2 * (pr*re)/(pr+re) if (pr+re)>0 else 0.
    return round(f1,3)

# %%
def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.

    F-Score, also called F-measure, is a measure of predictive 
    performance that's calculated from the Precision and Recall 
    metrics. The F-score applies additional weights, valuing 
    one of precision or recall more than the other

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    pr = precision(y_true, y_pred)
    re = recall(y_true, y_pred)
    return (1+beta**2) * (pr*re)/((beta**2 * pr)+re) if (pr+re)>0 else 0

# %%
def kernel_function(x1, x2):
	return vector_dot_vector(x1, x2)

# %%
def gini_impurity(y):
    """
    Calculate Gini Impurity for a list of class labels.

    Gini Impurity checks how often a randomly selected sample would 
    be mislabeled if assigned by class probability. 
    It is computationally simple and used in tree-based classifiers.
    The lower the better since this means that there is a low prob in 
    miss-classifying an item, if we classify it by class probability. 

    Learn more about the Gini Index:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places

    In the context of decision trees we decide the separating question 
    based in creating the smallest possible accumulation gini index (at 
    least for the DecisionTreeClassifier problem)
    """
    unique, counts = np.unique(y, return_counts=True)
    # counter = dict(zip(unique, counts)) # this could be useful instead of doing it with Counter package
    probs = counts/len(y)
    val = 1 - sum(probs**2)
    return round(val,3)

# %%
def cosine_similarity(v1, v2):
    norm1 = norm_2(v1)
    norm2 = norm_2(v2)

    if v1.shape != v2.shape:
        raise ValueError('Mismatched shape.')
    if (v1.shape == 0) or (v2.shape == 0):
        raise ValueError('Vector with zero shape.')
    if norm1==0 or norm2==0:
        raise ValueError('Vector with zero magnitude.')
    
    dot_prod = vector_dot_vector(v1, v2)
    return dot_prod/(norm1*norm2)

# %% 
def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].

    Args:
        x: A list of numerical values

    Returns:
        A new list with values normalized to [0, 1]

    There is a variant of this function as part of `feature_scaling()`
    """
    return [(y-min(x))/(max(x)-min(x)) for y in x]
# %%
def model_fit_quality(training_accuracy, test_accuracy):
    """
    Determine if the model is overfitting, underfitting, or a good fit based on training and test accuracy.
    :param training_accuracy: float, training accuracy of the model (0 <= training_accuracy <= 1)
    :param test_accuracy: float, test accuracy of the model (0 <= test_accuracy <= 1)
    :return: int, one of '1', '-1', or '0'.
    """
    if (training_accuracy - test_accuracy) > 0.2:
        return 1
    if (training_accuracy + test_accuracy) < 1.4:
        return -1
    return 0

# %%
def jaccard_index(y_true, y_pred):
    '''
    The Jaccard Index, also known as the Jaccard Similarity Coefficient, 
    is a statistic used to measure the similarity between sets.

    Usage in Machine Learning

    The Jaccard Index is particularly useful in:

    - Evaluating clustering algorithms.
    - Comparing binary classification results.
    - Document similarity analysis.
    - Image segmentation evaluation.

    In this implementation we only consider binary inputs
    '''
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    intersection = (y_true & y_pred).sum()
    union = (y_true | y_pred).sum()
    return 0.0 if union == 0 else round(intersection / union, 3)

# %%
def confusion_matrix(y_true, y_pred):

    # y_true = [point[0] for point in data]
    # y_pred = [point[1] for point in data]

    tp, fp, tn, fn = tp_fp_tn_fn(y_true, y_pred)
    return np.array([[tp, fp],
                     [fn, tn]])

# %%
def OSA(source: str, target: str) -> int:
	# Your code here
	pass

# %%
def euclid_distance(
        x: tuple,
        y: tuple
) -> float:
    return sum([(x[i]-y[i])**2 for i in range(len(x))])**.5

# %%
def argmin(l:list):
    return l.index(min(l))

# %% 
def k_means_clustering(
        points: list[tuple[float, ...]],
        k: int,
        initial_centroids: list[tuple[float, ...]],
        max_iterations: int
) -> list[tuple[float, ...]]:
    # extra: what happens with draws ? in the following implementation the first centroid wins, but we may shuffle them to deal with that


    # all the points must have the same dimension
    if False in [len(point)==len(points[0]) for point in points+initial_centroids]:
        raise ValueError('Dimension mismatch')
    
    if k != len(initial_centroids):
        raise ValueError("k must equal number of initial centroids")
    
    # for each point in the dataset find their distance to each centroid, make a helper distance function 
    
    centroids = initial_centroids
    for _ in range(max_iterations):
        c_id_point=[]
        for point in points: 
            # distances = [d(point, centroid_1), d(point, centroid_2), ..., d(point, centroid_-1)]
            distances = []
            for centroid in centroids:
                distances.append(euclid_distance(point, centroid))
            c_id = argmin(distances) # centroid id/cluster of point
            c_id_point.append((c_id, point))

        # update step
        new_centroids=[]
        # calculate the mean of all points assigned to the cluster

        for id_c, centroid in enumerate(centroids):
            cp_coordinates = []
            for c_id, point in c_id_point:
                if c_id == id_c:
                    cp_coordinates.append(list(point))
            if len(cp_coordinates)==0: # in case of empty clusters
                new_centroids.append(centroid) # take the previous centoid
            else:
                # move the centroid to this position 
                mean_coordinates = matrix_mean(cp_coordinates, axis=1)
                new_centroids.append(tuple(mean_coordinates))
        centroids = list(new_centroids)

    # stopping criteria:

    # centroids no longer move a lot (threshold)
    
    return centroids # round to 4 decimals


# %% 
def r_squared(y_true, y_pred):
    '''
    R-squared, also known as the coefficient of determination, is a statistical measure 
    that represents the proportion of the variance for a dependent variable that's explained 
    by an independent variable or variables in a regression model. It provides insight into 
    how well the model fits the data.

    e.g. if the R-squared value is calculated to be 0.989, 
    indicating that the regression model explains 98.9% of the variance in the dependent variable.
    
    It is calculated as ` R^2 = 1 - RSS / SST` where
    - RSS: sum of squared residuals (it's a scaled version of the MSE)
    - SST: total sum of squares (is a scaled version of the variance)
    '''
    RSS = mse(y_true, y_pred) * len(y_true)
    SST = mean_variance(y_true)[1] * len(y_true)
    return 1 - (RSS / SST)


# %%
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    '''
    Args:
        actual (list): y_true
        predicted (list): y_pred

    Returns:
        tuple:
            - confusion_matrix: A 2x2 matrix.
            - accuracy: A float representing the accuracy of the model.
            - f1_score: A float representing the F1 score of the model.
            - specificity: A float representing the specificity of the model.
            - negative_predictive_value: A float representing the negative predictive value.

    '''
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return (confusion_matrix(actual, predicted),
            round(accuracy_score(actual, predicted), 3),
            round(f1_score(actual, predicted), 3),
            round(specificity(actual, predicted), 3),
            round(negative_predictive_value(actual, predicted), 3))

# %%
'''
Function is deprecated since it add unessecary complexity, 
there is already `np.array_split()` that does the same as split 
but with automatic distribution to the remainders.

def make_folds(x, n_samples, k):
    x = np.asarray(x)
    equal_folds = np.split(x[:n_samples-(n_samples%k)], k)
    res = list(x[n_samples-(n_samples%k):])

    if len(res)==0:
        equal_folds = [list(fold) for fold in equal_folds]
        return equal_folds
    
    res = np.split(res,len(res))
    print(res)

    folds = []
    for id, r in enumerate(res):
        new_fold = np.concat((equal_folds[id], r))
        folds.append(new_fold.tolist())

    folds.extend(equal_folds[len(res):])
    return folds
    
'''

# %%
def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.

    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)

    Returns:
        List of (train_indices, test_indices) tuples
    """
    
    # Generate indices from 0 to n_samples-1.
    indices = np.arange(n_samples)
    # Shuffle the indices if required.
    if shuffle==True:
        np.random.shuffle(indices)
    # Split the indices into k roughly equal folds (if n_samples is not divisible by k, distribute the remainder to the first folds).
    # folds = make_folds(indices, n_samples, k) # instead of using the complicated manual function we do the following
    folds = np.array_split(indices, k)
    # For each fold i, use fold i as the test set and combine all other folds as the training set.
    k_folds = []
    '''    
    We refactor this code to make it simpler, using the fact 
    that arrays indices never get out of bounds.

    for i, fold in enumerate(folds):
        test = fold
        if i+1 < len(folds):
            train = np.array(folds[:i] + folds[i+1:]).flatten().tolist()
            k_folds.append((train, test))
        else: 
            train = np.array(folds[:i]).flatten().tolist()
            k_folds.append((train, test)) 
    '''

    for i in range(k):
        test = folds[i].tolist()
        train = np.concatenate(folds[:i]+folds[i+1:]).tolist()
        k_folds.append((train, test))

    # Return the list of (train_indices, test_indices) tuples.
    return k_folds

# %%
def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> tuple[np.ndarray, float, list[float]]:
    
    weights = np.asarray(initial_weights)
    bias = np.asarray(initial_bias)[None]
    features = np.asarray(features)
    # merge weights and biases into one matrix
    parameters = np.concatenate((bias, weights))
    padding = np.ones(features.shape[0])[:, None]
    features = np.concatenate((padding, features), axis=1)

    z = features @ parameters
    y_hat = sigmoid(z)

    losses = []
    
    for _ in range(epochs):
        # forward
        z = features @ parameters
        y_hat = sigmoid(z)
        # loss calculation
        loss = mse(labels, y_hat)
        losses.append(round(loss, 4))
        print(loss)
        # grad calculation
        delta = 2 * (y_hat - labels) * y_hat * (1 - y_hat)
        grad = (1/len(labels)) * features.T @ delta
        # back prop
        parameters -= learning_rate * grad
    

    return parameters[1:].round(4), parameters[0].round(4), losses


# %% 
def to_categorical(x, n_col=0):
    '''
    https://www.deep-ml.com/problems/34
    One-Hot Encoding of Nominal Values
    '''
    b = np.zeros((x.size, x.max() + 1 if (x.max()+1 > n_col) else n_col))
    b[np.arange(x.size), x] = 1
        

    return b

# %%
def divide_on_feature(X, feature_i, threshold):
    '''
    https://www.deep-ml.com/problems/31
    Docstring for divide_on_feature
    
    :param X: data
    :param feature_i: id of feature to apply threshold
    :param threshold: threshold
    '''
    return [X[X[:,0] >= threshold], X[X[:,0] < threshold]]

