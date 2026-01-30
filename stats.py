# %% 
import numpy as np
from linalg import calculate_matrix_mean
import math

# %% 
def mean_variance(
	vector: list[int|float]
) -> int|float:
	mean = sum(vector) / len(vector)
	variance = sum((x - mean) ** 2 for x in vector) / len(vector)
	return mean, variance
# %%
def covariance(
    vector_1: list[int|float],
    vector_2: list[int|float],
    mean_1: int|float,
    mean_2: int|float,
) -> int|float:
	diff_1 = [x-mean_1 for x in vector_1]
	diff_2 = [x-mean_2 for x in vector_2]
	return sum([x_diff * y_diff for x_diff,y_diff in zip(diff_1, diff_2)]) / (len(vector_1)-1)
	
covariance([1,2,3],[-4,-5,-6], sum([1,2,3])/3, sum([-4,-5,-6])/3 )
	

	
# %%
def calculate_covariance_matrix(
    vectors: list[list[float]]
) -> list[list[float]]:
	cov_mat = []
	row_mean_matrix = calculate_matrix_mean(vectors, 'row')
	for i in range(len(vectors)):
		cov_mat_row = []
		for j in range(len(vectors)):
			cov_mat_row.append(covariance(vectors[i], vectors[j], row_mean_matrix[i], row_mean_matrix[j]))
		cov_mat.append(cov_mat_row)	
	return cov_mat


# %%
def phi_corr(x: list[int], y: list[int]) -> float:
	"""
	Calculate the Phi coefficient (correlation) between two binary variables.

	Args:
	x (list[int]): A list of binary values (0 or 1).
	y (list[int]): A list of binary values (0 or 1).

	Returns:
	float: The Phi coefficient rounded to 4 decimal places.
	"""
	x00 = x01 = x10 = x11 = 0
	for a, b in zip(x, y):
		x00 += (a, b) == (0, 0)
		x01 += (a, b) == (0, 1)
		x10 += (a, b) == (1, 0)
		x11 += (a, b) == (1, 1)

	denom = ((x00 + x01)*(x10 + x11)*(x00 + x10)*(x01 + x11))**0.5
	return round((x11*x00 - x10*x01) / denom, 4) if denom else 0.0

# %%
def poisson_probability(k, lam):
    """
    Poisson distribution is a discrete probability distribution that expresses the probability 
    of a given number of events occurring in a fixed interval of time or space, provided these 
    events occur with a known constant mean rate and independently of the time since the last event.  
    e.g. if the mean of an instance happening in the next hour is lam, the Poisson distr. answer the 
    question, what is the 
    
    
    `Prob(k; lam) := lam^k * exp(-lam) / k!`

    Calculate the probability of observing exactly k events in a fixed interval,
    given the mean rate of events lam, using the Poisson distribution formula.
    :param k: Number of events (non-negative integer)
    :param lam: The average rate (mean) of occurrences in a fixed interval
    """
    val = (lam ** k) * math.exp(-lam) / math.factorial(k)
    return round(val,5)

poisson_probability(3, 5)
# %%

def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials,
    where the success probability is p.
    
    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial
    
    Returns:
        Probability of k successes
    """
    return math.comb(n, k) * p**k * (1-p)**(n-k)

# %%
def normal_pdf(x, mean, std_dev):
    """
    Formula: 
        1 / sqrt(2*pi*sigma) * exp(-(x - mu)**2 / 2* sigma **2)

    Empirical Rule:
    - 68% of data falls within 1 standard deviation (μ±σμ±σ).
    - 95% of data falls within 2 standard deviations (μ±2σμ±2σ).
    - 99.7% of data falls within 3 standard deviations (μ±3σμ±3σ).



    Calculate the probability density function (PDF) of the normal distribution.
    :param x: The value at which the PDF is evaluated.
    :param mean: The mean (μ) of the distribution.
    :param std_dev: The standard deviation (σ) of the distribution.
    """
    val = (1 /(std_dev* math.sqrt(2*math.pi))) * math.exp(-(x - mean)**2 / (2* (std_dev **2)))
    return round(val,5)

# %%
def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    data = np.asarray(data)
    dim = data.size
    # mean
    mean = np.sum(data) / dim
    # median
    sorted_data = np.sort(data)
    median = sorted_data[(dim - 1) // 2]  if  ((dim % 2) == 1) else (sorted_data[dim//2 -1] + sorted_data[dim//2]) / 2
    # mode
    unique, counts = np.unique(data, return_counts=True)
    max_ids = np.argwhere(counts == np.max(counts)).flatten()
    mode = np.min(unique[max_ids])
    # variance
    variance = np.sum((counts/dim) *(unique - mean)**2)
    # standard deviation
    standard_deviation = np.sqrt(variance)
    # 25%
    q_1 = sorted_data[int(0.25*(dim-1))]
    # 50%
    q_2 = median
    # 75 %
    q_3 = sorted_data[int(0.75*(dim-1))]
    # IQR 
    iqr = q_3 - q_1 # used for outlier detection 
    return {
        'mean': mean,
        'median': median,
        'mode': mode, 
        'variance': variance,
        'standard_deviation':standard_deviation,
        '25th_percentile': q_1, 
        '50th_percentile': q_2,
        '75th_percentile': q_3,
        'interquartile_range':iqr
    }



# %%
def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
	"""
	Impute missing values in a 2D array using the specified strategy.

	Args:
		data: 2D numpy array with missing values represented as np.nan
		strategy: Imputation strategy - 'mean', 'median', or 'mode'
		
	Returns:
		2D numpy array with missing values imputed
	"""
	data = np.asarray(data)
	data =  data.T # now it's easier to to index the columns

	for column in data:
		if np.sum(np.isnan(column)) > 0: 
			index_nan = np.argwhere(np.isnan(column) == True)
			complement_column = np.delete(column, index_nan)
			column[index_nan] = descriptive_statistics(complement_column)[strategy]

	return data.T



# %% 
def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
    """
    https://www.deep-ml.com/problems/205
    Compute entropy of P and cross-entropy between P and Q.

    Args:
        P: True probability distribution
        Q: Predicted probability distribution

    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    H =  sum(-p*np.log(p, out=np.zeros_like(p, dtype=np.float64), where=(p!=0)) for p in P)
    Q =  sum(-P[i]*np.log(Q[i], where=(Q[i]!=0)) for i in range(len(P)))
    return H, Q

	