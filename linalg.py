# %%
import numpy as np
from math import sqrt
from typing import Optional, Literal

# %%
def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
    '''
    Return the element-wise sum of vectors 'a' and 'b'.
    If vectors have different lengths, return -1.
    '''
    if len(a)!=len(b):
        return -1
    return [a[i]+b[i] for i in range(len(a))]

# %%
def cross_product(a: list[int|float], b: list[int|float]) -> list[int|float]:
    '''
    The vectors must be 3-d
    '''
    return [a[1]*b[2]-a[2]*b[1], b[0]*a[2]-a[0]*b[2], a[0]*b[1]-b[0]*a[1]]

# %%
def vector_dot_vector(
    u: list[int|float],
    v: list[int|float]
) -> int|float:
    if len(u) == len(v):
        dot = 0
        for i in range(len(u)):
            dot += u[i] * v[i]
        return dot
    else: 
        return -1
    
# %%
def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L
    """
    v_dot_l = vector_dot_vector(v, L)
    l_dot_l = vector_dot_vector(L, L)
    return [(v_dot_l/l_dot_l)*l for l in L]
    
# %%
def norm_2(v: np.ndarray|list[int|float]) -> float:
    # return np.sqrt(sum(v**2)) # this is optimal for numpy arrays but we replace it with the more general
    return (vector_dot_vector(v,v))**0.5

# %%
def matrix_dot_vector(
    a: list[list[int|float]],
    b: list[int|float]
) -> list[int|float]:
    if len(a[0]) == len(b):
        prod = []
        for i in range(len(a)):
            prod.append(vector_dot_vector(a[i], b))
        return prod
    else:
        return -1


# %%
def transpose_matrix(
    a: list[list[int|float]]
) -> np.ndarray:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    return np.array([list(i) for i in zip(*a)])

# %%
def reshape_matrix(
	a: list[list[int|float]],
	new_shape: tuple[int, int]
) -> list[list[int|float]]:
	arr_a = np.array(a)
	reshaped_matrix = arr_a.reshape(new_shape).tolist() if np.prod(new_shape)==np.prod(arr_a.shape) else []
	return reshaped_matrix	

# %%
def calculate_matrix_mean(
	matrix: list[list[float]]|np.ndarray,
    mode: Optional[Literal['row', 'column']] = 'row'
) -> list[float]:
    '''
    Return a list of the mean values of mode.

    With numpy this is an one-liner
    matrix = np.array(matrix)
    return np.mean(matrix, axis=0) if mode=='row' else np.mean(matrix, axis=1)

    '''
    if mode == 'row':
        shape = len(matrix[0])
        return [sum(row)/shape for row in matrix]
    elif mode == 'column':
        matrix_t = transpose_matrix(matrix)
        shape = len(matrix_t[0])
        return [sum(row)/ shape for row in matrix_t]
    else:
        return -1

# %%
def scalar_multiply(
	matrix: list[list[int|float]],
	scalar: int|float
) -> list[list[int|float]]:
	return [[scalar * x for x in row] for row in matrix]

# %%
def trace(
    matrix: list[list[int|float]]
) -> int|float:
    return sum([j[i] for i, j in enumerate(matrix)]) 

# %%
def det_2x2(
    matrix: list[list[int|float]]
) -> int|float:
    return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

# %%
def solve_2_deg_poly(
    poly: list[int|float]
) -> list[int|float]:
    D = poly[1]**2 - 4* poly[0] * poly[2]
    if D > 0:
        roots = [(-poly[1]+sqrt(D))/(2*poly[0]), (-poly[1]-sqrt(D))/(2*poly[0])]
    elif D==0:
        roots = [-poly[1]/(2*poly[0])]
    else:
        return -1
    return roots

# %%
def calculate_eigenvalues(
	matrix: list[list[float|int]]
) -> list[float]:
	return solve_2_deg_poly([1, -trace(matrix), det_2x2(matrix)])

# %%
def transform_matrix(
    A: list[list[int|float]],
    T: list[list[int|float]],
    S: list[list[int|float]]
) -> list[list[int|float]]:
    
    A = np.array(A)
    T = np.array(T)
    S = np.array(S)

    try:
        T_inv = np.linalg.inv(T)
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError as err:
        return -1 
    
    return T_inv @ A @ S
    
	# return transformed_matrix

# %%
def inverse_2x2(
    matrix: list[list[float]]|np.ndarray
) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)

         A⁻¹ = (1/det) × [[d, -b], [-c, a]]
    """
    return scalar_multiply(matrix=[[matrix[1][1], -matrix[0][1]], [-matrix[1][0], matrix[0][0]]], scalar=1/det_2x2(matrix)) if det_2x2(matrix) != 0 else None

# %%
def matrixmul(
	a:list[list[int|float]],
    b:list[list[int|float]]
)-> list[list[int|float]]:
    
    c = []
    if len(a[0]) != len(b):
        return -1
    b_T = transpose_matrix(b)
    for row_a in a:
        c_row = []
        for row_b in b_T:
            c_row.append(vector_dot_vector(row_a, row_b))
        c.append(c_row)
    return c    



# %%
def solve_jacobi(
    A: np.ndarray,
    b: np.ndarray,
    n: int
) -> list:
    '''
    https://en.wikipedia.org/wiki/Jacobi_method
    '''
    x = np.zeros(A.shape[1])
    for _ in range(n):
        x_old = x.copy()
        for i in range(A.shape[1]):
            sum_term = sum([A[i,j]* x_old[j] for j in range(A.shape[0]) if j!=i])
            x[i] = round((1/A[i,i]) * (b[i] - sum_term), 4) 
    return x.tolist()

# %%
def jacobi_rotation(A:np.ndarray) -> np.ndarray:
    if sum((transpose_matrix(A) != A).flatten()) != 0:
        raise AssertionError("Argument not symmetric")
    else:
        theta = np.pi/4 if A[0,0]==A[1,1] else .5 * np.arctan(2*A[0,1]/(A[0,0]-A[1,1]))
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
    return R


# %%
def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 Jacobi rotationmatrix using one Jacobi rotation.
    https://en.wikipedia.org/wiki/Singular_value_decomposition

    The idea is basically that we take Jacobi rotations on the symmetric `A.T @ A` 

    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    B = A.T @ A 

    V = jacobi_rotation(B) # get angle to rotate B, one JR suffices since 2-d

    eigenvalues = np.diag(V.T @ B @ V)
    singular_values = np.sqrt(np.maximum(eigenvalues, 0)) # sorted array of eigenvalues of A.T @ A i.e. ev's of A but squared

    Sigma = np.diag(singular_values) # make the singular matrix 

    Sigma_inv = np.linalg.inv(Sigma)
    U = A @ V @ Sigma_inv

    return U, singular_values, V.T

# %%

def svd_2x2(A: np.ndarray) -> tuple:
    """
    IT DOESN'T WORK YET

    Compute SVD of a 2x2 matrix.

    Args:
        A: 2x2 numpy array

    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 orthogonal matrix (right singular vectors)
    """
    # compute auxiliary values
    y1 = A[1,0] + A[0,1]
    x1 = A[0,0] - A[1,1] 
    y2 = A[1,0] - A[0,1]
    x2 = A[0,0] + A[1,1] 

    # compute norms
    h1 = np.sqrt(x1**2 + y1**2)
    h2 = np.sqrt(x2**2 + y2**2)

    # compute singular values 
    sigma1 = (h1 + h2)/2
    sigma2 = np.abs(h1 - h2)/2
    s = np.array([sigma1, sigma2])

    # build U from rotation angles
    # handle h1 or h2 = 0 to avoid division by zero
    if h1 != 0:
        c1 = x1 / h1
    else:
        c1 = 1

    if h2 != 0:
        c2 = x2 / h2
    else:
        c2 = 1
    print(c1), print(c2)
    theta1 = np.arcsin(c1)
    # Combine rotations to get U
    rot1 = np.array([
        [np.cos(theta1), -np.sin(theta1)],
        [np.sin(theta1), np.cos(theta1)]
    ])
    print(rot1)
    
    # U = ???????????  

    # Ensure columns are normalized
    U[:,0] /= np.linalg.norm(U[:,0])
    U[:,1] = U[:,1] / np.linalg.norm(U[:,1]) if (U[:,1] != 0).all() else U[:,1]

    # compute V using V = Sigma^-1 * U.T * A
    # handle zero singular values to avoid division by zero
    V = np.zeros((2,2))
    for i in range(2):
        if s[i] > 1e-12:
            V[:,i] = (U[:,i].T @ A) / s[i]
        else:
            # if singular value is zero, choose arbitrary orthogonal vector
            V[:,i] = np.array([-U[1,i], U[0,i]])
    
    # ensure V is orthogonal (normalize columns)
    V[:,0] /= np.linalg.norm(V[:,0])
    V[:,1] = V[:,1] / np.linalg.norm(V[:,1]) if (V[:,1] != 0).all() else V[:,1]
    print(V)

    # verification (optional, can remove in production)
    # assert np.allclose(A, U @ np.diag(s) @ V.T)
    # assert np.allclose(U @ U.T, np.identity(2))
    # assert np.allclose(V @ V.T, np.identity(2))
    # assert sigma1 >= 0 and sigma2 >= 0

    return U, s, V.T

    
# %%
assert (np.arange(10).reshape(2,-1) == np.arange(10).reshape(2,-1)).all()
# %%
def lu_decomposition(A: list) -> tuple:
    """
    Perform LU decomposition on a square matrix using Doolittle's method.
    https://en.wikipedia.org/wiki/LU_decomposition
    
    Args:
        A: Square matrix as a list of lists

    Returns:
        tuple: (L, U) where L is lower triangular with 1s on diagonal,
                U is upper triangular, and A = L @ U
    """
    A = np.array(A).astype(float)
    U = np.zeros(A.shape)
    L = np.identity(A.shape[0])

    U=A
    for j in range(0, A.shape[1]-1):
        for i in range(1+j, A.shape[0]):
            term = U[i,j] / U[j,j]
            L[i,j]=term
            U[i]=U[i]-term*U[j]
    return (L, U)

# %%
def diag(A:list[list[int|float]]|np.ndarray) -> list:
    '''
    Return a list of the main diagonal elements of a matrix
    '''
    A = np.asarray(A)
    return [A[i, i] for i in range(A.shape[0])]

# %% 
def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
    """
    Compute the determinant and trace of a square matrix.

    Args:
        matrix: A square matrix (n x n) represented as list of lists

    Returns:
        Tuple of (determinant, trace)
    """
    _, U = lu_decomposition(matrix)
    det = np.prod(diag(U))
    tr = trace(matrix)

    return det, tr

# %% 
def create_submatrix(
        matrix: list[list[int|float]],
        i: int, 
        j: int
) -> list[list[int|float]]:
    
    if i>=len(matrix) or j>=len(matrix[0]):
        raise ValueError('index out of bounds')
    if i==len(matrix)-1:
        sub_row = matrix[:i]
    else:
        sub_row = matrix[:i] + matrix[i+1:]
    
    if j==len(matrix[0])-1:
        sub_col = [row[:j] for row in sub_row]
    else:
        sub_col = [row[:j] + row[j+1:] for row in sub_row]

    return sub_col

# %%
def determinant_4x4(matrix: list[list[int|float]]) -> float:
    determinant = 0
    if len(matrix)==2 and len(matrix[0])==2: # stopping criterion
        return det_2x2(matrix)

    roe = matrix[0] # the row of expansion is the first row
    signs = [1,-1,1,-1] # [+,-,+,-]

    for id, entry in enumerate(roe):
        multiplier = signs[id] * entry
        submatrix = create_submatrix(matrix, i=0, j=id)
        determinant += multiplier * determinant_4x4(submatrix)
    return determinant


# %%
def make_diagonal(x):
    return np.diag(x)

# %%
def transform_basis(
        B: list[list[int]],
        C: list[list[int]]
    ) -> list[list[float]]:

	pass

# %%
def matrix_mean(matrix: list[list[int|float]], axis:int=0) -> float:
    '''
    take the mean of the values row-wise (axis=0) or column-wise (axis=1)
    '''
    if axis==0:
        return [sum(matrix[i])/len(matrix[i]) for i in range(len(matrix))]
    else:
        return [sum([matrix[i][j] for i in range(len(matrix))]) / len([matrix[i][j] for i in range(len(matrix))]) for j in range(len(matrix[0]))]

# %%
def compute_norm(arr: np.ndarray, norm_type: str = 'l2') -> float:
    """
    Compute the specified norm of the input array.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    arr = np.asarray(arr, dtype=np.float32)
    if norm_type == 'l1':
        norm = np.sum(np.abs(arr.flatten()))
    elif norm_type == 'l2':
        norm = np.sqrt(np.sum((arr.flatten())**2))
    else:
        norm = np.sqrt(np.sum(np.sum(arr**2, axis=1), axis=0))

    return norm

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
def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.

    Args:
        vectors: List of vectors, where each vector is a list of floats.
                    All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    tol=1e-12
    A = np.asarray(vectors, dtype=float)

    m = A.shape[0]
    n = A.shape[1] if m>0 else None
    if m == 0:
        return True
    if m > n:
        return False

    A = A.copy()
    rank = 0

    for col in range(n):
        # find pivot
        pivot_rows = np.where(np.abs(A[rank:, col]) > tol)[0]
        if pivot_rows.size == 0:
            continue

        pivot = pivot_rows[0] + rank
        A[[rank, pivot]] = A[[pivot, rank]]

        A[rank] /= A[rank, col]

        # eliminate
        for r in range(m):
            if r != rank:
                A[r] -= A[r, col] * A[rank]

        rank += 1
        if rank == m:
            break

    return rank == m