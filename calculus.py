# %%
import numpy as np
from typing import Callable


# %%
def poly_term_derivative(c: float, x: float, n: float) -> float:
    '''
    https://www.deep-ml.com/problems/116
    derivative of a polynomial term of the form c * x^n at a given *point x, 
    where c is a coefficient and n is the exponent
    '''
    if n>0:
        return n*c*x**(n-1)
    else:
        return 0

# %%
def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    https://www.deep-ml.com/problems/309
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' = f' * g + f * g' as a list of floats rounded to 4 decimal places
    """
    grad_f = [poly_term_derivative(f_coeffs[i], 1, i) for i in range(len(f_coeffs))][1:]
    grad_g = [poly_term_derivative(g_coeffs[i], 1, i) for i in range(len(g_coeffs))][1:]

    poly_grad_f = np.poly1d(grad_f[::-1])
    poly_grad_g = np.poly1d(grad_g[::-1])
     

    poly_f = np.poly1d(f_coeffs[::-1])
    poly_g = np.poly1d(g_coeffs[::-1])
    f_times_g_prime = poly_grad_f * poly_g + poly_f * poly_grad_g
    
    return np.round((np.asarray(f_times_g_prime.coefficients[::-1])), 4)
   

print(product_rule_derivative([1, 2], [3, 4]))
# %%

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    https://www.deep-ml.com/problems/312
    Compute the derivative of f(x) = g(x)/h(x) at *point x using the quotient rule.

    f' = (g' h - g h') / h^2

    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: *Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    g = np.poly1d(g_coeffs)
    h = np.poly1d(h_coeffs)

    denominator = h**2

    grad_g = np.poly1d([poly_term_derivative(g_coeffs[i], 1, len(g_coeffs)-i-1) for i in range(len(g_coeffs))][:-1])
    grad_h = np.poly1d([poly_term_derivative(h_coeffs[i], 1, len(h_coeffs)-i-1) for i in range(len(h_coeffs))][:-1])

    numerator = grad_g * h - g * grad_h

    return numerator(x)/denominator(x)

quotient_rule_derivative(g_coeffs = [1, 0, 1], h_coeffs = [1, 2], x = 2.0)

# %%
def gradient_direction_magnitude(gradient: list) -> dict:
    """
    https://www.deep-ml.com/problems/308
    Calculate the magnitude and direction of a gradient vector.

    Args:
        gradient: A list representing the gradient vector

    Returns:
        Dictionary containing:
        - magnitude: The L2 norm of the gradient
        - direction: Unit vector in direction of steepest ascent
        - descent_direction: Unit vector in direction of steepest descent
    """
    gradient = np.asarray(gradient)

    magnitude = np.linalg.norm(gradient)
    direction = (gradient / magnitude) if magnitude != 0. else np.zeros(gradient.shape)
    descent_direction = -direction

    return {
        'magnitude':magnitude,
        'direction':direction,
        'descent_direction':descent_direction
    }

result = gradient_direction_magnitude([0.0, 0.0])
print(f"{result['magnitude']:.4f},{[round(d,4) for d in result['direction']]}")


# %%
def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
    """
    https://www.deep-ml.com/problems/215
    Compute partial derivatives of multivariable functions.

    Args:
        func_name: Function identifier
            'poly2d': f(x,y) = x²y + xy²
            'exp_sum': f(x,y) = e^(x+y)
            'product_sin': f(x,y) = x·sin(y)
            'poly3d': f(x,y,z) = x²y + yz²
            'squared_error': f(x,y) = (x-y)²
        point: *Point (x, y) or (x, y, z) at which to evaluate

    Returns:
        Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
    """
    if func_name == 'poly2d':
        partial_x = lambda x, y: 2*x*y + y**2
        partial_y = lambda x, y: x**2 + 2*x*y
        return (partial_x(*point), partial_y(*point))  # very nice trick to unravel (x, y) into x, y
    
    if func_name == 'exp_sum':
        partial_x = lambda x, y: np.exp(x+y) 
        partial_y = lambda x, y: np.exp(x+y)
        return (partial_x(*point), partial_y(*point))
    
    if func_name == 'product_sin':
        partial_x = lambda x, y: np.sin(y)
        partial_y = lambda x, y: x*np.cos(y)
        return (partial_x(*point), partial_y(*point))

    if func_name == 'poly3d':
        partial_x = lambda x, y, z: 2*x*y
        partial_y = lambda x, y, z: x**2 + z**2
        partial_z = lambda x, y, z: 2*y*z
        return (partial_x(*point), partial_y(*point), partial_z(*point))
    
    if func_name == 'squared_error':
        partial_x = lambda x, y: 2*(x-y)
        partial_y = lambda x, y: -2*(x-y)
        return (partial_x(*point), partial_y(*point))


compute_partial_derivatives(func_name='poly2d', point=(2.0, 3.0))

# %%
def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    https://www.deep-ml.com/problems/214
    Compute derivative of composite functions using chain rule.

    Args:
        functions: List of function names (applied right to left)
                    Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative

    Returns:
        Derivative value at x

    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """
    # 'square'
    square = (lambda x: x**2, lambda x: 2*x)
     
    # 'sin'
    sin = (lambda x: np.sin(x), lambda x: np.cos(x))

    # 'exp'
    exp = (lambda x: np.exp(x), lambda x: np.exp(x))
    
    # 'log'
    log = (lambda x: np.log(x), lambda x: 1/x)

    available ={
        'square':square[0],
        'sin':sin[0],
        'exp':exp[0],
        'log':log[0]
    }

    grads ={
        'square':square[1],
        'sin':sin[1],
        'exp':exp[1],
        'log':log[1]
    }
    # for = [x**2, sin]
    # back = cos(square(x)) * grad_square(x)

    # forward
    forward = [x]
    for func in functions[::-1]: # order of function applied
        f = available[func]
        x = f(x)
        forward.append(x)  # [x = u0, f_1(x) = u1, f_2(f_1(x)) = u2]

    backward = 1
    for i, func in enumerate(functions[::-1]):
        f_prime = grads[func]
        backward *= f_prime(forward[i])
        

    return backward

# %%
def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    https://www.deep-ml.com/problems/202
    Compute the Jacobian matrix using numerical differentiation.

    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences

    Returns:
        Jacobian matrix as list of lists
    """

    # f = (f1, ..., fm)
    # x = (x1, ..., xn)
    # J.shape = (m ,n)
    # i.e. row1 = [partial1 f1, partial2 f1, ...., partialn f1]

    # compute partials
    jac_matrix = []
    for j in range(len(x)):
        x_tilde = x.copy()
        x_tilde[j] = x_tilde[j] + h
        partial = (np.asarray(f(x_tilde)) - np.asarray(f(x))) / h
        jac_matrix.append(partial)

    return np.asarray(jac_matrix).T

# %%
def f(p): return p[0]**2 + p[1]**2

jacobian_matrix(f, x = [0.0, 0.0])

# %%
def compute_hessian(
        f: Callable[[list[float]], float],
        point: list[float],
        h: float = 1e-5
) -> list[list[float]]:
    """
    Compute the Hessian matrix of function f at the given point using finite differences.

    Args:
        f: A scalar function that takes a list of floats and returns a float
        point: The point at which to compute the Hessian (list of coordinates)
        h: Step size for finite differences (default: 1e-5)
        
    Returns:
        The Hessian matrix as a list of lists (n x n where n = len(point))
    """
    hessian_matrix = []

    for i in range(len(point)):
        hessian_row = []
        for j in range(len(point)):
            if i == j:
                # diagonal 
                point_tilde_p = point.copy()
                point_tilde_m = point.copy()
                point_tilde_p[i] = point_tilde_p[i] + h
                point_tilde_m[i] = point_tilde_m[i] - h

                second_partial = (f(point_tilde_p) + f(point_tilde_m) - (2 * f(point))) / h**2
                hessian_row.append(second_partial)
            
            else:
                # off diagonal
                point_tilde_p_p = point.copy()
                point_tilde_p_m = point.copy()
                point_tilde_m_p = point.copy()
                point_tilde_m_m = point.copy()

                point_tilde_p_p[i] = point_tilde_p_p[i] + h
                point_tilde_p_p[j] = point_tilde_p_p[j] + h

                point_tilde_p_m[i] = point_tilde_p_m[i] + h
                point_tilde_p_m[j] = point_tilde_p_m[j] - h

                point_tilde_m_p[i] = point_tilde_m_p[i] - h
                point_tilde_m_p[j] = point_tilde_m_p[j] + h

                point_tilde_m_m[i] = point_tilde_m_m[i] - h
                point_tilde_m_m[j] = point_tilde_m_m[j] - h

                second_partial = (f(point_tilde_p_p)+f(point_tilde_m_m)-f(point_tilde_p_m)-f(point_tilde_m_p))/(4*h**2)
                hessian_row.append(second_partial)
        hessian_matrix.append(hessian_row)
    return hessian_matrix

# %%
def f(p): return p[0]**2 + p[1]**2
result = compute_hessian(f, [0.0, 0.0])
print([[round(v, 4) for v in row] for row in result])



            


	
# %%

point = np.arange(10)
point[1]= point[1] + 10
point[3]= point[3] + 100

point
