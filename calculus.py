# %%
import numpy as np

# %%
def poly_term_derivative(c: float, x: float, n: float) -> float:
    '''
    derivative of a polynomial term of the form c * x^n at a given point x, 
    where c is a coefficient and n is the exponent
    '''
    return n*c*x**(n-1)

# %%
def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' = f' * g + f * g' as a list of floats rounded to 4 decimal places
    """
    grad_f = [poly_term_derivative(f_coeffs[i], 1, i) for i in range(len(f_coeffs))]
    grad_g = [poly_term_derivative(g_coeffs[i], 1, i) for i in range(len(g_coeffs))]

    print(grad_f)
    print(grad_g)


print(product_rule_derivative([0, 1], [0, 1]))
# %%

a = [0,2,3]
b = [4,5,6]
prod = []

np.deg(np.poly1d(a) * np.poly1d(b))


