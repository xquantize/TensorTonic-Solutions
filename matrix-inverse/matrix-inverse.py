import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # codes
    A = np.array(A)
    
    # check its 2d array
    if A.ndim != 2:
        return None
    
    # check matrix is square nxn
    rows, cols = A.shape
    if rows != cols:
        return None

    # chec maxtrix is singular, determinant is near zero
    if np.abs(np.linalg.det(A)) < 1e-10:
        return None

    # compute inverse or return none
    try:
        A_inv = np.linalg.inv(A)
        return A_inv
    except np.linalg.LinAlgError:        
        return None
