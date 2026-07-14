import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.asarray(T)
    points = np.asarray(points)
    
    is_single_point = (points.ndim == 1)
    pts = np.atleast_2d(points)

    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h = np.hstack((pts, ones))

    transformed_h = (T @ pts_h.T).T
    transformed_spatial = transformed_h[:, :3]

    if is_single_point:
        output = transformed_spatial[0]
    else:
        output = transformed_spatial

    return output
        