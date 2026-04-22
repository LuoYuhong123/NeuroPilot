import h5py
import numpy as np
import cv2
import scipy.ndimage as ndimage

def imregister_wrapper(f2, u, v, f1=None):
    if f1 is None:  # Equivalent to checking if nargin < 4
        f1 = f2

    # Create an array `w` with an additional dimension for `u` and `v`
    w = np.zeros((u.shape[0], u.shape[1], 2), dtype='double')
    w[:, :, 0] = u
    w[:, :, 1] = v

    # Call the helper function
    registered = imregister_wrapper_w(f2, w, f1)
    return registered



def imregister_wrapper_w(f2, w, f1=None, interpolation_method='linear'):
    if f1 is None:
        f1 = f2

    assert f2.shape[:2] == w.shape[:2] and f1.shape[:2] == w.shape[:2], \
        f"imregister sizes do not match: f1 = {f1.shape[:2]}, f2 = {f2.shape[:2]}, w = {w.shape[:2]}"
    assert f2.shape == f1.shape, \
        f"Dimensions of f1 ({f1.shape}) and f2 ({f2.shape}) do not match."

    # 选择插值方式
    if interpolation_method == 'nearest':
        order = 0
    elif interpolation_method == 'linear':
        order = 1
    else:  # cubic
        order = 3

    # 传入插值 order
    registered = warp_image_2(f2, w, order=order)

    idx = np.isnan(registered)
    registered[idx] = f1[idx]

    return registered




def warp_image(f2, w, order=3):
    """
    Mimic MATLAB's `imwarp` functionality by applying displacement vectors `w`
    to image `f2` with interpolation.
    """
    coords = np.indices(f2.shape[:2], dtype='float64')
    coords[0] += w[:, :, 0]
    coords[1] += w[:, :, 1]

    # Flatten the coordinates and interpolate with map_coordinates
    warped = map_coordinates(f2, [coords[0].ravel(), coords[1].ravel()], order=order, mode='constant', cval=np.nan)
    return warped.reshape(f2.shape)



def warp_image_2(f2, w, order=3):
    height, width = f2.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    flow_x = w[:, :, 0]
    flow_y = w[:, :, 1]

    map_x = x + flow_x
    map_y = y + flow_y

    coordinates = np.stack((map_y, map_x), axis=-1)

    # 改动这里：使用 order 变量（0=nearest, 1=linear, 3=cubic）
    # 'reflect'   constant
    registered_image = ndimage.map_coordinates(
        f2,
        coordinates.transpose(2, 0, 1),
        order=order,
        mode='reflect',
        cval=np.nan
    )

    return np.array(registered_image, dtype=np.float64)



def imwarp_optical_flow(f2, w):
    """
    Apply optical flow-based transformation to an image using a flow field.

    Parameters:
    - f2: Input image (numpy array).
    - w: Optical flow matrix of size (height, width, 2) where each entry in the 3rd dimension represents [dx, dy].
    - interpolation_method: Interpolation method (e.g., 'cv2.INTER_CUBIC').
    - fill_value: Value to use for out-of-bound pixels (default: np.nan).

    Returns:
    - Transformed image (numpy array).
    """
    # Get the height and width of the image
    height, width = f2.shape

    # Create meshgrid for image coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Apply optical flow to get the new coordinates
    flow_x = w[:, :, 0]  # Flow in the x direction
    flow_y = w[:, :, 1]  # Flow in the y direction

    # New coordinates after applying the optical flow (displacement)
    map_x = x + flow_x
    map_y = y + flow_y

    # Convert to the required type for remap function (float32)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # Apply remap using the optical flow as a mapping
    registered_image = cv2.remap(f2, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

    return registered_image
if __name__ == '__main__':

    f1_level = np.array(h5py.File('test_imregister_wrapper.mat')['f1_level'], dtype=np.float64).transpose()
    f2_level = np.array(h5py.File('test_imregister_wrapper.mat')['f2_level'], dtype=np.float64).transpose()
    u = np.array(h5py.File('test_imregister_wrapper.mat')['u'], dtype=np.float64).transpose()
    v = np.array(h5py.File('test_imregister_wrapper.mat')['v'], dtype=np.float64).transpose()
    gt_tmp = np.array(h5py.File('test_imregister_wrapper.mat')['tmp'], dtype=np.float64).transpose()
    w = np.zeros((u.shape[0], u.shape[1], 2), dtype='double')
    w[:, :, 0] = u
    w[:, :, 1] = v
    tmp = imregister_wrapper(f2_level,u,v,f1_level)
    tmp_cv = imwarp_optical_flow(f2_level,w)
    idx = np.isnan(tmp_cv)
    tmp_cv[idx] = f1_level[idx]
    mse = np.sqrt(np.mean((tmp - gt_tmp)**2))
    print('mse:',mse)
    error_rate = mse / (np.max(gt_tmp) - np.min(gt_tmp))
    print('error rate:',error_rate)
    mse_cv = np.sqrt(np.mean((tmp_cv - gt_tmp) ** 2))
    print('mse_cv:', mse_cv)
    error_rate_cv = mse_cv / (np.max(gt_tmp) - np.min(gt_tmp))
    print('error rate cv:', error_rate_cv)