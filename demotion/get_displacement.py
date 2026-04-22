import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
import h5py as h5
from demotion.warpingDepth import warpingDepth
from demotion.get_motion_tensor_gc import get_motion_tensor_gc
from demotion.imregister_wrapper import imregister_wrapper
from joblib import Parallel, delayed

from demotion import level_solver_acc
from demotion.imresize import imresize
import time
# Function to compute displacement


def get_displacement(fixed, moving, kwargs):
    # Default parameters
    alpha = [2, 2]
    update_lag = 10
    iterations = 20
    min_level = 0
    levels = 50
    eta = 0.75
    a_smooth = 0.5

    # Get the size of the fixed image
    m, n = fixed.shape

    # Initialize u and v to zeros
    u_init = np.zeros((m, n))
    v_init = np.zeros((m, n))

    # Set default weight and a_data
    weight = np.ones((1, 1), dtype=np.float64)
    a_data = 0.45 * np.ones(1)


    # Parse optional arguments from kwargs
    if 'weight' in kwargs:
        weight = np.array(kwargs['weight'],dtype = np.float64)
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
        if isinstance(alpha,int) or isinstance(alpha,float):
            alpha = np.array(alpha * np.ones(2))
        else:
            alpha = kwargs['alpha'][0][0]
            alpha = np.array(alpha * np.ones(2))
    if 'eta' in kwargs:
        eta = kwargs['eta']
        if not isinstance(eta, int) and not isinstance(eta, float):
            eta = kwargs['eta'][0][0]
    if 'levels' in kwargs:
        levels = kwargs['levels']
        if not isinstance(levels, int) and not isinstance(levels, float):
            levels = kwargs['levels'][0][0]
    if 'update_lag' in kwargs:
        update_lag = kwargs['update_lag']
        if not isinstance(update_lag, int) and not isinstance(update_lag, float):
            update_lag = kwargs['update_lag'][0][0]
    if 'iterations' in kwargs:
        iterations = kwargs['iterations']
        if not isinstance(iterations, int) and not isinstance(iterations, float):
            iterations = kwargs['iterations'][0][0]
    if 'uv' in kwargs:
        u_init = kwargs['uv'][0]
        v_init = kwargs['uv'][1]
    if 'a_data' in kwargs:
        a_data = kwargs['a_data']
        a_data = np.array(a_data,dtype=np.float64)
    if 'a_smooth' in kwargs:
        a_smooth = kwargs['a_smooth']
        if not isinstance(a_smooth, int) and not isinstance(a_smooth, float):
            a_smooth = kwargs['a_smooth'][0][0]
    if 'min_level' in kwargs:
        min_level = kwargs['min_level']
        if not isinstance(min_level, int) and not isinstance(min_level, float):
            min_level = kwargs['min_level'][0][0]

    # Convert images to double
    f1_low = fixed.astype(np.float64)
    f2_low = moving.astype(np.float64)

    method = Image.Resampling.BICUBIC

    # Ensure the maximum pyramid level
    max_level_y = warpingDepth(eta, levels, m, m)
    max_level_x = warpingDepth(eta, levels, n, n)
    max_level = min(max_level_x, max_level_y) * 4
    max_level_y = min(max_level_y, max_level)
    max_level_x = min(max_level_x, max_level)

    local_weight = weight.ndim == fixed.ndim and np.sum(weight.shape == fixed.shape) == fixed.ndim
    weight_level = weight

    if max(max_level_x, max_level_y) <= min_level:
        min_level = max(max_level_x, max_level_y) - 1
    if min_level < 0:
        min_level = 0
    max_level_x = int(max_level_x)
    max_level_y = int(max_level_y)
    min_level = int(min_level)
    #print('min level:',min_level)
    # Iterate through each level
    for i in range(max(max_level_x, max_level_y), min_level - 1, -1):
        # Compute each level's size
        #print('level:',i)
        level_size = (round(m * eta ** min(i, max_level_y)),
                      round(n * eta ** min(i, max_level_x)))

        # Resize fixed and moving images to the current level size
        #f1_level = np.array(Image.fromarray(f1_low).resize(level_size[::-1], resample=method))
        #f2_level = np.array(Image.fromarray(f2_low).resize(level_size[::-1], resample=method))
        f1_level = imresize(f1_low,output_shape = level_size[::-1])
        f2_level = imresize(f2_low, output_shape=level_size[::-1])

        if local_weight:
            #weight_level = np.pad(np.array(Image.fromarray(weight).resize(level_size[::-1], resample=method)), pad_width=1, mode='constant', constant_values=0.0)
            weight_level = np.pad(imresize(weight,output_shape = level_size[::-1]),pad_width=1, mode='constant', constant_values=0.0)
        # Compute the zoom factor
        hx = m / f1_level.shape[0]
        hy = n / f1_level.shape[1]

        # Compute displacement for each level
        if i == max(max_level_x, max_level_y):
            #print('add_boundary')
            #u = add_boundary(np.array(Image.fromarray(u_init).resize(level_size[::-1], resample=method)))
            #v = add_boundary(np.array(Image.fromarray(v_init).resize(level_size[::-1], resample=method)))
            u = add_boundary(imresize(u_init, output_shape = level_size[::-1]))
            v = add_boundary(imresize(v_init, output_shape = level_size[::-1]))
            tmp = f2_level.copy()
        else:
            #print('add_boundary and warp')
            #u = add_boundary(np.array(Image.fromarray(u[1:-1, 1:-1]).resize(level_size[::-1], resample=method)))
            #v = add_boundary(np.array(Image.fromarray(v[1:-1, 1:-1]).resize(level_size[::-1], resample=method)))
            u = add_boundary(imresize(u[1:-1,1:-1], output_shape=level_size[::-1]))
            v = add_boundary(imresize(v[1:-1,1:-1], output_shape=level_size[::-1]))
            try:
                # input_u = np.float64(u[1:-1, 1:-1]) / np.float64(hx)
                # input_f2 = np.float64(f2_level)
                tmp = imregister_wrapper(np.array(f2_level,dtype=np.float64), np.array(u[1:-1, 1:-1] / hx,dtype=np.float64), np.array(v[1:-1, 1:-1] / hy,dtype=np.float64), np.array(f1_level,dtype=np.float64))
                #tmp_gt = np.array(h5.File('test_imregister_wrapper.mat')['tmp'], dtype=np.float64).transpose()
                #print('warp done')
            except Exception as e:
                #print(e)
                raise RuntimeError("Error using imregister for compensating flow increments, try increasing alpha!")

        # Get motion tensor
        J11, J22, J33, J12, J13, J23 = get_motion_tensor_gc(f1_level, tmp, hx, hy)
        #print('get_motion_tensor done')
        # Scaling factor for alpha
        if i == min_level:
            alpha_scaling = 1
        else:
            alpha_scaling = eta ** (-0.5 * i)

        # Solve for du and dv
        plhs = [None,None]
        prhs = [J11, J22, J33, J12, J13, J23, weight_level, u, v,
                              alpha * alpha_scaling, iterations, update_lag, 0, a_data, a_smooth, hx, hy]
        du, dv = level_solver_acc.level_solver_acc(plhs,prhs)
        #print('level solver done')
        # Apply median filtering if the level size is sufficiently large
        if min(level_size) > 5:

            du[1:-1, 1:-1] = median_filter(du[1:-1, 1:-1], size=(5, 5), mode='reflect')
            dv[1:-1, 1:-1] = median_filter(dv[1:-1, 1:-1], size=(5, 5), mode='reflect')
            #print('median filter done')
        # Update u and v with the displacements
        u += du
        v += dv

    # Final displacement field
    w = np.zeros((u.shape[0] - 2, u.shape[1] - 2, 2), dtype=np.float64)
    w[:, :, 0] = u[1:-1, 1:-1]
    w[:, :, 1] = v[1:-1, 1:-1]
    flow = w
    # Resize to original dimensions if necessary
    if min_level > 0:
        flow = np.zeros((m,n,2),dtype = np.float64)
        #print(w.shape)
        flow[:, :, 0] = imresize(w[:, :, 0], output_shape = (m,n))
        flow[:, :, 1] = imresize(w[:, :, 1], output_shape = (m,n))

    return flow


def add_boundary(f):
    # Add boundary padding to the input array
    f_padded = np.pad(f, pad_width=1, mode='edge')
    return set_boundary(f_padded)


def set_boundary(f):
    # Set boundary values by copying from adjacent values
    f[:, 0] = f[:, 1]
    f[:, -1] = f[:, -2]
    f[0, :] = f[1, :]
    f[-1, :] = f[-2, :]
    return f

def read_v(file_path):
    # Initialize an empty dictionary to store key-value pairs
    data_dict = {}

    with h5.File(file_path, 'r') as f:
        # Access the dataset named 'v'
        dataset = f['v']

        # Iterate over the dataset to extract key-value pairs
        for i in range(0, len(dataset), 2):
            # Extract the key
            key_ref = dataset[i][0]
            key = None

            # Debugging: Print the key_ref type and value
            #print(f"Processing key reference at index {i}: {key_ref} (type: {type(key_ref)})")

            if isinstance(key_ref, h5.Reference):
                key_data = f[key_ref][()]
                # Debugging: Print key_data after dereferencing
                #print(f"Key data after dereferencing: {key_data} (type: {type(key_data)})")

                # Convert key_data to a string if it's an array of ASCII codes
                if isinstance(key_data, np.ndarray):
                    if key_data.dtype == np.uint8 or key_data.dtype == np.uint16 or key_data.dtype == np.int8 or key_data.dtype == np.int32:
                        key = ''.join(chr(x) for x in key_data.flatten())  # Convert ASCII to string
                        #print(f"Key after ASCII conversion: {key}")
                    elif key_data.size == 1:
                        key = key_data.item()  # Extract the single item
                        if isinstance(key, bytes):
                            key = key.decode('utf-8')
                    else:
                        key = str(key_data)  # Fallback to string conversion
                elif isinstance(key_data, bytes):
                    key = key_data.decode('utf-8')
                else:
                    key = str(key_data)  # Fallback to string conversion if needed
            elif isinstance(key_ref, bytes):
                key = key_ref.decode('utf-8')
            else:
                key = str(key_ref)

            # Debugging: Print the final key value
            #print(f"Final key: {key}")

            # Extract the value (which can be a reference, array, or direct value)
            value_ref = dataset[i + 1][0]
            value = None

            # Debugging: Print the value reference
            #print(f"Processing value reference at index {i + 1}: {value_ref} (type: {type(value_ref)})")

            if isinstance(value_ref, h5.Reference):
                value_data = f[value_ref][()]
                # Debugging: Print value data after dereferencing
                #print(f"Value data after dereferencing: {value_data} (type: {type(value_data)})")

                # Handle different possible types for the value
                if isinstance(value_data, bytes):
                    value = value_data.decode('utf-8')  # Convert bytes to string if applicable
                elif isinstance(value_data, np.ndarray):
                    if value_data.dtype == 'object':
                        # Decode if the array contains bytes/strings
                        value = [elem.decode('utf-8') if isinstance(elem, bytes) else elem for elem in value_data]
                    else:
                        value = value_data.tolist()  # Convert numerical array to list
                else:
                    value = value_data  # Use directly if it's already a primitive value
            elif isinstance(value_ref, np.ndarray):
                value = value_ref.tolist()  # Convert to list if ndarray
            else:
                value = value_ref  # Direct assignment if simple value

            # Debugging: Print the final value
            #print(f"Final value for key '{key}': {value}")

            # Store the key-value pair in the dictionary
            data_dict[key] = value
    return data_dict
if __name__ == '__main__':

    import numpy as np
    from PIL import Image
    from scipy.ndimage import median_filter
    import h5py as h5
    from warpingDepth import warpingDepth
    from get_motion_tensor_gc import get_motion_tensor_gc
    from imregister_wrapper import imregister_wrapper
    import level_solver_acc
    from imresize import imresize
    import time



    C = np.array(h5.File('test_get_displacement.mat')['C'], dtype=np.float64).transpose((1,0))
    c_ref = np.array(h5.File('test_get_displacement.mat')['c_ref'], dtype=np.float64).transpose((1,0))
    gt_w_tmp = np.array(h5.File('test_get_displacement.mat')['w_tmp'], dtype=np.float64).transpose((2,1,0))
    #print(gt_w_tmp.shape)
    v = read_v('test_get_displacement.mat')
    print(v)

    st_time = time.time()
    w_tmp = get_displacement(c_ref,C,v)
    ed_time = time.time()
    print('Time Consume:',ed_time - st_time)
    print(w_tmp.shape)
    print('w_tmp:')
    print(w_tmp[:5,:5,0])
    print(w_tmp[:5,:5,1])
    print('gt_w_tmp:')
    print(gt_w_tmp[:5,:5,0])
    print(gt_w_tmp[:5,:5,1])
    #np.save('w_tmp.npy',w_tmp)
    mse = np.sqrt(np.mean((w_tmp - gt_w_tmp)**2))
    error_rate = mse / (np.max(gt_w_tmp) - np.min(gt_w_tmp))
    print('MSE:',mse)
    print('Error Rate:',error_rate)



    
    
    