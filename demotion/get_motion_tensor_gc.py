
import numpy as np
np.set_printoptions(precision=4)

def get_motion_tensor_gc(f1, f2, hx, hy):

    # Pad f1 and f2 symmetrically
    hx = np.float16(hx)
    hy = np.float16(hy)
    f1 = np.pad(f1, pad_width=1, mode='symmetric')
    f2 = np.pad(f2, pad_width=1, mode='symmetric')
    # Apply set_boundary if needed (commented out to match MATLAB behavior)


    # Compute gradients of f1 and f2
    _, fx1 = np.gradient(f1, hx, hy, edge_order=2)
    _, fx2 = np.gradient(f2, hx, hy, edge_order=2)

    fx1[:, 0] = fx1[:, -1] = 0
    fx2[:, 0] = fx2[:, -1] = 0



    fx = 0.5 * (fx1 + fx2)
    ft = f2 - f1


    # Remove padding and pad symmetrically again
    fx = np.pad(fx[1:-1, 1:-1], pad_width=1, mode='symmetric')
    ft = np.pad(ft[1:-1, 1:-1], pad_width=1, mode='symmetric')

    # Compute gradients of fx and ft
    fxy, _ = np.gradient(fx, hx, hy,edge_order=2)
    fyt, fxt = np.gradient(ft, hx, hy,edge_order=2)

    fxy[0, :] = fxy[-1, :] = 0
    fyt[0, :] = fyt[-1, :] = 0
    fxt[:, 0] = fxt[:, -1] = 0
    # Apply set_boundary if needed (commented out to match MATLAB behavior)
############################debug line#################################
    # Compute second-order gradients for f1 and f2
    fxx1, fyy1 = gradient2(f1, hx, hy)
    fxx2, fyy2 = gradient2(f2, hx, hy)

    fxx = 0.5 * (fxx1 + fxx2)
    fyy = 0.5 * (fyy1 + fyy2)



    # Dataterm normalization from Zimmer et al.
    reg_x = 1.0 / (np.sqrt(fxx ** 2 + fxy ** 2) ** 2 + 1e-6)
    reg_y = 1.0 / (np.sqrt(fxy ** 2 + fyy ** 2) ** 2 + 1e-6)

    # Compute motion tensor components with normalized dataterm
    J11 = reg_x * fxx ** 2 + reg_y * fxy ** 2
    J22 = reg_x * fxy ** 2 + reg_y * fyy ** 2
    J33 = reg_x * fxt ** 2 + reg_y * fyt ** 2
    J12 = reg_x * fxx * fxy + reg_y * fxy * fyy
    J13 = reg_x * fxx * fxt + reg_y * fxy * fyt
    J23 = reg_x * fxy * fxt + reg_y * fyy * fyt

    # Apply boundary condition
    J11 = set_boundary0(J11)
    J22 = set_boundary0(J22)
    J33 = set_boundary0(J33)
    J12 = set_boundary0(J12)
    J13 = set_boundary0(J13)
    J23 = set_boundary0(J23)

    return J11, J22, J33, J12, J13, J23

def gradient2(f, hx, hy):
    fxx = np.zeros_like(f)
    fyy = np.zeros_like(f)

    #f = f.transpose()
    # Compute second derivative with respect to x
    fxx[1:-1, 1:-1] = (f[1:-1, :-2] - 2 * f[1:-1, 1:-1] + f[1:-1, 2:]) / hx**2

    # Compute second derivative with respect to y
    fyy[1:-1, 1:-1] = (f[:-2, 1:-1] - 2 * f[1:-1, 1:-1] + f[2:, 1:-1]) / hy**2

    #return fxx.transpose(), fyy.transpose()
    return fxx,fyy

def set_boundary0(f):
    # Set the boundaries of the array to 0
    f[:, 0] = 0
    f[:, -1] = 0
    f[0, :] = 0
    f[-1, :] = 0
    return f



if __name__ == '__main__':
    import numpy as np
    # from scipy.ndimage import pad, gradient
    import h5py as h5
    from scipy.ndimage import convolve, gaussian_filter

    f1_level = np.array(h5.File('test_get_motion_tensor_gc.mat')['f1_level'], dtype=np.float64).transpose((1,0))
    J11 = np.array(h5.File('test_get_motion_tensor_gc.mat')['J11'], dtype=np.float64).transpose((1,0))
    J22 = np.array(h5.File('test_get_motion_tensor_gc.mat')['J22'], dtype=np.float64).transpose((1,0))
    J33 = np.array(h5.File('test_get_motion_tensor_gc.mat')['J33'], dtype=np.float64).transpose((1,0))
    J12 = np.array(h5.File('test_get_motion_tensor_gc.mat')['J12'], dtype=np.float64).transpose((1,0))
    J13 = np.array(h5.File('test_get_motion_tensor_gc.mat')['J13'], dtype=np.float64).transpose((1,0))
    J23 = np.array(h5.File('test_get_motion_tensor_gc.mat')['J23'], dtype=np.float64).transpose((1,0))
    hx = np.array(h5.File('test_get_motion_tensor_gc.mat')['hx'], dtype=np.float64)
    hy = np.array(h5.File('test_get_motion_tensor_gc.mat')['hy'], dtype=np.float64)
    tmp = np.array(h5.File('test_get_motion_tensor_gc.mat')['tmp'], dtype=np.float64).transpose((1,0))

    (j11 ,j22 ,j33 ,j12 ,j13 ,j23) = get_motion_tensor_gc(f1_level ,tmp ,hx ,hy)
    print('done')
    def mse(m ,n):
        return np.sqrt(np.mean(( m -n )**2))
    def error_rate(error ,m):
        return error / (np.max(m) - np.min(m)) * 100
    def error_rate_2(m ,n):
        return np.max(np.abs( m -n)) / (np.max(n) - np.min(n))

    # print("J11 error: %f, error rate: %f " %(mse(j11 ,J11) ,error_rate_2(j11 ,J11)) )
    # print("J22 error: %f, error rate: %f " % (mse(j22, J22), error_rate_2(j22, J22)))
    # print("J33 error: %f, error rate: %f " % (mse(j33, J33), error_rate_2(j33, J33)))
    # print("J12 error: %f, error rate: %f " % (mse(j12, J12), error_rate_2(j12, J12)))
    # print("J13 error: %f, error rate: %f " % (mse(j13, J13), error_rate_2(j13, J13)))
    # print("J23 error: %f, error rate: %f " % (mse(j23, J23), error_rate_2(j23, J23)))