def warpingDepth(eta, levels, m, n):
    eta = float(eta)
    levels = int(levels)
    m = int(m)
    n = int(n)
    
    min_dim = min(m, n)
    warping_depth = 0
    d = warping_depth

    for i in range(1, levels + 1):
        warping_depth += 1
        min_dim *= eta
        if round(min_dim) < 10:
            break
        d = warping_depth
    return d

if __name__ == '__main__':
    import h5py as h5
    import numpy as np
    eta = np.array(h5.File('test_warpingDepth.mat')['eta'], dtype=np.float64)
    levels = np.array(h5.File('test_warpingDepth.mat')['levels'])

    m = np.array(h5.File('test_warpingDepth.mat')['m'], dtype=np.float64)
    gt_max_level_y = np.array(h5.File('test_warpingDepth.mat')['max_level_y'], dtype=np.float64)

    max_level_y = warpingDepth(eta,levels,m,m)
    error = max_level_y - gt_max_level_y
    print('Error:  ',error,'Error Rate:',error / gt_max_level_y)