import numpy as np
import numba as nb
import time
from joblib import Parallel, delayed
OMEGA = 1.95

def nonlinearity_smoothness(psi_smooth, du, u, dv, v, a, hx, hy):
    eps = 1e-5
    u_full = u + du
    v_full = v + dv
    ux, uy = np.gradient(u_full, hx, hy)
    vx, vy = np.gradient(v_full, hx, hy)

    grad_norm_sq = ux ** 2 + uy ** 2 + vx ** 2 + vy ** 2
    psi_smooth[:, :] = a * (np.maximum(grad_norm_sq, eps)) ** (a - 1)

    return psi_smooth


def apply_boundary_conditions(du, dv):
    du[:, 0], du[:, -1] = du[:, 1], du[:, -2]
    dv[:, 0], dv[:, -1] = dv[:, 1], dv[:, -2]
    du[0, :], du[-1, :] = du[1, :], du[-2, :]
    dv[0, :], dv[-1, :] = dv[1, :], dv[-2, :]

    return du, dv


@nb.njit(parallel=False, fastmath=True)
def update_du_dv(nx, ny, n_channels, a_smooth, alpha_stencil, data_weight, OMEGA,
                 u_flat, v_flat, du_flat, dv_flat, psi_flat, psi_smooth_flat,
                 j11_flat, j12_flat, j13_flat, j22_flat, j23_flat, weight_is_smaller=True):
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            denom_u = 0.0
            denom_v = 0.0
            num_u = 0.0
            num_v = 0.0

            idx = j + i * ny
            s_idx = np.array([
                j + (i - 1) * ny,
                j + (i + 1) * ny,
                (j + 1) + i * ny,
                (j - 1) + i * ny
            ])

            if a_smooth != 1:
                for d in range(4):
                    tmp = 0.5 * (psi_smooth_flat[idx] + psi_smooth_flat[s_idx[d]]) * alpha_stencil[d]
                    num_u += tmp * (u_flat[s_idx[d]] + du_flat[s_idx[d]] - u_flat[idx])
                    num_v += tmp * (v_flat[s_idx[d]] + dv_flat[s_idx[d]] - v_flat[idx])
                    denom_u += tmp
                    denom_v += tmp
            else:
                for d in range(4):
                    num_u += alpha_stencil[d] * (u_flat[s_idx[d]] + du_flat[s_idx[d]] - u_flat[idx])
                    num_v += alpha_stencil[d] * (v_flat[s_idx[d]] + dv_flat[s_idx[d]] - v_flat[idx])
                    denom_u += alpha_stencil[d]
                    denom_v += alpha_stencil[d]

            # 预计算基础索引偏移
            base_idx = j + i * ny

            # 使用向量化操作处理channels
            if weight_is_smaller:
                for k in range(n_channels):
                    nd_idx = base_idx + k * (nx * ny)
                    psi_val = psi_flat[nd_idx]

                    num_u -= data_weight[k] * psi_val * (j13_flat[nd_idx] + j12_flat[nd_idx] * dv_flat[idx])
                    denom_u += data_weight[k] * psi_val * j11_flat[nd_idx]
                    denom_v += data_weight[k] * psi_val * j22_flat[nd_idx]

                du_kp1 = num_u / denom_u
                du_flat[idx] = (1 - OMEGA) * du_flat[idx] + OMEGA * du_kp1

                for k in range(n_channels):
                    nd_idx = base_idx + k * (nx * ny)
                    num_v -= data_weight[k] * psi_flat[nd_idx] * (j23_flat[nd_idx] + j12_flat[nd_idx] * du_flat[idx])

                dv_kp1 = num_v / denom_v
                dv_flat[idx] = (1 - OMEGA) * dv_flat[idx] + OMEGA * dv_kp1

            else:
                for k in range(n_channels):
                    nd_idx = base_idx + k * (nx * ny)
                    psi_val = psi_flat[nd_idx]

                    num_u -= data_weight[nd_idx] * psi_val * (j13_flat[nd_idx] + j12_flat[nd_idx] * dv_flat[idx])
                    denom_u += data_weight[nd_idx] * psi_val * j11_flat[nd_idx]
                    denom_v += data_weight[nd_idx] * psi_val * j22_flat[nd_idx]

                du_flat[idx] = (1 - OMEGA) * du_flat[idx] + OMEGA * du_kp1

                for k in range(n_channels):
                    nd_idx = base_idx + k * (nx * ny)
                    num_v -= data_weight[nd_idx] * psi_flat[nd_idx] * (
                                j23_flat[nd_idx] + j12_flat[nd_idx] * du_flat[idx])

                dv_kp1 = num_v / denom_v
                dv_flat[idx] = (1 - OMEGA) * dv_flat[idx] + OMEGA * dv_kp1

    return du_flat, dv_flat


def level_solver_acc(plhs, prhs):
    nrhs = len(prhs)

    j11 = prhs[0]
    j22 = prhs[1]
    j33 = prhs[2]
    j12 = prhs[3]
    j13 = prhs[4]
    j23 = prhs[5]

    data_weight = prhs[6]
    u = prhs[7]
    v = prhs[8]
    alpha = prhs[9]

    iterations = int(prhs[10])
    update_lag = int(prhs[11])
    verbose = int(prhs[12])
    a_data = prhs[13]
    a_smooth = int(prhs[14])
    hx = float(prhs[15]) if nrhs == 17 else 1.0
    hy = float(prhs[16]) if nrhs == 17 else 1.0

    ny, nx = u.shape
    n_channels = 1 if j11.ndim < 3 else j11.shape[2]

    du = np.zeros((ny, nx), dtype=np.float64)
    dv = np.zeros((ny, nx), dtype=np.float64)
    psi = np.ones_like(j11, dtype=np.float64)
    psi_smooth = np.ones((ny, nx), dtype=np.float64)

    alpha_stencil = np.array([
        alpha[0] / hx ** 2,
        alpha[0] / hx ** 2,
        alpha[1] / hy ** 2,
        alpha[1] / hy ** 2
    ], dtype=np.float64).squeeze()

    data_weight = np.array(data_weight, dtype=np.float64)
    weight_ny, weight_nx = data_weight.shape
    weight_size = weight_ny * weight_nx
    data_weight = data_weight.squeeze()

    a_smooth = np.array(a_smooth, dtype=np.float64).squeeze()

    time_consume_by_update_du_dv = 0
    time_consume_by_apply_boundary_conditions = 0
    if weight_size < ny * nx:
        for iteration in range(1, iterations + 1):
            if iteration % update_lag == 0:
                tmp = (j11 * du ** 2 +
                       j22 * dv ** 2 +
                       j23 * dv +
                       2 * j12 * du * dv +
                       2 * j13 * du +
                       j23 * dv +
                       j33)
                # print(f'iteration:{iteration}')
                psi[:, :] = a_data * (np.maximum(tmp, 0) + 1e-5) ** (a_data - 1)
                if a_smooth != 1:
                    psi_smooth = nonlinearity_smoothness(psi_smooth, du, u, dv, v, a_smooth, hx, hy)
            # time_start = time.time()
            du, dv = apply_boundary_conditions(du, dv)
            # time_consume_by_apply_boundary_conditions += time.time() - time_start

            # flatten arrays before sending to update_du_dv
            u_flat = u.ravel()
            v_flat = v.ravel()
            du_flat = du.ravel()
            dv_flat = dv.ravel()
            psi_flat = psi.ravel()
            j11_flat = j11.ravel()
            j12_flat = j12.ravel()
            j13_flat = j13.ravel()
            j22_flat = j22.ravel()
            j23_flat = j23.ravel()

            # time_start = time.time()
            du_flat, dv_flat = update_du_dv(nx=nx, ny=ny, n_channels=n_channels,
                                            a_smooth=a_smooth, alpha_stencil=alpha_stencil,
                                            data_weight=data_weight, OMEGA=OMEGA,
                                            u_flat=u_flat, v_flat=v_flat, du_flat=du_flat, dv_flat=dv_flat,
                                            psi_flat=psi_flat, psi_smooth_flat=psi_smooth.ravel(),
                                            j11_flat=j11_flat, j12_flat=j12_flat, j13_flat=j13_flat,
                                            j22_flat=j22_flat, j23_flat=j23_flat)
            # if iteration == 1:
            #     time_compiled = time.time()
            # time_consume_by_update_du_dv += time.time() - time_start
            # if iteration == 1:
            #     compiling_time = time_compiled - time_start
            #     print("time consume by update_du_dv(with compilation): ", compiling_time)
            du = du_flat.reshape((ny, nx))
            dv = dv_flat.reshape((ny, nx))
    else:
        for iteration in range(1, iterations + 1):
            if iteration % update_lag == 0:
                tmp = (j11 * du ** 2 +
                       j22 * dv ** 2 +
                       j23 * dv +
                       2 * j12 * du * dv +
                       2 * j13 * du +
                       j23 * dv +
                       j33)
                print(f'iteration:{iteration}')
                psi[:, :] = a_data * (np.maximum(tmp, 0) + 1e-5) ** (a_data - 1)
                if a_smooth != 1:
                    psi_smooth = nonlinearity_smoothness(psi_smooth, du, u, dv, v, a_smooth, hx, hy)

                du, dv = apply_boundary_conditions(du, dv)

                # flatten arrays before sending to update_du_dv
                u_flat = u.ravel()
                v_flat = v.ravel()
                du_flat = du.ravel()
                dv_flat = dv.ravel()
                psi_flat = psi.ravel()
                j11_flat = j11.ravel()
                j12_flat = j12.ravel()
                j13_flat = j13.ravel()
                j22_flat = j22.ravel()
                j23_flat = j23.ravel()

                # time_start = time.time()
                du_flat, dv_flat = update_du_dv(nx=nx, ny=ny, n_channels=n_channels,
                                                a_smooth=a_smooth, alpha_stencil=alpha_stencil,
                                                data_weight=data_weight, OMEGA=OMEGA,
                                                u_flat=u_flat, v_flat=v_flat, du_flat=du_flat, dv_flat=dv_flat,
                                                psi_flat=psi_flat, psi_smooth_flat=psi_smooth.ravel(),
                                                j11_flat=j11_flat, j12_flat=j12_flat, j13_flat=j13_flat,
                                                j22_flat=j22_flat, j23_flat=j23_flat, weight_is_smaller=False)
                # if iteration == 1:
                #     time_compiled = time.time()
                # time_consume_by_update_du_dv += time.time() - time_start
                # if iteration == 1:
                #     compiling_time = time_compiled - time_start
                #     print("time consume by update_du_dv(with compilation): ", compiling_time)
                du = du_flat.reshape((ny, nx))
                dv = dv_flat.reshape((ny, nx))

    plhs[0] = du
    plhs[1] = dv
    # print("time consume by apply_boundary_conditions: ", time_consume_by_apply_boundary_conditions)
    # print("time consume by update_du_dv(without compile): ", time_consume_by_update_du_dv - compiling_time)
    return du, dv


if __name__ == "__main__":
    import h5py as h5
    import warnings

    warnings.filterwarnings("ignore")

    J11 = np.array(h5.File('test__data.mat')['J11'], dtype=np.float64)
    J22 = np.array(h5.File('test__data.mat')['J22'], dtype=np.float64)
    J33 = np.array(h5.File('test__data.mat')['J33'], dtype=np.float64)
    J12 = np.array(h5.File('test__data.mat')['J12'], dtype=np.float64)
    J13 = np.array(h5.File('test__data.mat')['J13'], dtype=np.float64)
    J23 = np.array(h5.File('test__data.mat')['J23'], dtype=np.float64)

    weight_level = np.array(h5.File('test__data.mat')['weight_level'])
    u = np.array(h5.File('test__data.mat')['u'])
    v = np.array(h5.File('test__data.mat')['v'])
    alpha = np.array(h5.File('test__data.mat')['ALPHA'])
    iterations = 50
    update_lag = np.array(h5.File('test__data.mat')['update_lag'])
    a_data = np.array(h5.File('test__data.mat')['a_data'])
    a_smooth = np.array(h5.File('test__data.mat')['a_smooth'])
    # print("a_smooth:", a_smooth)
    hx = np.array(h5.File('test__data.mat')['hx'], dtype=np.float64)
    hy = np.array(h5.File('test__data.mat')['hy'], dtype=np.float64)

    du_expected = np.array(h5.File('test__data.mat')['du'], dtype=np.float64)  # .transpose()
    dv_expected = np.array(h5.File('test__data.mat')['dv'], dtype=np.float64)  # .transpose()

    plhs = [None, None]
    prhs = [J11, J22, J33, J12, J13, J23, weight_level, u, v,
            alpha, iterations, update_lag, 0, a_data, a_smooth, hx, hy]

    import time

    print("Running level_solver...")
    st = time.time()
    print(f'start time: {st}')
    level_solver_acc(plhs, prhs)
    print(f'total time: {time.time() - st}')

    du, dv = plhs  # equals to (du, dv) = level_solver(...)
    # print("du:", du)
    # print("du_expected:", du_expected)
    print(f'du_expected - du: {np.max(du_expected - du), np.min(du_expected - du)}')
    print(
        f'error rate in du: {np.max(np.abs(du_expected - du)) / (np.max(np.abs(du_expected) - np.min(np.abs(du_expected)))) * 100:.2f}%')
    # print("dv:", dv)
    # print("dv_expected:", dv_expected)
    print(f'dv_expected - dv: {np.max(dv_expected - dv), np.min(dv_expected - dv)}')
    print(
        f'error rate in dv: {np.max(np.abs(dv_expected - dv)) / (np.max(np.abs(dv_expected) - np.min(np.abs(dv_expected)))) * 100:.2f}%')