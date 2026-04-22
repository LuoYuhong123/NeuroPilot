from demotion.get_displacement import get_displacement
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import time
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib



def process_frame(i, c_ref, c, varargin):
    return get_displacement(c_ref, c[:, :, i], varargin)


def process_frame_2(i, c_ref, c, varargin):
    return get_displacement(c[:, :, i], c_ref, varargin)


def get_displacements_old(c, c_ref, varargin):
    m, n, t = c.shape

    w = np.zeros((m, n, 2, t), dtype=np.float64)
    results = Parallel(n_jobs=-1)(
        delayed(process_frame)(i, c_ref, c, varargin) for i in range(t)
    )
    for i, w_tmp in enumerate(results):
        w[:, :, :, i] = w_tmp

    return w


def get_displacements(c, c_ref, varargin):
    m, n, t = c.shape
    w = np.zeros((m, n, 2, t), dtype=np.float64)

    # 进度条与 joblib 并行集成
    # with tqdm_joblib(tqdm(total=t, desc="Processing frames", unit="frm")):
    with tqdm_joblib( tqdm(total=t, desc="➡️ Processing frames", unit="frm", colour="cyan") ):
        results = Parallel(n_jobs=-1)(
            delayed(process_frame)(i, c_ref, c, varargin) for i in range(t)
        )

    for i, w_tmp in enumerate(results):
        w[:, :, :, i] = w_tmp
    return w



def get_displacements_2(c, c_ref, varargin):
    m, n, t = c.shape

    w = np.zeros((m, n, 2, t), dtype=np.float64)


    results = Parallel(n_jobs=-1)(
        delayed(process_frame_2)(i, c_ref, c, varargin) for i in range(t)
    )


    for i, w_tmp in enumerate(results):
        w[:, :, :, i] = w_tmp

    return w