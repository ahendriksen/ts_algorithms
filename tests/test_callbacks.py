import pytest
import numpy as np
import torch
import tomosipo as ts
from ts_algorithms import sirt, TrackResidualMseCb, TrackMseCb


def test_callbacks_GPU():
    iterations = 5

    vg = ts.volume(shape=32)
    pg = ts.parallel(angles=32, shape=48)
    A = ts.operator(vg, pg)

    x = torch.zeros(*A.domain_shape)
    x[4:28, 4:28, 4:28] = 1.0
    x[12:22, 12:22, 12:22] = 0.0
    x = x.cuda()
    y = A(x)
    
    mse_callback = TrackMseCb(x)
    res_callback = TrackResidualMseCb(A, y)
    x_sirt = sirt(A, y, num_iterations=iterations, callbacks=(mse_callback,res_callback))
    
def test_early_stopping():
    x = torch.ones((1, 1, 1))

    mse_callback = TrackMseCb(x, keep_best_x=True, early_stopping_iterations=2)
    assert mse_callback(x*5, 0) == False
    assert mse_callback(x*3, 1) == False
    assert mse_callback(x*1.5, 2) == False
    assert mse_callback(x*2, 3) == False
    assert mse_callback(x*4, 4) == True
    
    assert mse_callback.best_iteration == 2
    assert abs(mse_callback.best_score - 0.5**2) < ts.epsilon
    assert mse_callback.best_x[0,0,0] == 1.5
