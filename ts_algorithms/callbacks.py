import tomosipo as ts
import torch
import math
import time
from abc import ABC, abstractmethod
from datetime import timedelta

def call_all_callbacks(callbacks, x, iteration):
    stop = False
    for callback in callbacks:
        stop = stop or callback(x, iteration)
    return stop
    
    
class TrackMetricCb(ABC):
    def __init__(self, keep_best_x=False, early_stopping_iterations=None, lower_is_better=True):
        self._metric_log = []
        self._keep_best_x = keep_best_x
        self._early_stopping_iterations = early_stopping_iterations
        self._lower_is_better = lower_is_better
        if self._lower_is_better:
            self._best_score = float('inf')
        else:
            self._best_score = float('-inf')
        self._best_iteration = None

    @abstractmethod
    def calc_metric(self, x, iteration):
        pass

    def __call__(self, x, iteration):
        self._metric_log.append(self.calc_metric(x, iteration).cpu())
        
        # Update the best score and iteration if neccessary
        if ((self._lower_is_better and self._metric_log[-1] < self._best_score)
            or (not self._lower_is_better and self._metric_log[-1] > self._best_score)):
            self._best_score = self._metric_log[-1]
            self._best_iteration = iteration
            
            if self._keep_best_x:
                self._best_x = x.clone()
                
        # If you are using early stopping, and the score hasn't improved for
        # self._early_stopping_iterations iterations, signal that the algorithm
        # should stop
        if (self._keep_best_x
            and self._early_stopping_iterations is not None
            and iteration - self._best_iteration >= self._early_stopping_iterations):
            return True
        else:
            return False
                    

    @property
    def metric_log(self):
        return self._metric_log
    
    @property    
    def best_score(self):
        return self._best_score
        
    @property    
    def best_iteration(self):
        return self._best_iteration
        
    @property    
    def best_x(self):
        return self._best_x


class TrackMseCb(TrackMetricCb):
    def __init__(self, x_reference, volume_mask=None, keep_best_x=False, early_stopping_iterations=None):
        super().__init__(keep_best_x=keep_best_x, early_stopping_iterations=early_stopping_iterations)
        self._x_reference = x_reference
        self._volume_mask = volume_mask
        if self._volume_mask is not None:
            self._mask_sum = torch.sum(volume_mask)
        
    def calc_metric(self, x, iteration):
        squared_error = (x - self._x_reference) ** 2
        if self._volume_mask is not None:
            return torch.sum(squared_error * self._volume_mask) / self._mask_sum
        else:
            return torch.mean(squared_error)


class TrackResidualMseCb(TrackMetricCb):
    def __init__(self, A, y_reference, projection_mask=None, keep_best_x=False, early_stopping_iterations=None):
        super().__init__(keep_best_x=keep_best_x, early_stopping_iterations=early_stopping_iterations)
        self._y_reference = y_reference
        self._A = A
        self._projection_mask = projection_mask
        if self._projection_mask is not None:
            self._mask_sum = torch.sum(projection_mask)
        
    def calc_metric(self, x, iteration):
        squared_residual_error = (self._A(x) - self._y_reference) ** 2
        if self._projection_mask is not None:
            return torch.sum(squared_residual_error * self._projection_mask) / self._mask_sum
        else:
            return torch.mean(squared_residual_error)


class TimeoutCb():
    def __init__(self, duration):
        if isinstance(duration, timedelta):
            self._duration = duration.total_seconds()
        else:
            self._duration = duration
        self.reset_start_time()
            
    def reset_start_time(self):
        self._start_time = time.time()
    
    def __call__(self, x, iteration):
        if time.time()-self._start_time >= self._duration:
            return True
        else:
            return False
