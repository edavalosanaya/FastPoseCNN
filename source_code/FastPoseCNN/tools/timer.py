import numpy as np
import time
import functools
import torch

# Useful link here: https://stackoverfow.com/questions/14636350/toggling-decorators

class TimerDecorator(object):
    """ Decorator for timing functions """

    def __init__(self, name):
        self.name = name
        self.runtimes = []
        self.enabled = False

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __call__(self, function):

        @functools.wraps(function)
        def wrapper(*args, **kwargs):

            # If the decorator is enabled:
            if self.enabled:
            
                # Obtain time information before
                self.start.record()
                
                # Perform the function
                result = function(*args, **kwargs)

                # Obtain time information after
                self.end.record()

                # Waits for everything to finish running 
                torch.cuda.synchronize()

                # Calculate runtime in milliseconds
                runtime = self.start.elapsed_time(self.end)
                
                # Storing runtime
                self.runtimes.append(runtime)

            else:

                # Else return the simple results of the target function
                return function(*args, **kwargs)
            
            return result
        return wrapper
        
    @property
    def average(self):
        self._average = np.mean(np.array(self.runtimes))
        return self._average

    @property
    def fps(self):
        self._fps = 1000 / self.average
        return self._fps

    def clear(self):
        self.runtimes.clear()