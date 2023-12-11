import numpy as np

class Barrier():
    # type = {'up', 'down', 'na'}
    def __init__(self, H_in, H_out, ki_type, ko_type):
        self._H_in = H_in
        self._H_out = H_out
        self._ki_type = ki_type
        self._ko_type = ko_type
        self._ki = False
        self._ko = False

    @property
    def H_in(self):
        return self._H_in

    @property
    def H_out(self):
        return self._H_out

    @property
    def ki_type(self):
        return self._ki_type
    
    @property
    def ko_type(self):
        return self._ko_type

    @property
    def ki(self):
        return self._ki

    @property
    def ko(self):
        return self._ko
    
    def reset(self):
        self._ki = False
        self._ko = False
    
    def test_ki(self, arr_S):
        self.reset()
        match self.ki_type:
            case 'up':
                if np.max(arr_S) >= self.H_in:
                    self._ki = True
                return self.ki
            case 'down':
                if np.min(arr_S) <= self.H_in:
                    self._ki = True
                return self.ki
            case 'na':
                return None

    def test_ko(self, arr_S):
        self.reset()
        match self.ko_type:
            case 'up':
                if np.max(arr_S) >= self.H_out:
                    self._ko = True
                return self.ko
            case 'down':
                if np.min(arr_S) <= self.H_out:
                    self._ko = True
                return self.ko
            case 'na':
                return None
    