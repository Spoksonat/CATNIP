import numpy as np
from scipy import signal as sig

class Inline:
    def __init__(self, I_ref, I_samp, dict_params):
        
        self.dict_params = dict_params
        self.I_ref = I_ref
        self.I_samp = I_samp
        self.FFcorr()

    def FFcorr(self):
    
        self.T = self.I_samp/self.I_ref