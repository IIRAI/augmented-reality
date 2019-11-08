
import numpy as np

class FadingFilter:
    '''
    Implement a fading filter of the first and second order.  
    input:  
    -`filter_memory`: float (0, 1), memory factor of the filter.  
    -`sample_time`: sample time of the input signal.  
    '''

    def __init__(self, filter_memory: float, sample_time: float):
        self.homography_old = np.zeros([3, 3])
        self.Dhomography_old = np.zeros([3, 3])
        self.beta = filter_memory
        self.Ts = sample_time

    def I_order_ff(self, homography):
        '''
        First order fading filter:
            x_{n} = x_{n-1} + (1 - filter_memory) * (x_{new} - x_{n-1})
        '''
        # filter
        H_dif = homography - self.homography_old
        gain = 1 - self.beta
        homography_filtered = self.homography_old + (gain * H_dif)
        # update filter homography, old step value
        self.homography_old = homography_filtered
        return homography_filtered

    def II_order_ff(self, homography):
        '''
        Second order fading filter:  
            model prediction:  
                p = x_{n-1} + (dx_{n-1} * Ts)  
            error between measure and prediction:  
                x_dif = x_{new} - p  
            Filter:
                x_{n}  = p + (1 - filter_memory^2) * x_dif  
                dx_{n} = dx_{n-1} + [(1 - filter_memory)^2 / Ts] * x_dif  

        '''
        # filter
        gain = 1 - np.power(self.beta, 2)
        Dgain = np.power((1 - self.beta), 2) / self.Ts
        # model prediction
        prediction = self.homography_old + (self.Dhomography_old * self.Ts)
        H_dif = homography - prediction
        homography_filtered  = prediction + (gain * H_dif)
        Dhomography_filtered = self.Dhomography_old + (Dgain * H_dif)
        # update filter homography, old step value
        self.homography_old = homography_filtered
        self.Dhomography_old = Dhomography_filtered
        return homography_filtered
