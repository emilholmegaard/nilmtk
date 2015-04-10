from __future__ import print_function, division
from _warnings import warn
from nilmtk.dataset_evaluators import Evaluate_Meter_VSD

class Evaluate_VSD(object):

    def __init__(self, metergroup, path=None):
        self.metergroup = metergroup
        self.path = path
        
    
    def evaluate(self):
        """
        Parameters
        ----------
        
        Returns
        -------

        Raises
        ------
        """
        
        for meter in self.metergroup.meters:
            e = Evaluate_Meter_VSD(meter)
            if e.is_vsd():
                warn('Meter: {} is a possible variable speed drive.'.format(meter), RuntimeWarning)
                
            if self.path:
                e.plot(self.path+'/meter_{}.png'.format(meter))
            
        