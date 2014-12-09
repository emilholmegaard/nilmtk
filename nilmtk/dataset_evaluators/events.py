from __future__ import print_function, division
from ..preprocessing import Clip
from nilmtk.exceptions import TooFewSamplesError, WrongResolution
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from astropy.convolution.boundary_extend import DTYPE

class Events(object):
    resolutions = ['minute','hour','day','week','month']
       
    def __init__(self, meter=None):
        self.meter = meter
        self.events = None
        
    def get_events(self, resolution='minute'):
        if not resolution in self.resolutions:
            raise WrongResolution('Resolution may be: '+str(self.resolutions))
        
        if not self.events:
            e = self.get_events_raw()
            self.events = e
        e = self.events
        if resolution == 'week':
            e = e.groupby(lambda x: (x.week)).count()
        if resolution == 'day':
            e = e.groupby(lambda x: (x.day)).count()
        if resolution == 'hour':
            e = e.groupby(lambda x: (x.hour)).count()
        if resolution == 'minute':
            e = e.groupby(lambda x: (x.minute)).count()
    
        e = pd.DataFrame({'T' : pd.Series(data=e.index, dtype='int32'), 'E' : pd.Series(data=e.values, dtype='int32')})
        e = e.sort(columns='T')
        
        return e
    
            
    def plot(self, resolution='minute', path=None):
        """
        Parameters
        ----------
        resolution : week, day, hour, minute
        path : save figure to path
        Returns
        ----------
        ax : matplotlib.axes, optional
        """ 
        if not resolution in self.resolutions:
            raise WrongResolution('Resolution may be: '+str(self.resolutions))    
        data = self.get_events(resolution)  
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.bar(data.index, data['E'], width=1)
        ax.set_ylabel('Events')
        ax.set_title('Events by '+resolution)
        
        if not path is None:
            fig = ax.get_figure()
            fig.savefig(path)
            plt.clf()
        return ax

            
    def get_events_raw(self):
        '''
        Parameters
        -------
        meter : ElecMeter for which to find events, an event is either delta changes higher than 10 or lower than 10
        Returns
        -------
        df : pd.Series timestamp, true for event
        '''
        
        DATA_THRESHOLD = 10
        
        serie = []
        for chunk in self.meter.power_series(preprocessing=[Clip()]):
            d = chunk[chunk.diff() > DATA_THRESHOLD].dropna()
            serie.append(d)
            d = chunk[chunk.diff() < DATA_THRESHOLD*-1].dropna()
            serie.append(d)
         
        try:
            df = pd.DataFrame(serie)#, dtype=np.bool
            df = df.T
            df = df.iloc[:,1]
            return df
        except:
            raise TooFewSamplesError("Cannot calculate events because"
                                     " data does not contain events with a difference between +-{:d}."
                                     .format(DATA_THRESHOLD))
       
            
    