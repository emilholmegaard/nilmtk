from __future__ import print_function, division
from nilmtk.preprocessing import Clip
from os.path import isfile
import matplotlib.pyplot as plt
import pandas as pd 
import pandas.io.parsers as parsers
import numpy as np 
from nilmtk.plots import plot_3d
from nilmtk.dataset_evaluators.weatherimpact import WeatherImpact
from nilmtk.dataset_evaluators.events import Events

class ExternalImpact(object):

    def __init__(self, dataset=None,meter=None, external_data_path=None):
        self.dataset = dataset
        self.externaldata = self.__get_data_from_file(external_data_path)
        self.freq
        self.meter = meter
        self.weatherdata = self.__get_weather_data()
        try:
            self.events = self.__get_events_data()
        except:
            self.events = pd.DataFrame()
        self.powerdata = None
        self.data = pd.DataFrame()

        
    def __get_power_data(self, meter):
        '''
        Returns
        -------
        data : pd.Series timestamp, consumption
        '''
        data = meter.power_series_all_data(preprocessing=[Clip()])#,sections=meter.good_sections()
        return data
    
    def __get_weather_data(self):
        '''
        Get weather data from weather impact.
        '''
        weather = WeatherImpact(dataset=self.dataset, meter=self.meter)
        df = weather.get_weather_data()
        df = df.resample(rule=self.freq)
        return df
    
    def __get_events_data(self):
        '''
        Get event data.
        '''
        event = Events(self.meter)
        e = event.get_events_raw()
        df = pd.DataFrame(data={'event':e.values}, index=e.index, dtype=np.bool)
        df = df.resample(rule=self.freq, how='count')
        return df
    
    def get_data(self):
        if self.data.empty:
            data = self.__get_power_data(self.meter)
            df = pd.DataFrame(data={"power":data.values}, index=data.index, dtype=np.float32).dropna()
            self.powerdata =df
            df=df.resample(rule=self.freq)
            #Join the two data sets and remove NaN, for having proper indexing
            df = self.externaldata.join(df).dropna()
            df = df.join(self.weatherdata['temperature']).dropna()
            df = df.join(self.events).fillna(0)
            self.data = df
        return self.data
    
    def plot(self, type='temperature',regression=False, path=None):
        df = self.get_data()
        
        if type == 'event':
            ax = plot_3d(x=df['power'],y=df['value'],z=df['event'],regression=regression, xlabel='Power', ylabel='Pallet Movements', zlabel='Events')
        else:
            ax = plot_3d(x=df['power'],y=df['value'],z=df['temperature'],regression=regression, xlabel='Power', ylabel='Pallet Movements', zlabel='Temperature')
        
        if not path is None:
            fig = ax.get_figure()
            fig.savefig(path, bbox_inches='tight')
            plt.clf()    
        
        return ax
    
    def __get_data_from_file(self, path):
        assert isfile(path)
        if '.csv' in path:
            #Except date(dd/mm/yyyy);value
            df = parsers.read_csv(path, sep=';', index_col=0, parse_dates=True, dayfirst=True)
            self.freq = str(pd.infer_freq(df.index))
            df.index = pd.to_datetime(df.index.values, unit='s', utc=True)
            
            return df