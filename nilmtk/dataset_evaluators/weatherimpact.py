'''
Created on Nov 19, 2014

@author: em
'''
from __future__ import print_function, division
from nilmtk.exceptions import UnableToGetWeatherData
from nilmtk.plots import calc_regression, plot_3d
from nilmtk.preprocessing import Clip
from nilmtk.dataset_evaluators.events import Events
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import urllib2, json, time, datetime


class WeatherImpact:
    def __init__(self, dataset, meter):
        
        self.data = dataset
        self.meter = meter
        self.weatherstationid = 0
        self.weatherdata = pd.DataFrame()
        self.correlations = None
        self.raw = None
        
        
    def get_weather_data(self):
        """
        Parameters
        -------
        Returns
        -------
        pd.DataFrame : [timestamp, humidity, temperature, wind]
    
        """
        if self.weatherdata.empty:
            self.__get_weather_data()
        
        return self.weatherdata
            

    def get_correlation(self):
        """
        Parameters
        ----------
        Returns
        -------
        df : DataFrame correlation matrix 
        """
        
        if self.weatherdata.empty:
            self.__get_weather_data()
        
        if not self.correlations:
            t = self.meter.power_series_all_data(preprocessing=[Clip()])
            p = pd.DataFrame(t, dtype=np.float32)
            p.columns = p.columns.droplevel(1)
            
            p=p.resample('D', loffset='20h')
            w = self.weatherdata.resample('D' ,loffset='20h')
            
            self.raw = w.join(p).dropna()
            self.correlations = self.raw.corr(method='pearson', min_periods=1)
            self.raw = self.raw.join(self.__get_events_data()).dropna()
        
        return self.correlations
        
    def plot(self, type='',regression=False, path=None):
        if self.correlations is None:
            self.get_correlation()
        
        if type=='event':
            ax = plot_3d(x=self.raw['power'],y=self.raw['temperature'],z=self.raw['event'],regression=regression, xlabel='Power', ylabel='Temperature', zlabel='Events')
        else:
            ax = self.__plot_power_temperature(regression)
        
        if not path is None:
            fig = ax.get_figure()
            fig.savefig(path, bbox_inches='tight')
            plt.clf()
            
        return ax
   
    def __plot_power_temperature(self, regression):
        fig = plt.figure()      
        ax = plt.subplot(111)
        if regression:
            predict_y , r_squared = calc_regression(self.raw['power'],self.raw['temperature'])
            reg = ax.plot(self.raw['power'], predict_y, 'k-', label='Power vs Temperature, R2:'+str(r_squared))
            ax.legend(handles=[reg[0]], loc=4)
        ax.scatter(x=self.raw['power'],y=self.raw['temperature'])
        return ax
     
    def __get_weather_data(self):
        """
        Parameters
        ----------
        resolution : hour, day, week, month
    
        Returns
        -------
        events : float [0,1]
            The number of transistions
    
        Raises
        ------
        urllib2.HTTPError
        UnableToGetWeatherData
        """
        
        if self.weatherstationid > 0:
            #TODO remove meter here - use parameter
            data=[]
            for section in self.meter.good_sections():
                start = str(int((self.__roundTime(dt=section.start) - datetime.datetime(1970,1,1,0,0)).total_seconds()))
                end = str(int((self.__roundTime(dt=section.end) - datetime.datetime(1970,1,1,0,0)).total_seconds()))#str(section.end.to_pydatetime.im_self.value)[:-9] 
                if self.__roundTime(dt=section.end)-self.__roundTime(dt=section.start) > datetime.timedelta(hours=1):
                    data.append(self.__get_openweathermap_data(start,end))
                
            self.weatherdata = pd.concat(data)
                
        else:
            self.__get_weatherstation()
            self.__get_weather_data()
        
                
    def __get_events_data(self):
        '''
        Get event data.
        '''
        try:
            event = Events(self.meter)
            e = event.get_events_raw()
            df = pd.DataFrame(data={'event':e.values}, index=e.index, dtype=np.bool)
            df = df.resample(rule='D' ,loffset='20h', how='sum')
        except:
            df = pd.DataFrame(data={'event':False}, index=self.weatherdata.index)
        return df
        
    def __get_openweathermap_data(self, start, end):
            """
        Parameters
        ----------
        start : unix timestamp
        end : unix timestamp
    
        Returns
        -------
        pd.DataFrame : [timestamp, humidity, temperature, wind]
        Raises
        ------
        UnableToGetWeatherData
        """
            #openweathermaps only provide hourly measures
            #dataurl = 'http://api.openweathermap.org/data/2.5/history/station?id='+str(self.weatherstationid)+'&type=hour&start='+start+'&end='+end
            dataurl = 'http://api.openweathermap.org/data/2.5/history/city?id='+str(self.weatherstationid)+'&type=hour&start='+start+'&end='+end
            
            response = urllib2.urlopen(dataurl)
            d = json.load(response)
            
            timestamp, temperature, wind, humidity = [],[],[],[]
            
            if d['cnt'] <= 0:
                raise UnableToGetWeatherData("WeatherData from OpenWeatherMap was empty!")
            for element in d["list"]:
                dt =  element["dt"]
                try:
                    #t = float(element["temp"]["v"] -272.15)
                    t = float(element["main"]["temp"] -272.15)
                except:
                    t = None
                try:
                    #w = float(element["wind"]["speed"]["v"])
                    w = float(element["wind"]["speed"])
                except:
                    w = None
                try:
                    #h = float(element["humidity"]["v"])
                    h = float(element["main"]["humidity"])
                except:
                    h = None
                        
                ts =  time.mktime(datetime.datetime.fromtimestamp(dt).timetuple())
                timestamp.append(ts)
                temperature.append(t)
                wind.append(w)
                humidity.append(h)
    
            df = pd.DataFrame(data={'temperature':temperature, 'wind':wind, 'humidity':humidity}, index=timestamp,  dtype=np.float32)
            # raw data isn't always sorted
            df = df.sort_index()
            # Convert the integer index column to timezone-aware datetime 
            df.index = pd.to_datetime(df.index.values, unit='s', utc=True)
            
            return df

    def __get_weatherstation(self):
        #Possible error with list or dict in metadata
        try:
            city = self.data.metadata.get("geo_location")[0].get("locality")
            country = self.data.metadata.get("geo_location")[0].get("country")
        except:
            city = self.data.metadata.get("geo_location").get("locality")
            country = self.data.metadata.get("geo_location").get("country")
        
        
        try:
            weatherurl = 'http://api.openweathermap.org/data/2.5/weather?q='+city+','+country
            response = urllib2.urlopen(weatherurl)
            w = json.load(response)
            #id=w.get("sys").get("id")
            id = w.get("id")
            self.weatherstationid = id
        except:
            raise UnableToGetWeatherData("Cannot find weather station in area of "+city+", "+country)
    
    def __roundTime(self,dt=None):
        """
        Round a timeframe object to any time laps in seconds
        dt : timeframe object, default now.
        """
        
        roundTo = 60
        
        if dt == None : 
            dt = datetime.datetime.now()
        else:
            dt = datetime.datetime(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second,dt.microsecond)
        seconds = (dt - dt.min).seconds
        # '//' is a floor division
        rounding = (seconds+roundTo/2) // roundTo * roundTo
        dt = dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)
        return dt