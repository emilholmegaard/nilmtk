'''
Created on Nov 19, 2014

@author: em
'''
from __future__ import print_function, division
from nilmtk.exceptions import UnableToGetWeatherData, WrongResolution
from nilmtk.preprocessing import Clip
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import urllib2, json, time, datetime
from scipy import stats


class WeatherImpact:
    def __init__(self, dataset, meter, resolution='hour'):
        resolutions = ['hour','day','week','month']
        if not resolution in resolutions:
            raise WrongResolution('Resolution may be: '+str(resolutions))
        
        self.data = dataset
        self.meter = meter
        self.weatherstationid = 0
        self.weatherdata = pd.DataFrame()
        self.correlations = None
        self.temperature = None
        self.correlations_temperature = None
        self.wind = None
        self.correlations_wind = None
        self.humidity = None
        self.correlations_humidity = None
        self.raw_group = None
        self.raw = None
        self.resolution = resolution
        
        if self.resolution == 'hour':
            self.dataresolution = 'minute'

        if self.resolution == 'day':
            self.dataresolution = 'hour'
        
        if self.resolution == 'week':
            self.dataresolution = 'day'
        
        if self.resolution == 'month':
            self.dataresolution = 'week'

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
            t = self.meter.power_series_all_data(preprocessing=[Clip()],sections=self.meter.good_sections())
            p = pd.DataFrame(t, dtype=np.float32)
            p.columns = p.columns.droplevel(1)
            
            
            #data = self.weatherdata.join(p).dropna()
            #freq_bins_power = np.arange(0, int(data.power.max())+1000, 1000)
            
            p=p.resample('D', loffset='20h')
            w = self.weatherdata.resample('D' ,loffset='20h')
            
            self.raw = w.join(p).dropna()
            #data['binned'] = data['power'].apply(self.__map_bin, bins=freq_bins_power)
            #self.correlations = data.groupby('binned').corr(method='pearson', min_periods=1).fillna(0)
            
            #self.raw_group = self.raw.groupby(pd.TimeGrouper(group))
            #df = self.raw_group.corr(method='pearson', min_periods=1).fillna(0)
            self.correlations = self.raw.corr(method='pearson', min_periods=1)
            
            
            '''
            data = self.temperature.join(p).dropna() #The mapper won't be happy with NA values
            data['binned'] = data['power'].apply(self.__map_bin, bins=freq_bins_power)
            self.correlations_temperature = data.groupby('binned').corr(method='pearson', min_periods=1).dropna()
            
            data = self.wind.join(p).dropna() #The mapper won't be happy with NA values
            data['binned'] = data['power'].apply(self.__map_bin, bins=freq_bins_power)
            self.correlations_wind = data.groupby('binned').corr(method='pearson', min_periods=1).dropna()
            
            data = self.humidity.join(p).dropna() #The mapper won't be happy with NA values
            data['binned'] = data['power'].apply(self.__map_bin, bins=freq_bins_power)
            self.correlations_humidity = data.groupby('binned').corr(method='pearson', min_periods=1).dropna()
            '''
            
            return self.correlations
        
    def plot(self, regression=False, path=None):
        if self.correlations is None:
            self.get_correlation()
            
        #if type=='scatter':
        fig = plt.figure()      
        ax = plt.subplot(111)
        if regression:
            slope, intercept, r_value, p_value, slope_std_error = stats.linregress(self.raw['power'],self.raw['temperature'])
            predict_y = intercept + slope * self.raw['power']
            pred_error = self.raw['temperature'] - predict_y
            degrees_of_freedom = len(self.raw['power']) - 2
            residual_std_error = np.sqrt(np.sum(pred_error**2) / degrees_of_freedom)

            # Plotting
            ax.plot(self.raw['power'], predict_y, 'k-')
            ax.annotate('Residual Std. Error: '+str(residual_std_error)+'',xy=(0 , -80),xycoords=("data", "axes points"))
        
        ax.scatter(x=self.raw['power'],y=self.raw['temperature'])
        ax.annotate('Temperature ['+str(self.raw['temperature'].values.min())+';'+str(self.raw['temperature'].values.max())+']',xy=(0 , -40),xycoords=("data", "axes points"))
        '''
        else:
            fig = plt.figure()
            ax = plt.subplot(111)
            heatmap = ax.imshow(self.correlations, interpolation='nearest',  aspect='auto')
            xlabels = ['humidity','power','temperature','wind']
            if weather_parameter == 'temperature':
                heatmap = ax.imshow(self.correlations_temperature, interpolation='nearest',  aspect='auto')
                xlabels = 'temperature'
            if weather_parameter == 'wind':
                heatmap = ax.imshow(self.correlations_wind, interpolation='nearest',  aspect='auto')
                xlabels = 'wind'
            if weather_parameter == 'humidity':
                heatmap = ax.imshow(self.correlations_humidity, interpolation='nearest',  aspect='auto')
                xlabels = 'humidity'
            
            
            heatmap.set_cmap(cm.RdBu)
            
            #ax.set_yticks(np.arange(len(self.correlations.index.get_level_values('binned'))/400), minor=False)
            ax.set_xticks(np.arange(4), minor=False)
            
            #Problems here!!
            bins = self.correlations.index.get_level_values('binned')[0::4]
            ax.set_yticklabels(bins)
            
            ax.set_xticklabels(xlabels)
            
            plt.colorbar(heatmap)
            '''   

        if not path is None:
            fig = ax.get_figure()
            fig.savefig(path)
            plt.clf()
            
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
        
                
        #response = urllib2.urlopen(dataurl)
        #d = json.load(response)
        
    def __get_openweathermap_data(self, start, end):
            """
        Parameters
        ----------
        resolution : hour, day, week, month
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
            if self.dataresolution == 'minute':
                dataurl = 'http://api.openweathermap.org/data/2.5/history/station?id='+str(self.weatherstationid)+'&type=hour&start='+start+'&end='+end
            else:
                dataurl = 'http://api.openweathermap.org/data/2.5/history/station?id='+str(self.weatherstationid)+'&type='+self.dataresolution+'&start='+start+'&end='+end
            
            response = urllib2.urlopen(dataurl)
            d = json.load(response)
            
            timestamp, temperature, wind, humidity = [],[],[],[]
            
            if d['cnt'] <= 0:
                raise UnableToGetWeatherData("WeatherData from OpenWeatherMap was empty!")
            for element in d["list"]:
                dt =  element["dt"]
                try:
                    t = float(element["temp"]["v"] -272.15)
                except:
                    t = None
                try:
                    w = float(element["wind"]["speed"]["v"])
                except:
                    w = None
                try:
                    h = float(element["humidity"]["v"])
                except:
                    h = None
                        
                ts =  time.mktime(datetime.datetime.fromtimestamp(dt).timetuple())
                timestamp.append(ts)
                temperature.append(t)
                wind.append(w)
                humidity.append(h)
    
            df = pd.DataFrame(data={'temperature':temperature, 'wind':wind, 'humidity':humidity}, index=timestamp,  dtype=np.float32)
            
            df_temperature = pd.DataFrame(data={'temperature':temperature}, index=timestamp,  dtype=np.float32)
            df_wind = pd.DataFrame(data={'wind':wind}, index=timestamp,  dtype=np.float32)
            df_humidity = pd.DataFrame(data={'humidity':humidity}, index=timestamp,  dtype=np.float32)
            
            # raw data isn't always sorted
            df = df.sort_index()
            df_temperature = df_temperature.sort_index()
            df_wind = df_wind.sort_index()
            df_humidity = df_humidity.sort_index()
            # Convert the integer index column to timezone-aware datetime 
            df.index = pd.to_datetime(df.index.values, unit='s', utc=True)
            df_temperature.index = pd.to_datetime(df_temperature.index.values, unit='s', utc=True)
            df_wind.index = pd.to_datetime(df_wind.index.values, unit='s', utc=True)
            df_humidity.index = pd.to_datetime(df_humidity.index.values, unit='s', utc=True)
            
            if self.dataresolution == 'minute':
                df = df.asfreq('1Min',method='bfill')
                df_temperature = df_temperature.asfreq('1Min', method='bfill')
                df_wind = df_wind.asfreq('1Min', method='bfill')
                df_humidity = df_humidity.asfreq('1Min', method='bfill')
            
            if self.temperature is None:
                self.temperature=df_temperature
            else:
                self.temperature.append(df_temperature)
            
            if self.wind is None:
                self.wind=df_wind
            else:
                self.wind.append(df_wind)
                
            if self.humidity is None:
                self.humidity=df_humidity
            else:
                self.humidity.append(df_humidity)
                
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
            id = w.get("sys").get("id")
            self.weatherstationid = id
        except:
            raise UnableToGetWeatherData("Cannot find weather station in area of "+city+", "+country)
    
    def __roundTime(self,dt=None):
        """
        Round a timeframe object to any time laps in seconds
        dt : timeframe object, default now.
        """
        
        roundTo = 60
        if self.dataresolution == 'week':
                roundTo = 7*24*60*60
        if self.dataresolution == 'day':
                roundTo = 24*60*60
        if self.dataresolution == 'hour':
                roundTo = 60*60
        if self.dataresolution == 'minute':
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
    
    def __map_bin(self, x, bins):
        x = int(x)
        if x == max(bins):
            right = True
        else:
            right = False
        bin = bins[np.digitize([x], bins, right=right)[0]]
        bin_lower = bins[np.digitize([x], bins,  right=right)[0]-1]
        return '[{0}-{1}]'.format(bin_lower, bin)