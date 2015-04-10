'''
Created on Nov 25, 2014

@author: em
'''
from __future__ import print_function, division
from nilmtk.dataset_evaluators import Events
from nilmtk.preprocessing import Clip
from ..plots import plot_series, appliance_label
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd 
import numpy as np
from scipy import stats
import datetime 

class EventImpact(object):

    def __init__(self, metergroup=None, ):
        self.data = metergroup
        self.events = None
        self.correlations = None
        self.events_at_same_time = None 
    
    def get_correlation_6h_periods(self):
        self.correlations = [self.__calc_correlation(0, 6), self.__calc_correlation(6, 12), self.__calc_correlation(12, 18), self.__calc_correlation(18, 0)]
        return self.correlations
    
    def get_events_at_same_time(self):
        if not self.events_at_same_time:
            self.__create_events_dict(include_main=True)
            
            keys = self.events.keys()
            event_matrix=[[0 for y in range(len(keys))] for x in range(len(keys))]

            k1_num = 0
            for k1 in keys:
                k2_num  =0 
                for k2 in keys:
                    if not k1_num == k2_num:
                        #Join the to list of events, and remove NaN gives the list of events that occur at same time
                        event_count = pd.DataFrame(data={k1:self.events[k1]}, index=self.events[k1].index).join(pd.DataFrame(data={k2:self.events[k2]},index=self.events[k2].index)).dropna()
                        event_matrix[k1_num][k2_num] = event_count
                    else:
                        event_matrix[k1_num][k2_num] = -1
                    k2_num = k2_num + 1
                k1_num = k1_num + 1
        
            self.events_at_same_time = event_matrix
        
        return self.events_at_same_time
        
    def __create_events_dict(self, include_main=False):
        '''
        Create self.events, an dict with pd.Series (timestamp with event, value)
        '''
        event_dict = {}
        for m in self.data.meters:
            try:
                #Remove main meters
                if not include_main:
                    if m.upstream_meter():
                        event_dict[m.appliance_label()] = Events(meter=m).get_events_raw().dropna()#m.power_series_all_data(preprocessing=[Clip()], sections=self.data.meters[1].good_sections())#
                else:
                    event_dict[m.appliance_label()] = Events(meter=m).get_events_raw().dropna()
            except:
                print('Too few events for '+str(m.appliance_label()))
        self.events = event_dict
    
    def __calc_correlation(self, start=0, end=0):
        '''
        Get the impact of the different appliances event with each other
        '''
        #Get events
        if not self.events:
            self.__create_events_dict()
           
        #Appliances
        keys = self.events.keys()
        
        #Create empty matrix for correlation matrice
        event_matrix=[[0 for y in range(len(keys))] for x in range(len(keys))]
        events_split = {}   
        for k in keys:
            #Take events in time periode
            events_split[k] = self.events[k].ix[self.events[k].index.indexer_between_time(datetime.time(start), datetime.time(end))]
            
        k1_num = 0
        for k1 in keys:
            k2_num  =0 
            e1 = events_split[k1].sort_index()
            for k2 in keys:
                e2 = events_split[k2].sort_index()
                corr = e1.corr(other=e2, method='pearson', min_periods=3)
                #Using pandas.Series.corr it is not always 1 when k1==k2
                if np.array_equal(e1, e2) and k1==k2:
                    corr = 1
                event_matrix[k1_num][k2_num] = corr
                k2_num = k2_num + 1
            k1_num = k1_num + 1
        
        return event_matrix
        
    def plot(self, type='correlation', path=None):
        """
        Parameters
        ----------
        path : save figure to path
        Returns
        ----------
        ax : matplotlib.axes, optional
        """
        
        if type == 'correlation':
            ax = self.__plot_correlation()
        if type == 'time':
            ax = self.__plot_events_time()
        
        if not path is None:
            fig = ax.get_figure()
            fig.savefig(path, bbox_inches='tight')
            plt.clf()
            
        return ax

    def __plot_events_time(self):
        '''
        Create plot for events at the same time
        '''
        if self.events_at_same_time is None:
            self.get_events_at_same_time()
            
        fig = plt.figure()      
        ax = plt.subplot(111)
        #Take main meter
        power_series = self.data.meters[1].power_series_all_data(preprocessing=[Clip()], sections=self.data.meters[1].good_sections())
        ax = plot_series(power_series, ax=ax, label='Main', unit='Power')
        
        label=[]
        keys = self.events.keys()
        colors = ['g', 'r', 'y', 'c','m','k']
        if len(keys) > 6:
            for i in range(0,(len(keys)-6)):
                colors.append('g')
        
        k2_num = 0 
        for k2 in keys:
            #Avoid plotting events from the same meter and main meter
            if len(self.events_at_same_time[0])  >= 2:
                if k2_num  > 1 :
                    if not self.events_at_same_time[1][k2_num][k2].empty:
                        ax.scatter(x=self.events_at_same_time[1][k2_num][k2].index, y=self.events_at_same_time[1][k2_num][k2],color=colors[k2_num], zorder=3, label=keys[k2_num]+' '+str(len(self.events_at_same_time[1][k2_num][k2].values))+' events')
                        keys = self.events.keys()
            k2_num = k2_num + 1
             
        plt.suptitle('Events for two appliance types at same time')
        plt.legend(loc=4)
        
        return ax
        
    def __plot_correlation(self):
        '''
        Create plot for correlation, with 4 periods
        '''
        if self.correlations is None:
            self.get_correlation_6h_periods()
            
        fig, axes = plt.subplots(nrows=2, ncols=2)
        count = 0
        
        for ax in axes.flat:
            im = ax.imshow(self.correlations[count],vmin=-1.0,vmax=1.0, interpolation='nearest',cmap=cm.get_cmap('RdBu'))#).RdBu )
            ax.set_yticks(np.arange(len(self.events.keys())), minor=False)
            ax.set_xticks(np.arange(len(self.events.keys())), minor=False)
            if count==0:
                ax.set_title('Period: 0-6')
            if count==1:
                ax.set_title('Period: 6-12')
            if count==2:
                ax.set_title('Period: 12-18')
            if count==3:
                ax.set_title('Period: 18-24')
            
            count = count + 1
        
        label = [] 
        for k in self.events.keys():
            #if k == '':
            #    k='main'
            label.append(appliance_label(label=k,remove_all=True))
        
            
        for i, row in enumerate(axes):
            for j, cell in enumerate(row):
                if i == len(axes) - 1:
                    cell.set_xticklabels(label, rotation=90)
                else:
                    cell.set_xticklabels([])
                if j == 0:
                    cell.set_yticklabels(label)
                else:
                    cell.set_yticklabels([])
        

        plt.suptitle('Correlation between appliances minute pr minute in periods')
        cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
        cbar = plt.colorbar(im, cax=cax, **kw)
        return ax