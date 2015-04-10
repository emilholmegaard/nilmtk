from __future__ import print_function, division
from nilmtk.preprocessing import Clip
from sklearn.cluster import KMeans
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from nilmtk.plots import appliance_label

class Evaluate_Meter_VSD(object):

    def __init__(self, meter=None, ratio=0.7):
        self.meter = meter
        self.clusterdata = pd.DataFrame()
        self.clusters = None
        self.MIN_OBS = 250
        #ratio of zero slopes to indicate vsd
        self.slope_ratio = ratio
        self.possible_vsd = None
        
    
    def is_vsd(self):
        """
        Parameters
        ----------
        
        Returns
        -------
        isVSD : bool
       
        Raises
        ------
        """
        
        if not self.possible_vsd:
            self.__kmean_cluster()
            zero_slope_score = self.__zero_slope(chunksize_factor=0.01)
            self.possible_vsd = (zero_slope_score > self.slope_ratio)
            
        return self.possible_vsd
        
    
    def __zero_slope(self, chunksize_factor, max_slope = .001):
        """return the ratio of zero slopes

        chunksize_factor --> 0->1
        returns  zero_slopes / total_chunks
        """
        if chunksize_factor >= 1:
            chunksize_factor = 0.01
        chunksize = len(self.clusterdata)*chunksize_factor
        total_chunks = len(self.clusterdata) % chunksize
        midindex = chunksize / 2
        zero_slopes = []
        for index in xrange(len(self.clusterdata) - chunksize):
            chunk = self.clusterdata[index : index + chunksize, :]
            # subtract the endpoints of the chunk
            # if not sufficient, maybe use a linear fit
            dx, dy = abs(chunk[0] - chunk[-1])
            #print dy, dx, dy / dx
            if 0 <= dy / dx < max_slope:
                zero_slopes.append(chunk[midindex])    
        
        return len(zero_slopes)/total_chunks
        
    def __kmean_cluster(self):
        if not self.possible_vsd:
            k_means = KMeans(init='k-means++', n_clusters=3)
            self.clusters = k_means.fit(self.clusterdata).labels_
            self.clusterdata =  self.clusterdata.join(pd.DataFrame(data={'cluster':self.clusters}))
            return self.__cluster_postprocess()
        else:
            return self.possible_vsd
    
    def __cluster_postprocess(self):
        cluster_len = len(set(self.clusters))
        for s in range(1, cluster_len+1):
            if (self.clusters == s).sum() < self.MIN_OBS:
                self.clusters = np.delete(self.clusters, np.any( self.clusters != s))
                self.clusterdata = self.clusterdata[self.clusterdata.cluster != s].dropna()
            
        cluster_color = pd.DataFrame(data={'cluster_color':self.clusters}).replace([0,1,2,3,4,5,6],['b','g','r','c','m','y','k'])
        self.clusterdata = self.clusterdata.join(cluster_color).dropna()
        #Get data ready for finding plateaus
        self.clusterdata=self.clusterdata.sort('power', ascending=False)
        self.clusterdata.index = range(0,len(self.clusterdata))
        self.possible_vsd = self.is_vsd()
        
        return self.possible_vsd
    
    def __get_data(self):
        '''
        Returns
        -------
        data : pd.Series timestamp, consumption
        '''
        data = self.meter.power_series_all_data(preprocessing=[Clip()])
        
        return data
      
    def plot(self, path=None):
        """
        Parameters
        path : save figure to path
        ----------
        ax : matplotlib.axes, optional
        """     
        is_vsd = self.is_vsd()  
        fig = plt.figure()
        ax = plt.subplot(111)
        
        
        i = self.clusterdata.index
        w= 0.0001
        
        for index in i:
            ax.bar(left=index*w, height=self.clusterdata['power'][index], width=w, color=self.clusterdata['cluster_color'][index], edgecolor = "none")
            
        ax.set_xlim([0, (max(i)*w)])
        
        ax.set_xticklabels(ax.get_xticks()*10000)
        
        ax.set_ylabel('Power (watt)')  
        ax.set_xlabel('Data Points') 
        
        if is_vsd:
            title = appliance_label(label=self.meter.appliance_label(),remove_all=True) + ' is VSD candidate'
        else:
            title = appliance_label(label=self.meter.appliance_label(),remove_all=True)
        
        ax.set_title(title)
        
        if not path is None:
            fig = ax.get_figure()
            fig.savefig(path, bbox_inches='tight')
            plt.clf()
        return ax