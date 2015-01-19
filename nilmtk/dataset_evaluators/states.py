from __future__ import print_function, division
from nilmtk.preprocessing import Clip
from sklearn.cluster import KMeans
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
from nilmtk.plots import appliance_label

class States(object):

    def __init__(self, meter=None, use_kmean=False):
        self.use_kmean = use_kmean
        self.meter = meter
        self.states = None
        self.clusterdata = pd.DataFrame()
        self.clusters = None
        self.MIN_OBS = 100
    def get_states(self):
        """
        Parameters
        ----------
        
        Returns
        -------
        states : int
       
        Raises
        ------
        """
        MAX_DATA = 4000
        if self.clusterdata.empty:
            data = self.__get_data()
            #"D": data.diff().abs().values
            df = pd.DataFrame(data={"power":data.values}, dtype=np.float32).dropna()
        
            c = df.describe().count
            if c > MAX_DATA:
                df = df[0:MAX_DATA]
       
            self.clusterdata = df
        # clustering
        if self.use_kmean:
            return self.__kmean_cluster()
        else:
            return self.__hcluster()
        
    def __hcluster(self):
        # clustering
        thresh = self.clusterdata["power"].max()/6
        if not self.states:
            Z = hcluster.linkage( self.clusterdata, "median" )  # N-1 "average" "centroid" "single"median complete
            self.clusters = hcluster.fcluster( Z, thresh , criterion="distance" )
            self.clusterdata =  self.clusterdata.join(pd.DataFrame(data={'cluster':self.clusters}))
            
            return self.__cluster_postprocess()
        else:
            return self.states
    
    def __kmean_cluster(self):
        if not self.states:
            k_means = KMeans(init='k-means++', n_clusters=3)
            self.clusters = k_means.fit(self.clusterdata).labels_
            self.clusterdata =  self.clusterdata.join(pd.DataFrame(data={'cluster':self.clusters}))
            return self.__cluster_postprocess()
        else:
            return self.states
    
    def __cluster_postprocess(self):
        cluster_len = len(set(self.clusters))
        for s in range(1, cluster_len+1):
            if (self.clusters == s).sum() < self.MIN_OBS:
                self.clusters = np.delete(self.clusters, np.any( self.clusters != s))
                self.clusterdata = self.clusterdata[self.clusterdata.cluster != s].dropna()
            
        cluster_color = pd.DataFrame(data={'cluster_color':self.clusters}).replace([0,1,2,3,4,5,6],['b','g','r','c','m','y','k'])
        self.clusterdata = self.clusterdata.join(cluster_color).dropna()
        self.states = len(set(self.clusters))
        
        return self.states
    
    def __get_data(self):
        '''
        Returns
        -------
        data : pd.Series timestamp, consumption
        '''
        data = self.meter.power_series_all_data(preprocessing=[Clip()])
        
        return data
      
    def plot(self,type=None, path=None):
        """
        Parameters
        type : if set to 'sort' it will arrange the values after the largest first
        path : save figure to path
        ----------
        ax : matplotlib.axes, optional
        """     
        self.get_states()  
        fig = plt.figure()
        ax = plt.subplot(111)
        
        
        if type == 'sort':
            self.clusterdata=self.clusterdata.sort('power', ascending=False)
            self.clusterdata.index = range(0,len(self.clusterdata))
        
        i = self.clusterdata.index
        w= 0.0001
        
        for index in i:
            ax.bar(left=index*w, height=self.clusterdata['power'][index], width=w, color=self.clusterdata['cluster_color'][index], edgecolor = "none")
            
        ax.set_xlim([0, (max(i)*w)])
        
        ax.set_xticklabels(ax.get_xticks()*10000)
        
        ax.set_ylabel('Power (watt)')  
        ax.set_xlabel('Data Points') 
        
        title = appliance_label(label=self.meter.appliance_label(),remove_all=True)
        ax.set_title(title)
        
        if not path is None:
            fig = ax.get_figure()
            fig.savefig(path, bbox_inches='tight')
            plt.clf()
        return ax