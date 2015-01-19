from __future__ import print_function, division
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from math import sqrt


SPINE_COLOR = 'gray'

_to_ordinalf_np_vectorized = np.vectorize(mdates._to_ordinalf)

def plot_series(series, **kwargs):
    """Plot function for series which is about 5 times faster than
    pd.Series.plot().

    Parameters
    ----------
    series : pd.Series
    ax : matplotlib Axes, optional
        If not provided then will generate our own axes.
    fig : matplotlib Figure
    date_format : str, optional, default='%d/%m/%y %H:%M:%S'
    tz_localize : boolean, optional, default is True
        if False then display UTC times.

    Can also use all **kwargs expected by `ax.plot`
    """
    ax = kwargs.pop('ax', None)
    fig = kwargs.pop('fig', None)
    date_format = kwargs.pop('date_format', '%d/%m/%y %H:%M:%S')
    tz_localize = kwargs.pop('tz_localize', True)
    unit = kwargs.pop('unit', 'watt')
    
    if ax is None:
        ax = plt.gca()

    if fig is None:
        fig = plt.gcf()

    x = _to_ordinalf_np_vectorized(series.index.to_pydatetime())
    ax.plot(x, series, **kwargs)
    tz = series.index.tzinfo if tz_localize else None
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format, 
                                                      tz=tz))
    ax.set_ylabel(unit)
    fig.autofmt_xdate()
    return ax

def plot_3d(x,y,z, xlabel='x', ylabel='y', zlabel='z', regression=False, **kwargs):
        """Plot function for plotting a 3d-plot, for dataset evaluators.

        Parameters
        ----------
        x : pd.Series
        y : pd.Series
        z : pd.Series
        xlabel : label for x
        ylabel : label for y
        zlabel : label for z 
        ax : matplotlib Axes, optional
        If not provided then will generate our own axes.
        fig : matplotlib Figure
        
        Can also use all **kwargs expected by `ax.plot`
        """
        ax = kwargs.pop('ax', None)
        fig = kwargs.pop('fig', None)
        
        
        if ax is None:
            ax = plt.subplot(111,  projection='3d')

        if fig is None:
            fig = plt.gcf()
       
        if regression:
            regression_xy, r_squared_xy = calc_regression(x,y)
            green = ax.plot(x, regression_xy, z.mean(), color='g', label=xlabel+' vs '+ylabel+', R2: '+str('%.2f' % r_squared_xy))
            regression_xz, r_squared_xz = calc_regression(x,z)
            y_temp = pd.DataFrame(data={'y':y.mean()}, index=x.index)
            red = ax.plot(x, y_temp['y'], regression_xz, color='r', label=xlabel+' vs '+zlabel+', R2: '+str('%.2f' % r_squared_xz))
            regression_yz, r_squared_yz = calc_regression(y,z)
            x_temp = pd.DataFrame(data={'x':x.mean()}, index=y.index)
            yellow = ax.plot(x_temp, y, regression_yz, color='y', label=ylabel+' vs '+zlabel+', R2: '+str('%.2f' % r_squared_yz))
            
            ax.legend(handles=[green[0], red[0], yellow[0]], loc=4)
            
        # Plotting
        ax.elev = -45
        ax.scatter(xs=x, ys=y, zs=z)
        ax.set_xlim([x.mean()/3.0, max(x)+100.0])
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        
        return ax
    
def calc_regression(x, y):
        '''
        Calculate the linear regression between x and y
        '''
        slope, intercept, r_value, p_value, slope_std_error  = stats.linregress(x,y)
        predict_y = intercept + slope * x
        r_squared = r_value**2
        #pred_error = y - predict_y
        #degrees_of_freedom = len(x) - 2
        #residual_std_error = np.sqrt(np.sum(pred_error**2) / degrees_of_freedom)
        return predict_y, r_squared

def plot_simple_metric(values, labels, **kwargs):
    """Plots the given values array

    Parameters
    ----------
    values : list of floats to plot 
    labels : list of labels that correspond to the values
    """
    ax = kwargs.pop('ax', None)
    fig = kwargs.pop('fig', None)
        
    if ax is None:
        ax = plt.subplot(111)

    if fig is None:
        fig = plt.gcf()
    
    labels = [appliance_label(label=l,remove_all=True) for l in labels]
    
    ax.plot(values, marker='v')
    ax.set_xlim(xmin=-0.5,xmax=len(labels)+0.5)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    
    return ax
    
        

def plot_metric(series, appliance=None, xlabels=None,ymin=None, ymax=None,marker=1,label='', **kwargs):
    """Plots the given metric - for a list of predictions

    Parameters
    ----------
    predictions, ground_truth : list of list of pd.Series representing F-metric
    labels : list of labels that correspond to the predictions/ground truth
    metric : the given metric to plot results
    """
    #dict for appliance_label : [f-score] 
    appliance_value = __appliance_value_map(series, appliance)    
    
    ax = kwargs.pop('ax', None)
    fig = kwargs.pop('fig', None)
        
    if ax is None:
        ax = plt.subplot(111)

    if fig is None:
        fig = plt.gcf()
    
    markers = ['o','v','^','<','>','1','2','3','4','8','s','p','*','h']
    i=0
    xlabel=[]
    
    
    for k in appliance_value.keys():
        ax.plot(appliance_value[k], marker=markers[i], label=appliance_label(k))
        i=i+1
       
    if ymin is None:
        ymin = min([min(m) for m in appliance_value.values()])-0.5
    if ymax is None:
        ymax = max([max(m) for m in appliance_value.values()])+0.5
        
        
    ax.set_ylim(ymin=ymin,ymax=ymax)
    
                
    if not xlabels is None:
        ax.set_xlim(xmin=-0.5,xmax=(len(xlabels)+0.5))
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels)
    else:
        ax.set_xlim(xmin=-0.5,xmax=(i+0.5))
        ax.set_xticks(np.arange(len(xlabel))+1)
        ax.set_xticklabels(xlabel,rotation='vertical')
    
    ax.legend(loc=4)
    
    return ax

def __plot_metric(series, appliance=None, xlabels=None,ymin=None, ymax=None,marker=1,label='', **kwargs):
    """Plots the given metric - for a list of predictions

    Parameters
    ----------
    predictions, ground_truth : list of list of pd.Series representing F-metric
    labels : list of labels that correspond to the predictions/ground truth
    metric : the given metric to plot results
    """
    #dict for appliance_label : [f-score] 
    appliance_value = __appliance_value_map(series, appliance)    
    
    ax = kwargs.pop('ax', None)
    fig = kwargs.pop('fig', None)
        
    if ax is None:
        ax = plt.subplot(111)

    if fig is None:
        fig = plt.gcf()
    
    markers = ['o','v','^','<','>','1','2','3','4','8','s','p','*','h']
    colors = ['b','g','r','y','c']
    i=1
    count = []
    values=[]
    xlabel=[]
    
    for k in appliance_value.keys():
        count.append(i)
        values.append(appliance_value[k][0])
        xlabel.append(appliance_label(k))
        i=i+1
    
    #ax.plot(count,values, marker=markers[marker], label=label)
    ax.scatter(count,values, marker=markers[marker], label=label, color=colors[marker])
    '''
    for k in appliance_value.keys():
        if i==1:
            ax.plot(y=appliance_value[k], marker=markers[i], label=__appliance_label(k))
        i=i+1
   '''
        
    if ymin is None:
        ymin = min([min(m) for m in appliance_value.values()])-0.5
    if ymax is None:
        ymax = max([max(m) for m in appliance_value.values()])+0.5
        
    
    ax.set_ylim(ymin=ymin,ymax=ymax)
    ax.set_xlim(xmin=0.5,xmax=i+0.5)
                
    if not xlabels is None:
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels,rotation='vertical')
    else:
        ax.set_xticks(np.arange(len(xlabel))+1)
        ax.set_xticklabels(xlabel,rotation='vertical')
    
    ax.legend(loc=1)#loc=4
    
    return ax


def plot_all_metric(df, appliance=None, xlabels=None,ymin=None, ymax=None, **kwargs):
    """Plots the given metric - for a list of predictions

    Parameters
    ----------
    predictions, ground_truth : list of list of pd.Series representing F-metric
    labels : list of labels that correspond to the predictions/ground truth
    metric : the given metric to plot results
    """
    
    ax = __plot_metric(df['F1'], appliance, label='F1', ymin=ymin, ymax=ymax)
    ax = __plot_metric(df['FTE'], appliance,marker=2,label='NEAE',ymin=ymin, ymax=ymax, ax=ax)
    ax = __plot_metric(df['MNE'], appliance,marker=3, label='MNE',ymin=ymin, ymax=ymax, ax=ax)
        
    return ax


def __appliance_value_map(series, appliance):
    #dict for appliance_label : [values] 
    appliance_value={}
    i=0
    for s in series:
        for m in s:
            j=0
            for index, value in m.iteritems():
                if not appliance_value.has_key(appliance[i][j]):
                    appliance_value[appliance[i][j]] = []
                appliance_value[appliance[i][j]].append(value)
                j=j+1
        i = i+1
    return appliance_value

def appliance_label(label, remove_all=False):
    try:
        if remove_all:
            label = label.split(',')[0]
        else:
            label = label.split(',')[0]+' '+label.split(',')[2]
        
        remove="()'"
        for i in range(0,len(remove)):
            label =label.replace(remove[i],"")
        return label.title()
    except:
        return label   
         
def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\usepackage{gensymb}'],
              'axes.labelsize': 12, # fontsize for x and y labels (was 8) 10
              'axes.titlesize': 12,
              'text.fontsize': 10, # was 8
              'legend.fontsize': 10, # was 8
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif',
              'font.weight': 'bold'
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

#    matplotlib.pyplot.tight_layout()

    return ax
