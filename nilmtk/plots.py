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
            regression_xy, r_squared_xy = __regression(x,y)
            green = ax.plot(x, regression_xy, z.mean(), color='g', label=xlabel+' vs '+ylabel+', R2: '+str(r_squared_xy))
            regression_xz, r_squared_xz = __regression(x,z)
            y_temp = pd.DataFrame(data={'y':y.mean()}, index=x.index)
            red = ax.plot(x, y_temp['y'], regression_xz, color='r', label=xlabel+' vs '+zlabel+', R2: '+str(r_squared_xz))
            regression_yz, r_squared_yz = __regression(y,z)
            x_temp = pd.DataFrame(data={'x':x.mean()}, index=y.index)
            yellow = ax.plot(x_temp, y, regression_yz, color='y', label=ylabel+' vs '+zlabel+', R2: '+str(r_squared_yz))
            
            ax.legend(handles=[green[0], red[0], yellow[0]], loc=4)
            
        # Plotting
        ax.elev = -45
        ax.scatter(xs=x, ys=y, zs=z)
        ax.set_xlim([x.mean()/3.0, max(x)+100.0])
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        
        return ax
    
def __regression(x, y):
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
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'text.fontsize': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
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
