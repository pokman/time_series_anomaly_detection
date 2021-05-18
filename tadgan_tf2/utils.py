# Auxiliary functions for (1) visualization and (2) evaluation.
#
#

import numpy as np
import matplotlib.pyplot as plt


def plot_time_series(t, x, x_win=None, labeled_intervals=None, detected_intervals=None,
                     date_range=None, value_range=None, title=None):
    
    """Plot a time series, together with any of the following details:
    - a collection of related time series over rolling windows
    - labeled anomalous intervals
    - detected anomalous intervals
    
    Paramters
    ---------
    t, x : 1D arrays
        Time stamps and values of a time series.
        
    x_win : 2D array or None
        A collection of time series over rolling windows.
        
    labeled_intervals, detected_intervals : 2D arrays or None
        Labeled and detected intervals, each of the shape (# intervals, 2).

    date_range : list of datetimes, or None 
        Left and right limits of the time axis. If None, the start and end of t
        are used.
    
    value_range : list or None
        Lower and upper limits of the vertical axis.
    
    title : str
        Title of the plot.
    """

    if date_range is not None:
        date_range = [np.datetime64(d) for d in date_range]
    else:
        date_range = [t[0], t[-1]]
    
    plt.figure(figsize=(15, 5))
    mask = (t >= date_range[0]) & (t <= date_range[1])    
    plt.plot(t[mask], x[mask], color='blue')

    if x_win is not None:
        n_windows, window_size = x_win.shape
        for i in range(n_windows):
            if (t[i] <= date_range[1]) & (t[i + window_size - 1] >= date_range[0]):
                plt.plot(t[i : i + window_size], x_win[i], color='red', alpha=0.1, lw=1.0)

    if labeled_intervals is not None:
        for start, end in labeled_intervals:
            if (start <= date_range[1]) & (end >= date_range[0]):
                plt.axvspan(start, end, color='blue', alpha=0.2)

    if detected_intervals is not None:
        for start, end in detected_intervals:
            if (start <= date_range[1]) & (end >= date_range[0]):
                plt.axvspan(start, end, color='red', alpha=0.2)

    plt.xlim(date_range)
    if value_range:
        plt.ylim(value_range)
    if title:
        plt.title(title)
    
    plt.show()


def evaluate_detected_anomalies(labeled_intervals, detected_intervals):

    """Calculate recall, precision and F1-score of the detected anomalous
    intervals with reference to the labeled ones.    
    """
    
    unit = np.timedelta64(1, 'D')
    len_labeled, len_detected, len_overlap = 0, 0, 0

    for start, end in labeled_intervals:
        len_labeled += (end - start) / unit

    for start, end in detected_intervals:
        len_detected += (end - start) / unit
        
    for start1, end1 in labeled_intervals:
        for start2, end2 in detected_intervals:
            start, end = max(start1, start2), min(end1, end2)
            if start < end:
                len_overlap += (end - start) / unit
                
    recall = len_overlap / len_labeled
    precision = len_overlap / len_detected
    f1 = 2 * recall * precision / (recall + precision + 1e-8)
    
    return recall, precision, f1
