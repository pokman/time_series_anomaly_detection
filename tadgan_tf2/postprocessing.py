# Post-processing steps in the use of the time series reconstruction model
# (TadGAN) for anomaly detection. These include modified and re-organized
# versions of similar functions in the Orion tutorial: 
# https://github.com/signals-dev/Orion/blob/master/notebooks/tulog/Tulog.ipynb
#
#

import numpy as np
from scipy import stats
import pandas as pd
from tadgan_tf2 import preprocessing


def merge_rolling_windows(x_win, step_size=1):
    '''Merge rolling-window time series data into single time series by taking
    median values per time stamp.
    
    The rolling window data (x_win) is expected to be an nD array, with n >= 2
    and of the shape (# windows, window size, ...).
    
    The output is the combined time series (x) in the form of an (n-1)D array
    of the shape (# time stamps, ...).
    '''
    n_windows, window_size = x_win.shape[:2]
    n = (n_windows - 1) * step_size + window_size
    x_mult = np.full((n, window_size, *(x_win.shape[2:])), np.nan)
    for i in range(window_size):
        x_mult[i::step_size, i][:n_windows] = x_win[:, i]
    x = np.nanmedian(x_mult, axis=1)
    return x


def _window_with_pad(a, window, pad_value, win_first=False):
    '''Helper function. Split a sequence into rolling windows after some
    padding, so that the number of windows equals the length of the sequence.
    '''
    pad_sizes = [(0, 0)] * len(a.shape)
    pad_sizes[0] = (window // 2, window - window // 2 - 1)
    a = np.pad(a, pad_sizes, 'constant', constant_values=pad_value)
    dummy_id = np.zeros(a.shape[0])
    a, _ = preprocessing.rolling_windows(a, dummy_id, window)
    if win_first:
        a = np.transpose(a, (1, 0, *range(2, len(a.shape))))
    return a


def _batch_dtw(a1, a2):
    '''Calculate the dynamic time warping distances between pairs of time 
    series in batch.
    
    The batches of time series (a1, a2) are expected to be nD arrays, with
    their first dimensions representing time and other dimensions identical.
    
    The output is the resulting batch of distances, as an (n-1)D array.
    '''
    assert a1.shape[1:] == a2.shape[1:]
    dtw = np.full((a1.shape[0]+1, a2.shape[0]+1, *(a1.shape[1:])), np.inf)
    dtw[0, 0] = 0
    for i in range(a1.shape[0]):
        for j in range(a2.shape[0]):
            dtw[i+1, j+1] = np.absolute(a1[i] - a2[j]) \
                + np.minimum(np.minimum(dtw[i, j+1], dtw[i+1, j]), dtw[i, j])
    return dtw[-1, -1]


def reconstruction_error_scores(x, x_recon, dtw_window=10, smooth_window=None, normalize=True):
    
    '''Evaluate the reconstruction of time series based on the rolling DTW
    (dynamic time warping) differences between the original and reconstructed 
    series.
    
    Parameters
    ----------
    x, x_recon : nD arrays
        Original and reconstructed time series data. The arrays are expected to
        have the same shape, with the first dimension representing time.
    
    dtw_window : int
        Size of the rolling windows for calculation of DTW distances.

    smooth_window : int or None
        Size of the rolling windows for calculation of running averages. If
        None, no such smoothing is applied.
    
    normalize : bool
        Whether to normalize the scores. 

    Returns
    -------
    scores : nD array
        Scores of reconstruction errors. The array has the same shape as the
        given time series data.
    '''
    
    assert x.shape == x_recon.shape

    # calculate rolling DTW distances between original and reconstructed series
    x_win = _window_with_pad(x, dtw_window, 0, win_first=True)
    x_recon_win = _window_with_pad(x_recon, dtw_window, 0, win_first=True)
    scores = _batch_dtw(x_win, x_recon_win)
    del x_win, x_recon_win
    
    # smooth score series by taking rolling averages
    if smooth_window:
        scores = np.nanmean(_window_with_pad(scores, smooth_window, np.nan), axis=1)
    
    # normalize scores (and take positive part only)
    if normalize:
        mean, sd = np.mean(scores, axis=0), np.std(scores, axis=0)
        scores = np.maximum(scores - mean, 0) / (sd + 1e-8)

    return scores


def _kde_mode(a):
    '''Find the element of a sequence with the highest Gaussian KDE.'''
    a = a[~np.isnan(a)]
    if len(a) == 1 or a.std() == 0.0:
        return a.mean()
    kde = stats.gaussian_kde(a)(a)
    mode = a[np.argmax(kde)]
    return mode


def unroll_critic_scores(scores_win, window_size, step_size=1, smooth_window=None, normalize=True):

    '''Transform critic scores associated to rolling windows (as produced by
    the reconstruction model) into values associated to individual time stamps.
    
    Parameters
    ----------
    scores_win : nD array
        Critic scores associated to rolling windows, of the shape (# windows,
        ...).                                                                       
    
    window_size : int
        Size of the underlying rolling windows.
    
    step_size : int
        Number of time stamps between successive windows.
    
    smooth_window : int or None
        Size of the rolling windows for calculation of running averages. If
        None, no such smoothing is applied.
    
    normalize : bool
        Whether to normalize the scores. 
    
    Returns
    -------
    scores : nD array
        Transformed critic scores, of the shape (# time stamps, ...).
    '''
    
    # take the majority value (in the sense of maximal Gaussian KDE) among the
    # scores associated to the rolling windows covering each time stamp
    n_windows = scores_win.shape[0]
    n = (n_windows - 1) * step_size + window_size
    scores_mult = np.full((n, window_size, *scores_win.shape[1:]), np.nan)
    for i in range(window_size):
        scores_mult[i::step_size, i][:n_windows] = scores_win
    scores = np.apply_along_axis(_kde_mode, 1, scores_mult)
    del scores_mult
    
    # smooth score series by taking rolling averages
    if smooth_window:
        scores = np.nanmean(_window_with_pad(scores, smooth_window, np.nan), axis=1)

    # normalize scores and take absolute values
    # Note: Unlike the original version, the mean is taken over all values
    # instead of only the values in the middle two quartiles.
    if normalize:
        mean, sd = np.mean(scores, axis=0), np.std(scores, axis=0)
        scores = np.absolute(scores - mean) / (sd + 1e-8)
    
    return scores


def _consec_seqs(bools):
    '''Locate the consecutive sequences of True in a 1D array of Booleans.'''
    starts = np.argwhere(bools & np.insert(~bools[:-1], 0, True)).flatten()
    ends = np.argwhere(bools & np.append(~bools[1:], True)).flatten() + 1
    return np.stack([starts, ends], axis=1)


def _find_local_anomalies(scores, threshold, pad=None):
    '''Locate consecutive sequences of scores exceeding a threshold, with the
    option of including positions within a given distance (pad).
    
    The scores are expected to be a 1D array.
    
    The outputs consist of a pandas data frame containing the endpoints of, and
    maximum score within, each sequence; as well as the maximum score outside
    all sequences.
    '''
    # locate sequences of scores exceeding the threshold
    anomalous = (scores > threshold)
    seqs = _consec_seqs(anomalous)

    # include positions within a given distance
    if pad and (pad > 0):
        for start, end in seqs:
            anomalous[max(start - pad, 0) : start] = True
            anomalous[end : min(end + pad, len(scores))] = True
        seqs = _consec_seqs(anomalous)

    # attach max scores within each sequence / outside all
    seqs = pd.DataFrame(seqs, columns=['start', 'end'])
    seqs['max_score'] = seqs.apply(lambda x: scores[x['start'] : x['end']].max(), axis=1)
    max_score_rest = 0 if anomalous.all() else scores[~anomalous].max()

    return seqs, max_score_rest


def identify_anomalous_sequences(scores, t, sd_threshold=4.0, pad=None,
                                 local_window=None, local_window_step=None, prune_param=None):
    
    '''Identify anomalous sequences in a series of scores. A score is flagged
    as anomalous if it exceeds the mean score by a specified number of standard
    deviations; and this is carried out in the context of the entire series 
    (global anomalies) as well as specified segments of it (local anomalies). 
    Any overlapping sequences thus obtained are subsequently merged.

    Parameters
    ----------
    scores : 1D array
        Scores used to identify anomalous sequences.
    
    t : 1D array
        Underlying time stamps associated to the scores.
        
    sd_threshold : float
        Threshold for a score to be considered anomalous, as measured in number
        of standard deviations from the mean.
        
    pad : int or None
        Number of extra time stamps to be included at both ends of each
        anomalous sequence. If None, no extra time stamps are included.
    
    local_window : int or None
        Size of rolling windows used in the search of local anomalies. If None,
        it is set to be 33% of the length of the series.
    
    local_window_step : int or None
        Number of steps between successive rolling windows used in the search 
        of local anomalies. If None, it is set to be 10% of the length of the
        series.
        
    prune_param : float or None
        Parameter of a pruning procedure for the locally identified sequences.
        If None, no pruning is applied.
    
    Returns
    -------
    seqs : pandas DataFrame
        Identified anomalous sequences as indicated by the first and last time
        stamps (start, end), together with associated scores (score).
    '''
    
    n = len(scores)
    local_window = local_window or int(n * 0.33)
    local_window_step = local_window_step or int(n * 0.1)
    all_local_seqs = []
    
    for window, step in [(n, 1), (local_window, local_window_step)]:

        n_windows = (n - window - 1) // step + 2
        for i in range(n_windows):
            
            window_start = i * step
            local_scores = scores[window_start : window_start + window]
            mean, sd = local_scores.mean(), local_scores.std()
            
            # identify anomalies based on a threshold
            threshold = mean + sd * sd_threshold  
            if local_scores.max() <= threshold:
                continue
            local_seqs, max_score_rest = _find_local_anomalies(local_scores, threshold, pad=pad)
            
            # prune anomalous sequences based on the max scores therein
            if prune_param:  # Orion: 0.1
                local_seqs = local_seqs.sort_values('max_score', ascending=False, ignore_index=True)
                max_scores = local_seqs['max_score'].to_numpy()
                max_score_drops = 1 - np.append(max_scores[1:], max_score_rest) / max_scores
                if max_score_drops.max() <= prune_param:
                    continue
                cutoff = np.argwhere(max_score_drops > prune_param).max() + 1
                local_seqs = local_seqs.iloc[:cutoff]
            
            # replace local indices by global ones
            local_seqs[['start', 'end']] += window_start
            
            # normalize associated scores
            local_seqs['score'] = (local_seqs['max_score'] - mean) / (sd + 1e-8) 
            local_seqs = local_seqs.drop('max_score', axis=1)
            
            all_local_seqs.append(local_seqs)
    
    if len(all_local_seqs) == 0:
        return pd.DataFrame(columns=['start', 'end', 'score'])

    # merge overlapping sequences and associated scores
    all_local_seqs = pd.concat(all_local_seqs)
    score_densities = np.zeros(n)
    for _, x in all_local_seqs.iterrows():
        score_densities[int(x['start']) : int(x['end'])] += x['score']
    seqs = _consec_seqs(score_densities > 0)
    seqs = pd.DataFrame(seqs, columns=['start', 'end'])
    seqs['score'] = seqs.apply(lambda x: score_densities[x['start'] : x['end']].mean(), axis=1)

    # replace indices by underlying time stamps
    seqs['start'] = seqs['start'].apply(lambda i: t[i])
    seqs['end'] = seqs['end'].apply(lambda i: t[i-1])
    
    return seqs


def apply_pipeline(x, t, x_win_recon, critic_scores, **params):
    
    '''Apply a pipeline of postprocessing steps taking outputs of the
    reconstruction model and producing the identified anomalous sequences.
    
    Parameters
    ----------
    x : 1D-array
        Original time series.
    
    t : 1D-array
        Time stamps of x.
    
    x_win_recon : 2D-array
        Reconstructed rolling windows of x.
    
    critic_scores : 1D-array
        Raw critic scores from the reconstruction model (TadGAN).
    
    params : dict
        Parameters of the postprocessing steps. The possible key-value pairs
        include:
        - 'recon_score_params': kwargs of reconstruction_error_scores()
        - 'critic_score_params': kwargs of unroll_critic_scores()
        - 'anomalous_seq_params': kwargs of identify_anomalous_sequences()
    
    Returns
    -------
    seqs : pandas DataFrame
        Anomalous sequences detected, including their starting and ending time
        stamps and associated scores.

    x_recon : 1D-array
        Reconstructed time series.
    
    combined_scores : 1D-array
        Scores used in the identification of anomalous sequences, based on 
        reconstruction errors and critic scores.
    '''
    
    recon_score_params = params.get('recon_score_params', {})
    critic_score_params = params.get('critic_score_params', {})
    anomalous_seq_params = params.get('anomalous_seq_params', {})
    
    window_size = x_win_recon.shape[1]
    x_recon = merge_rolling_windows(x_win_recon)
    recon_scores = reconstruction_error_scores(x, x_recon, **recon_score_params)
    critic_scores = unroll_critic_scores(critic_scores, window_size, **critic_score_params)
    combined_scores = recon_scores + critic_scores
    seqs = identify_anomalous_sequences(combined_scores, t, **anomalous_seq_params)

    return seqs, x_recon, combined_scores


