from scipy.interpolate import interp1d
import numpy as np
from scipy import signal, ndimage
import librosa
from matplotlib import pyplot as plt
from numba import jit

@jit(nopython=True)
def compute_local_average(x, M):
    """Compute local average of signal

    Args:
        x (np.ndarray): Signal
        M (int): Determines size (2M+1) in samples of centric window  used for local average

    Returns:
        local_average (np.ndarray): Local average signal
    """
    L = len(x)
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average


def compute_novelty_complex(x, Fs=1, N=1024, H=64, gamma=10.0, M=40, norm=True):
    """Compute complex-domain novelty function

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = 1024)
        H (int): Hop size (Default value = 64)
        gamma (float): Parameter for logarithmic compression (Default value = 10.0)
        M (int): Determines size (2M+1) in samples of centric window used for local average (Default value = 40)
        norm (bool): Apply max norm (if norm==True) (Default value = True)

    Returns:
        novelty_complex (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Fs_feature = Fs / H
    mag = np.abs(X)
    if gamma > 0:
        mag = np.log(1 + gamma * mag)
    phase = np.angle(X) / (2*np.pi)
    phase_diff = np.diff(phase, axis=1)
    phase_diff = np.concatenate((phase_diff, np.zeros((phase.shape[0], 1))), axis=1)
    X_hat = mag * np.exp(2*np.pi*1j*(phase+phase_diff))
    X_prime = np.abs(X_hat - X)
    X_plus = np.copy(X_prime)
    for n in range(1, X.shape[0]):
        idx = np.where(mag[n, :] < mag[n-1, :])
        X_plus[n, idx] = 0
    novelty_complex = np.sum(X_plus, axis=0)
    if M > 0:
        local_average = compute_local_average(novelty_complex, M)
        novelty_complex = novelty_complex - local_average
        novelty_complex[novelty_complex < 0] = 0
    if norm:
        max_value = np.max(novelty_complex)
        if max_value > 0:
            novelty_complex = novelty_complex / max_value
    return novelty_complex, Fs_feature


def resample_signal(x_in, Fs_in, Fs_out=100, norm=True, time_max_sec=None, sigma=None):
    """Resample and smooth signal

    Notebook: C6/C6S1_NoveltyComparison.ipynb

    Args:
        x_in (np.ndarray): Input signal
        Fs_in (scalar): Sampling rate of input signal
        Fs_out (scalar): Sampling rate of output signal (Default value = 100)
        norm (bool): Apply max norm (if norm==True) (Default value = True)
        time_max_sec (float): Duration of output signal (given in seconds) (Default value = None)
        sigma (float): Standard deviation for smoothing Gaussian kernel (Default value = None)

    Returns:
        x_out (np.ndarray): Output signal
        Fs_out (scalar): Feature rate of output signal
    """
    if sigma is not None:
        x_in = ndimage.gaussian_filter(x_in, sigma=sigma)
    T_coef_in = np.arange(x_in.shape[0]) / Fs_in
    time_in_max_sec = T_coef_in[-1]
    if time_max_sec is None:
        time_max_sec = time_in_max_sec
    N_out = int(np.ceil(time_max_sec*Fs_out))
    T_coef_out = np.arange(N_out) / Fs_out
    if T_coef_out[-1] > time_in_max_sec:
        x_in = np.append(x_in, [0])
        T_coef_in = np.append(T_coef_in, [T_coef_out[-1]])
    x_out = interp1d(T_coef_in, x_in, kind='linear')(T_coef_out)
    if norm:
        x_max = max(x_out)
        if x_max > 0:
            x_out = x_out / max(x_out)
    return x_out, Fs_out

def average_nov_dic(nov_dic, time_max_sec, Fs_out=100, norm=True, sigma=None):
    """Average respamples set of novelty functions

    Notebook: C6/C6S1_NoveltyComparison.ipynb

    Args:
        nov_dic (dict): Dictionary of novelty functions
        time_max_sec (float): Duration of output signals (given in seconds)
        Fs_out (scalar): Sampling rate of output signal (Default value = 100)
        norm (bool): Apply max norm (if norm==True) (Default value = True)
        sigma (float): Standard deviation for smoothing Gaussian kernel (Default value = None)

    Returns:
        nov_matrix (np.ndarray): Matrix containing resampled output signal (last one is average)
        Fs_out (scalar): Sampling rate of output signals
    """
    nov_num = len(nov_dic)
    N_out = int(np.ceil(time_max_sec*Fs_out))
    nov_matrix = np.zeros([nov_num + 1, N_out])
    for k in range(nov_num):
        nov = nov_dic[k][0]
        Fs_nov = nov_dic[k][1]
        nov_out, Fs_out = resample_signal(nov, Fs_in=Fs_nov, Fs_out=Fs_out,
                                          time_max_sec=time_max_sec, sigma=sigma)
        nov_matrix[k, :] = nov_out
    nov_average = np.sum(nov_matrix, axis=0)/nov_num
    if norm:
        max_value = np.max(nov_average)
        if max_value > 0:
            nov_average = nov_average / max_value
    nov_matrix[nov_num, :] = nov_average
    return nov_matrix, Fs_out

def principal_argument(v):
    """Principal argument function

    | Notebook: C6/C6S1_NoveltyPhase.ipynb, see also
    | Notebook: C8/C8S2_InstantFreqEstimation.ipynb

    Args:
        v (float or np.ndarray): Value (or vector of values)

    Returns:
        w (float or np.ndarray): Principle value of v
    """
    w = np.mod(v + 0.5, 1) - 0.5
    return w


def compute_novelty_energy(x, Fs=1, N=2048, H=128, gamma=10.0, norm=True):
    """Compute energy-based novelty function

    Notebook: C6/C6S1_NoveltyEnergy.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = 2048)
        H (int): Hop size (Default value = 128)
        gamma (float): Parameter for logarithmic compression (Default value = 10.0)
        norm (bool): Apply max norm (if norm==True) (Default value = True)

    Returns:
        novelty_energy (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
    """
    # x_power = x**2
    w = signal.hann(N)
    Fs_feature = Fs / H
    energy_local = np.convolve(x**2, w**2, 'same')
    energy_local = energy_local[::H]
    if gamma is not None:
        energy_local = np.log(1 + gamma * energy_local)
    energy_local_diff = np.diff(energy_local)
    energy_local_diff = np.concatenate((energy_local_diff, np.array([0])))
    novelty_energy = np.copy(energy_local_diff)
    novelty_energy[energy_local_diff < 0] = 0
    if norm:
        max_value = max(novelty_energy)
        if max_value > 0:
            novelty_energy = novelty_energy / max_value
    return novelty_energy, Fs_feature

def compute_novelty_spectrum(x, Fs=1, N=1024, H=256, gamma=100.0, M=10, norm=True):
    """Compute spectral-based novelty function

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = 1024)
        H (int): Hop size (Default value = 256)
        gamma (float): Parameter for logarithmic compression (Default value = 100.0)
        M (int): Size (frames) of local average (Default value = 10)
        norm (bool): Apply max norm (if norm==True) (Default value = True)

    Returns:
        novelty_spectrum (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Fs_feature = Fs / H
    Y = np.log(1 + gamma * np.abs(X))
    Y_diff = np.diff(Y)
    Y_diff[Y_diff < 0] = 0
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    return novelty_spectrum, Fs_feature

def compute_novelty_phase(x, Fs=1, N=1024, H=64, M=40, norm=True):
    """Compute phase-based novelty function

    Notebook: C6/C6S1_NoveltyPhase.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = 1024)
        H (int): Hop size (Default value = 64)
        M (int): Determines size (2M+1) in samples of centric window  used for local average (Default value = 40)
        norm (bool): Apply max norm (if norm==True) (Default value = True)

    Returns:
        novelty_phase (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Fs_feature = Fs / H
    phase = np.angle(X) / (2*np.pi)
    phase_diff = principal_argument(np.diff(phase, axis=1))
    phase_diff2 = principal_argument(np.diff(phase_diff, axis=1))
    novelty_phase = np.sum(np.abs(phase_diff2), axis=0)
    novelty_phase = np.concatenate((novelty_phase, np.array([0, 0])))
    if M > 0:
        local_average = compute_local_average(novelty_phase, M)
        novelty_phase = novelty_phase - local_average
        novelty_phase[novelty_phase < 0] = 0
    if norm:
        max_value = np.max(novelty_phase)
        if max_value > 0:
            novelty_phase = novelty_phase / max_value
    return novelty_phase, Fs_feature



def peak_picking_roeder(x, direction=None, abs_thresh=None, rel_thresh=None, descent_thresh=None, tmin=None, tmax=None):
    """| Computes the positive peaks of the input vector x

    Args:
        x (np.nadarray): Signal to be searched for (positive) peaks
        direction (int): +1 for forward peak searching, -1 for backward peak searching.
            default is dir == -1. (Default value = None)
        abs_thresh (float): Absolute threshold signal, i.e. only peaks
            satisfying x(i)>=abs_thresh(i) will be reported.
            abs_thresh must have the same number of samples as x.
            a sensible choice for this parameter would be a global or local
            average or median of the signal x.
            If omitted, half the median of x will be used. (Default value = None)
        rel_thresh (float): Relative threshold signal. Only peak positions i with an
            uninterrupted positive ascent before position i of at least
            rel_thresh(i) and a possibly interrupted (see parameter descent_thresh)
            descent of at least rel_thresh(i) will be reported.
            rel_thresh must have the same number of samples as x.
            A sensible choice would be some measure related to the
            global or local variance of the signal x.
            if omitted, half the standard deviation of W will be used.
        descent_thresh (float): Descent threshold. during peak candidate verfication, if a slope change
            from negative to positive slope occurs at sample i BEFORE the descent has
            exceeded rel_thresh(i), and if descent_thresh(i) has not been exceeded yet,
            the current peak candidate will be dropped.
            this situation corresponds to a secondary peak
            occuring shortly after the current candidate peak (which might lead
            to a higher peak value)!
            |
            | The value descent_thresh(i) must not be larger than rel_thresh(i).
            |
            | descent_thresh must have the same number of samples as x.
            a sensible choice would be some measure related to the
            global or local variance of the signal x.
            if omitted, 0.5*rel_thresh will be used. (Default value = None)
        tmin (int): Index of start sample. peak search will begin at x(tmin). (Default value = None)
        tmax (int): Index of end sample. peak search will end at x(tmax). (Default value = None)

    Returns:
        peaks (np.nadarray): Array of peak positions
    """

    # set default values
    if direction is None:
        direction = -1
    if abs_thresh is None:
        abs_thresh = np.tile(0.5*np.median(x), len(x))
    if rel_thresh is None:
        rel_thresh = 0.5*np.tile(np.sqrt(np.var(x)), len(x))
    if descent_thresh is None:
        descent_thresh = 0.5*rel_thresh
    if tmin is None:
        tmin = 1
    if tmax is None:
        tmax = len(x)

    dyold = 0
    dy = 0
    rise = 0  # current amount of ascent during a rising portion of the signal x
    riseold = 0  # accumulated amount of ascent from the last rising portion of x
    descent = 0  # current amount of descent (<0) during a falling portion of the signal x
    searching_peak = True
    candidate = 1
    P = []

    if direction == 1:
        my_range = np.arange(tmin, tmax)
    elif direction == -1:
        my_range = np.arange(tmin, tmax)
        my_range = my_range[::-1]

    # run through x
    for cur_idx in my_range:
        # get local gradient
        dy = x[cur_idx+direction] - x[cur_idx]

        if dy >= 0:
            rise = rise + dy
        else:
            descent = descent + dy

        if dyold >= 0:
            if dy < 0:  # slope change positive->negative
                if rise >= rel_thresh[cur_idx] and searching_peak is True:
                    candidate = cur_idx
                    searching_peak = False
                riseold = rise
                rise = 0
        else:  # dyold < 0
            if dy < 0:  # in descent
                if descent <= -rel_thresh[candidate] and searching_peak is False:
                    if x[candidate] >= abs_thresh[candidate]:
                        P.append(candidate)  # verified candidate as True peak
                    searching_peak = True
            else:  # dy >= 0 slope change negative->positive
                if searching_peak is False:  # currently verifying a peak
                    if x[candidate] - x[cur_idx] <= descent_thresh[cur_idx]:
                        rise = riseold + descent  # skip intermediary peak
                    if descent <= -rel_thresh[candidate]:
                        if x[candidate] >= abs_thresh[candidate]:
                            P.append(candidate)    # verified candidate as True peak
                    searching_peak = True
                descent = 0
        dyold = dy
    peaks = np.array(P)
    return peaks


def plot_signal(x, Fs=1, T_coef=None, ax=None, figsize=(6, 2), xlabel='Time (seconds)', ylabel='', title='', dpi=72,
                ylim=True, **kwargs):
    """Line plot visualization of a signal, e.g. a waveform or a novelty function.

    Args:
        x: Input signal
        Fs: Sample rate (Default value = 1)
        T_coef: Time coeffients. If None, will be computed, based on Fs. (Default value = None)
        ax: The Axes instance to plot on. If None, will create a figure and axes. (Default value = None)
        figsize: Width, height in inches (Default value = (6, 2))
        xlabel: Label for x axis (Default value = 'Time (seconds)')
        ylabel: Label for y axis (Default value = '')
        title: Title for plot (Default value = '')
        dpi: Dots per inch (Default value = 72)
        ylim: True or False (auto adjust ylim or nnot) or tuple with actual ylim (Default value = True)
        **kwargs: Keyword arguments for matplotlib.pyplot.plot

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
        line: The line plot
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot(1, 1, 1)
    if T_coef is None:
        T_coef = np.arange(x.shape[0]) / Fs

    if 'color' not in kwargs:
        kwargs['color'] = 'gray'

    line = ax.plot(T_coef, x, **kwargs)

    ax.set_xlim([T_coef[0], T_coef[-1]])
    if ylim is True:
        ylim_x = x[np.isfinite(x)]
        x_min, x_max = ylim_x.min(), ylim_x.max()
        if x_max == x_min:
            x_max = x_max + 1
        ax.set_ylim([min(1.1 * x_min, 0.9 * x_min), max(1.1 * x_max, 0.9 * x_max)])
    elif ylim not in [True, False, None]:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if fig is not None:
        plt.tight_layout()

    return fig, ax, line

def plot_function_peak_positions(nov, Fs_nov, peaks, title='', figsize=(8,2)):
    peaks_sec = peaks/Fs_nov
    fig, ax, line = plot_signal(nov, Fs_nov, figsize=figsize, color='k', title=title);
    plt.vlines(peaks_sec, 0, 1.1, color='r', linestyle=':', linewidth=1);
    return peaks_sec


def quantize_n_sec2beat(events,bpm):
    sec_per_16th_note = 15 / bpm
    a = (events // sec_per_16th_note)
    return a - a[0]