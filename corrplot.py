import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import convolve, correlate

def ricker_wavelet(f, dt, length):
    """
    Generate a Ricker wavelet.

    Args:
        f: Central frequency (Hz).
        dt: Sampling interval (s).
        length: Length of the wavelet in seconds.

    Returns:
        A NumPy array containing the Ricker wavelet.
    """
    t = np.arange(-length / 2, length / 2, dt)
    y = (1 - 2 * (np.pi * f * t)**2) * np.exp(-(np.pi * f * t)**2)
    return y

def plot_cross_correlations(cross_correlations_array, angles, dt, maxlag, clippc, savefig):
    """
    Plots cross-correlations and their average.

    Parameters:
    cross_correlations_random_array : ndarray
        2D array of cross-correlations to plot as an image.
    average_cross_correlation_random : ndarray
        1D array of average cross-correlation to plot as a line.
    angles : ndarray
        Array of angles for the x-axis of the cross-correlation image.
    dt : float
        Time step for lag axis.
    maxlag : float
        Maximum lag for the lag-axis of the cross-correlation image.
    clippc : float
        Clipping value for colorbar.
    savefig : bool
        Whether to save the figure or show it.
    """
    # set default maxlag value to be dt * len(cross_correlations_array)//2
    if maxlag is None:
        maxlag = dt * len(cross_correlations_array)//2

    # Calculate the average cross-correlation across all angles
    average_cross_correlation = np.mean(cross_correlations_array, axis=1)

    # Create the figure and grid
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    # Plot the cross-correlation image
    ax0 = plt.subplot(gs[0])
    im = ax0.imshow(cross_correlations_array, aspect='auto', cmap='Greys',
                    extent=[angles[0], angles[-1], -dt * (len(cross_correlations_array)//2), dt * (len(cross_correlations_array)//2)])
    ax0.set_ylabel('Lag (sec)')
    ax0.set_ylim(-maxlag, maxlag)
    ax0.set_xlabel('Angle')
    ax0.set_title('Cross-Correlation for Each Source')
    im.set_clim(-clippc * max(np.abs(average_cross_correlation)), clippc * max(np.abs(average_cross_correlation)))

    # Format the x-axis to show ticks as multiples of π
    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    ax0.set_xticks(ticks)
    ax0.set_xticklabels([r"0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])

    # Plot the average cross-correlation on the right
    ax1 = plt.subplot(gs[1])
    lag_axis = np.arange(-len(average_cross_correlation) // 2, len(average_cross_correlation) // 2) * dt
    ax1.plot(average_cross_correlation, lag_axis, color='k', linewidth=2)
    ax1.set_title('Average Cross-Correlation')

    # Remove the frame and y-axis ticks
    ax1.invert_yaxis()
    ax1.set_frame_on(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_ylim(maxlag, -maxlag)

    # Adjust the layout and show or save the figure
    plt.tight_layout()

    if savefig:
        plt.savefig('cross_correlation.png', dpi=300)
    else:
        plt.show()


def plot_mc_records(record1, record2, taxis, plotN):
    """
    Plots every N angles as lines in record 1 and record 2 side by side.

    Parameters:
    record1 : ndarray
        3D array of record 1 with the third dimension in different component.
    record2 : ndarray
        3D array of record 2 with the third dimension in different component.
    taxis : ndarray
        Array of time values for the x-axis.
    plotN : int
        Number of angles to skip.
    """

    # Plot every 100th angle for record1 and record2 side by side
    for ic in range(record1.shape[2]):
      print("Plotting component: ", ic)

      plt.figure(figsize=(12, 6))

      for i in range(0, len(angles), plotN):
        plt.subplot(1, 2, 1)
        plt.plot(taxis, record1[:, i, ic])
        plt.title('Record 1')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(1, 2, 2)
        plt.plot(taxis, record2[:, i, ic])
        plt.title('Record 2')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')


      plt.tight_layout()
      plt.show()

# prompt: plot every N angles as lines in record 1 and record 2 side by side
def plot_records(record1, record2, taxis, plotN):
    """
    Plots every N angles as lines in record 1 and record 2 side by side.

    Parameters:
    record1 : ndarray
        2D array of record 1.
    record2 : ndarray
        2D array of record 2.
    taxis : ndarray
        Array of time values for the x-axis.
    plotN : int
        Number of angles to skip.
    """

    nt, nangle = record1.shape

    # Plot every 100th angle for record1 and record2 side by side
    plt.figure(figsize=(12, 6))

    for i in range(0, nangle-1, plotN):
      plt.subplot(1, 2, 1)
      plt.plot(taxis, record1[:, i])
      plt.title('Record 1')
      plt.xlabel('Time (s)')
      plt.ylabel('Amplitude')

      plt.subplot(1, 2, 2)
      plt.plot(taxis, record2[:, i])
      plt.title('Record 2')
      plt.xlabel('Time (s)')
      plt.ylabel('Amplitude')


    plt.tight_layout()
    plt.show()

def plot_average_spectra(record1, record2, taxis, fmin, fmax):
    """Plots the average spectra of record1 and record2.
    Inputs:
        record1: 2D array of record 1.
        record2: 2D array of record 2.
        taxis: 1D array of time values for the x-axis.
        fmin: minimum frequency to plot.
        fmax: maximum frequency to plot.
    """

    plt.figure(figsize=(10, 6))

    # Calculate the average spectra
    avg_spectrum1 = np.mean(np.abs(np.fft.fft(record1, axis=0)), axis=1)
    avg_spectrum2 = np.mean(np.abs(np.fft.fft(record2, axis=0)), axis=1)

    # Frequencies
    frequencies = np.fft.fftfreq(len(taxis), d=taxis[1] - taxis[0])

    # Plot the spectra
    plt.plot(frequencies[:len(frequencies)//2], avg_spectrum1[:len(avg_spectrum1)//2], label='Record 1')
    plt.plot(frequencies[:len(frequencies)//2], avg_spectrum2[:len(avg_spectrum2)//2], label='Record 2')
    plt.xlim(fmin, fmax)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Average Spectra of Record 1 and Record 2')
    plt.legend()
    plt.grid(True)
    plt.show()