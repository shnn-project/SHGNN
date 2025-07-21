"""
SHNN Utility Functions

Additional Python utilities for the SHNN library.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union

def spike_raster_plot(spikes, ax=None, neuron_ids=None, time_range=None, **kwargs):
    """
    Create a spike raster plot.
    
    Args:
        spikes: List of Spike objects or spike data
        ax: Matplotlib axis (optional)
        neuron_ids: List of neuron IDs to plot (optional)
        time_range: Tuple of (start_time, end_time) (optional)
        **kwargs: Additional matplotlib arguments
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting functions")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract spike times and neuron IDs
    spike_times = []
    spike_neurons = []
    
    for spike in spikes:
        if hasattr(spike, 'time') and hasattr(spike, 'neuron_id'):
            # SHNN Spike object
            time = spike.time
            neuron_id = spike.neuron_id
        elif isinstance(spike, (tuple, list)) and len(spike) >= 2:
            # Tuple format (neuron_id, time)
            neuron_id, time = spike[0], spike[1]
        else:
            continue
            
        if neuron_ids is None or neuron_id in neuron_ids:
            if time_range is None or (time_range[0] <= time <= time_range[1]):
                spike_times.append(time)
                spike_neurons.append(neuron_id)
    
    # Create raster plot
    ax.scatter(spike_times, spike_neurons, s=1, **kwargs)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron ID')
    ax.set_title('Spike Raster Plot')
    
    if time_range:
        ax.set_xlim(time_range)
    
    return ax

def firing_rate_histogram(spikes, bin_size=0.01, time_range=None):
    """
    Calculate firing rate histogram.
    
    Args:
        spikes: List of Spike objects
        bin_size: Size of time bins in seconds
        time_range: Tuple of (start_time, end_time)
    
    Returns:
        Tuple of (bin_centers, firing_rates)
    """
    spike_times = []
    for spike in spikes:
        if hasattr(spike, 'time'):
            time = spike.time
        elif isinstance(spike, (tuple, list)) and len(spike) >= 2:
            time = spike[1]
        else:
            continue
            
        if time_range is None or (time_range[0] <= time <= time_range[1]):
            spike_times.append(time)
    
    if not spike_times:
        return np.array([]), np.array([])
    
    # Determine time range
    if time_range is None:
        min_time, max_time = min(spike_times), max(spike_times)
    else:
        min_time, max_time = time_range
    
    # Create histogram
    bins = np.arange(min_time, max_time + bin_size, bin_size)
    counts, bin_edges = np.histogram(spike_times, bins=bins)
    
    # Calculate firing rates (spikes per second)
    firing_rates = counts / bin_size
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, firing_rates

def isi_analysis(spikes, neuron_id=None):
    """
    Analyze inter-spike intervals.
    
    Args:
        spikes: List of Spike objects
        neuron_id: Specific neuron ID to analyze (None for all)
    
    Returns:
        Dictionary with ISI statistics
    """
    spike_times = []
    
    for spike in spikes:
        if hasattr(spike, 'time') and hasattr(spike, 'neuron_id'):
            if neuron_id is None or spike.neuron_id == neuron_id:
                spike_times.append(spike.time)
        elif isinstance(spike, (tuple, list)) and len(spike) >= 2:
            if neuron_id is None or spike[0] == neuron_id:
                spike_times.append(spike[1])
    
    spike_times = sorted(spike_times)
    
    if len(spike_times) < 2:
        return {'isi': [], 'mean_isi': 0, 'std_isi': 0, 'cv_isi': 0}
    
    # Calculate ISIs
    isis = np.diff(spike_times)
    
    return {
        'isi': isis.tolist(),
        'mean_isi': float(np.mean(isis)),
        'std_isi': float(np.std(isis)),
        'cv_isi': float(np.std(isis) / np.mean(isis)) if np.mean(isis) > 0 else 0,
        'min_isi': float(np.min(isis)),
        'max_isi': float(np.max(isis))
    }

def spikes_to_numpy(spikes, max_time=None, bin_size=0.001):
    """
    Convert spike trains to NumPy arrays.
    
    Args:
        spikes: List of Spike objects
        max_time: Maximum time for the array
        bin_size: Time bin size
    
    Returns:
        2D NumPy array (neurons x time_bins)
    """
    if not spikes:
        return np.array([])
    
    # Extract data
    spike_data = []
    for spike in spikes:
        if hasattr(spike, 'time') and hasattr(spike, 'neuron_id'):
            spike_data.append((spike.neuron_id, spike.time))
        elif isinstance(spike, (tuple, list)) and len(spike) >= 2:
            spike_data.append((spike[0], spike[1]))
    
    if not spike_data:
        return np.array([])
    
    # Determine dimensions
    neuron_ids = [s[0] for s in spike_data]
    spike_times = [s[1] for s in spike_data]
    
    max_neuron = max(neuron_ids)
    if max_time is None:
        max_time = max(spike_times)
    
    time_bins = int(max_time / bin_size) + 1
    
    # Create array
    spike_array = np.zeros((max_neuron + 1, time_bins))
    
    for neuron_id, spike_time in spike_data:
        time_bin = int(spike_time / bin_size)
        if 0 <= time_bin < time_bins:
            spike_array[neuron_id, time_bin] += 1
    
    return spike_array

def calculate_synchrony(spikes, time_window=0.01):
    """
    Calculate population synchrony measure.
    
    Args:
        spikes: List of Spike objects
        time_window: Time window for synchrony calculation
    
    Returns:
        Synchrony index (0-1)
    """
    if not spikes:
        return 0.0
    
    spike_times = []
    for spike in spikes:
        if hasattr(spike, 'time'):
            spike_times.append(spike.time)
        elif isinstance(spike, (tuple, list)) and len(spike) >= 2:
            spike_times.append(spike[1])
    
    if len(spike_times) < 2:
        return 0.0
    
    spike_times = sorted(spike_times)
    max_time = spike_times[-1]
    
    # Create time bins
    bins = np.arange(0, max_time + time_window, time_window)
    counts, _ = np.histogram(spike_times, bins=bins)
    
    # Calculate synchrony as coefficient of variation
    if len(counts) > 0 and np.mean(counts) > 0:
        synchrony = np.std(counts) / np.mean(counts)
    else:
        synchrony = 0.0
    
    return min(synchrony, 1.0)  # Cap at 1.0

def downsample_image(image, factor=2):
    """
    Downsample an image by averaging.
    
    Args:
        image: 2D numpy array
        factor: Downsampling factor
    
    Returns:
        Downsampled image
    """
    h, w = image.shape
    new_h, new_w = h // factor, w // factor
    
    # Crop to ensure even dimensions
    cropped = image[:new_h*factor, :new_w*factor]
    
    # Reshape and average
    downsampled = cropped.reshape(new_h, factor, new_w, factor).mean(axis=(1, 3))
    
    return downsampled

def normalize_image(image, method='minmax'):
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image array
        method: 'minmax' or 'zscore'
    
    Returns:
        Normalized image
    """
    if method == 'minmax':
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        else:
            return np.zeros_like(image)
    elif method == 'zscore':
        img_mean, img_std = image.mean(), image.std()
        if img_std > 0:
            return (image - img_mean) / img_std
        else:
            return np.zeros_like(image)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def performance_benchmark(func, *args, num_runs=10, **kwargs):
    """
    Benchmark function performance.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        num_runs: Number of benchmark runs
        **kwargs: Function keyword arguments
    
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'mean_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'num_runs': num_runs,
        'result': result  # Last result
    }