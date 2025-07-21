//! Python bindings for visualization and plotting utilities
//!
//! This module provides Python interfaces for creating plots and visualizations
//! of spiking neural network data, including raster plots, membrane potential traces,
//! weight matrices, and network connectivity graphs.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::{PyRuntimeError, PyValueError, PyImportError};

use shnn_core::{
    spike::{Spike, SpikeTime},
    neuron::NeuronState,
};

use crate::neuron::PySpike;
use crate::error_conversion::core_error_to_py_err;

/// Python wrapper for raster plot data
#[pyclass(name = "RasterPlot")]
pub struct PyRasterPlot {
    spike_times: Vec<f64>,
    neuron_ids: Vec<u32>,
    time_window: (f64, f64),
    neuron_range: (u32, u32),
}

#[pymethods]
impl PyRasterPlot {
    #[new]
    fn new(spikes: &PyList, time_window: Option<(f64, f64)>, neuron_range: Option<(u32, u32)>) -> PyResult<Self> {
        let mut spike_times = Vec::new();
        let mut neuron_ids = Vec::new();
        
        let mut min_time = f64::INFINITY;
        let mut max_time = f64::NEG_INFINITY;
        let mut min_neuron = u32::MAX;
        let mut max_neuron = u32::MIN;
        
        // Extract spike data
        for spike_obj in spikes.iter() {
            let py_spike: PySpike = spike_obj.extract()?;
            let time = py_spike.time();
            let neuron_id = py_spike.neuron_id();
            
            spike_times.push(time);
            neuron_ids.push(neuron_id);
            
            min_time = min_time.min(time);
            max_time = max_time.max(time);
            min_neuron = min_neuron.min(neuron_id);
            max_neuron = max_neuron.max(neuron_id);
        }
        
        let time_window = time_window.unwrap_or((min_time, max_time));
        let neuron_range = neuron_range.unwrap_or((min_neuron, max_neuron));
        
        Ok(Self {
            spike_times,
            neuron_ids,
            time_window,
            neuron_range,
        })
    }
    
    /// Get spike times
    fn get_spike_times(&self) -> Vec<f64> {
        self.spike_times.clone()
    }
    
    /// Get neuron IDs
    fn get_neuron_ids(&self) -> Vec<u32> {
        self.neuron_ids.clone()
    }
    
    /// Get time window
    fn get_time_window(&self) -> (f64, f64) {
        self.time_window
    }
    
    /// Get neuron range
    fn get_neuron_range(&self) -> (u32, u32) {
        self.neuron_range
    }
    
    /// Filter spikes by time window
    fn filter_by_time(&self, start_time: f64, end_time: f64) -> Self {
        let mut filtered_times = Vec::new();
        let mut filtered_neurons = Vec::new();
        
        for (i, &time) in self.spike_times.iter().enumerate() {
            if time >= start_time && time <= end_time {
                filtered_times.push(time);
                filtered_neurons.push(self.neuron_ids[i]);
            }
        }
        
        Self {
            spike_times: filtered_times,
            neuron_ids: filtered_neurons,
            time_window: (start_time, end_time),
            neuron_range: self.neuron_range,
        }
    }
    
    /// Filter spikes by neuron range
    fn filter_by_neurons(&self, min_neuron: u32, max_neuron: u32) -> Self {
        let mut filtered_times = Vec::new();
        let mut filtered_neurons = Vec::new();
        
        for (i, &neuron_id) in self.neuron_ids.iter().enumerate() {
            if neuron_id >= min_neuron && neuron_id <= max_neuron {
                filtered_times.push(self.spike_times[i]);
                filtered_neurons.push(neuron_id);
            }
        }
        
        Self {
            spike_times: filtered_times,
            neuron_ids: filtered_neurons,
            time_window: self.time_window,
            neuron_range: (min_neuron, max_neuron),
        }
    }
    
    /// Get binned spike counts
    fn get_binned_counts(&self, bin_size: f64) -> PyResult<(Vec<f64>, Vec<u32>)> {
        if bin_size <= 0.0 {
            return Err(PyValueError::new_err("Bin size must be positive"));
        }
        
        let (start_time, end_time) = self.time_window;
        let num_bins = ((end_time - start_time) / bin_size).ceil() as usize;
        let mut bin_centers = Vec::with_capacity(num_bins);
        let mut bin_counts = vec![0u32; num_bins];
        
        // Create bin centers
        for i in 0..num_bins {
            bin_centers.push(start_time + (i as f64 + 0.5) * bin_size);
        }
        
        // Count spikes in each bin
        for &time in &self.spike_times {
            if time >= start_time && time <= end_time {
                let bin_idx = ((time - start_time) / bin_size).floor() as usize;
                if bin_idx < num_bins {
                    bin_counts[bin_idx] += 1;
                }
            }
        }
        
        Ok((bin_centers, bin_counts))
    }
    
    /// Calculate firing rate over time
    fn get_firing_rate(&self, window_size: f64, step_size: Option<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if window_size <= 0.0 {
            return Err(PyValueError::new_err("Window size must be positive"));
        }
        
        let step = step_size.unwrap_or(window_size / 10.0);
        let (start_time, end_time) = self.time_window;
        let num_neurons = (self.neuron_range.1 - self.neuron_range.0 + 1) as f64;
        
        let mut time_points = Vec::new();
        let mut rates = Vec::new();
        
        let mut current_time = start_time + window_size / 2.0;
        while current_time <= end_time - window_size / 2.0 {
            let window_start = current_time - window_size / 2.0;
            let window_end = current_time + window_size / 2.0;
            
            let spike_count = self.spike_times.iter()
                .filter(|&&time| time >= window_start && time <= window_end)
                .count() as f64;
            
            let rate = spike_count / (window_size * num_neurons);
            
            time_points.push(current_time);
            rates.push(rate);
            
            current_time += step;
        }
        
        Ok((time_points, rates))
    }
    
    /// Export plot data for matplotlib
    fn to_matplotlib_data(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("spike_times", &self.spike_times)?;
            dict.set_item("neuron_ids", &self.neuron_ids)?;
            dict.set_item("time_window", self.time_window)?;
            dict.set_item("neuron_range", self.neuron_range)?;
            Ok(dict.to_object(py))
        })
    }
    
    fn __len__(&self) -> usize {
        self.spike_times.len()
    }
    
    fn __repr__(&self) -> String {
        format!(
            "RasterPlot({} spikes, time={:.3}-{:.3}s, neurons={}-{})",
            self.spike_times.len(),
            self.time_window.0,
            self.time_window.1,
            self.neuron_range.0,
            self.neuron_range.1
        )
    }
}

/// Python wrapper for membrane potential trace
#[pyclass(name = "MembraneTrace")]
pub struct PyMembraneTrace {
    times: Vec<f64>,
    voltages: Vec<f32>,
    spike_times: Vec<f64>,
    neuron_id: u32,
}

#[pymethods]
impl PyMembraneTrace {
    #[new]
    fn new(times: Vec<f64>, voltages: Vec<f32>, spike_times: Option<Vec<f64>>, neuron_id: Option<u32>) -> PyResult<Self> {
        if times.len() != voltages.len() {
            return Err(PyValueError::new_err("Times and voltages must have same length"));
        }
        
        Ok(Self {
            times,
            voltages,
            spike_times: spike_times.unwrap_or_default(),
            neuron_id: neuron_id.unwrap_or(0),
        })
    }
    
    /// Get time points
    fn get_times(&self) -> Vec<f64> {
        self.times.clone()
    }
    
    /// Get voltage values
    fn get_voltages(&self) -> Vec<f32> {
        self.voltages.clone()
    }
    
    /// Get spike times
    fn get_spike_times(&self) -> Vec<f64> {
        self.spike_times.clone()
    }
    
    /// Get neuron ID
    fn get_neuron_id(&self) -> u32 {
        self.neuron_id
    }
    
    /// Add spike markers
    fn add_spikes(&mut self, spike_times: Vec<f64>) {
        self.spike_times = spike_times;
    }
    
    /// Filter trace by time window
    fn filter_by_time(&self, start_time: f64, end_time: f64) -> Self {
        let mut filtered_times = Vec::new();
        let mut filtered_voltages = Vec::new();
        
        for (i, &time) in self.times.iter().enumerate() {
            if time >= start_time && time <= end_time {
                filtered_times.push(time);
                filtered_voltages.push(self.voltages[i]);
            }
        }
        
        let filtered_spikes: Vec<f64> = self.spike_times.iter()
            .filter(|&&time| time >= start_time && time <= end_time)
            .cloned()
            .collect();
        
        Self {
            times: filtered_times,
            voltages: filtered_voltages,
            spike_times: filtered_spikes,
            neuron_id: self.neuron_id,
        }
    }
    
    /// Calculate voltage statistics
    fn get_voltage_stats(&self) -> PyResult<PyObject> {
        if self.voltages.is_empty() {
            return Err(PyValueError::new_err("No voltage data"));
        }
        
        let mut voltages_f64: Vec<f64> = self.voltages.iter().map(|&v| v as f64).collect();
        voltages_f64.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mean = voltages_f64.iter().sum::<f64>() / voltages_f64.len() as f64;
        let variance = voltages_f64.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / voltages_f64.len() as f64;
        let std_dev = variance.sqrt();
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("mean", mean)?;
            dict.set_item("std", std_dev)?;
            dict.set_item("min", voltages_f64[0])?;
            dict.set_item("max", voltages_f64[voltages_f64.len() - 1])?;
            dict.set_item("median", voltages_f64[voltages_f64.len() / 2])?;
            dict.set_item("num_spikes", self.spike_times.len())?;
            Ok(dict.to_object(py))
        })
    }
    
    /// Export trace data for matplotlib
    fn to_matplotlib_data(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("times", &self.times)?;
            dict.set_item("voltages", &self.voltages)?;
            dict.set_item("spike_times", &self.spike_times)?;
            dict.set_item("neuron_id", self.neuron_id)?;
            Ok(dict.to_object(py))
        })
    }
    
    fn __len__(&self) -> usize {
        self.times.len()
    }
    
    fn __repr__(&self) -> String {
        format!(
            "MembraneTrace(neuron={}, {} points, {} spikes)",
            self.neuron_id,
            self.times.len(),
            self.spike_times.len()
        )
    }
}

/// Python wrapper for weight matrix visualization
#[pyclass(name = "WeightMatrix")]
pub struct PyWeightMatrix {
    weights: Vec<Vec<f32>>,
    pre_neuron_ids: Vec<u32>,
    post_neuron_ids: Vec<u32>,
}

#[pymethods]
impl PyWeightMatrix {
    #[new]
    fn new(
        weights: Vec<Vec<f32>>,
        pre_neuron_ids: Option<Vec<u32>>,
        post_neuron_ids: Option<Vec<u32>>,
    ) -> PyResult<Self> {
        if weights.is_empty() {
            return Err(PyValueError::new_err("Weight matrix cannot be empty"));
        }
        
        let rows = weights.len();
        let cols = weights[0].len();
        
        // Check all rows have same length
        for row in &weights {
            if row.len() != cols {
                return Err(PyValueError::new_err("All rows must have same length"));
            }
        }
        
        let pre_ids = pre_neuron_ids.unwrap_or_else(|| (0..rows as u32).collect());
        let post_ids = post_neuron_ids.unwrap_or_else(|| (0..cols as u32).collect());
        
        if pre_ids.len() != rows {
            return Err(PyValueError::new_err("Pre-neuron IDs length must match number of rows"));
        }
        if post_ids.len() != cols {
            return Err(PyValueError::new_err("Post-neuron IDs length must match number of columns"));
        }
        
        Ok(Self {
            weights,
            pre_neuron_ids: pre_ids,
            post_neuron_ids: post_ids,
        })
    }
    
    /// Get weight matrix
    fn get_weights(&self) -> Vec<Vec<f32>> {
        self.weights.clone()
    }
    
    /// Get pre-synaptic neuron IDs
    fn get_pre_neuron_ids(&self) -> Vec<u32> {
        self.pre_neuron_ids.clone()
    }
    
    /// Get post-synaptic neuron IDs
    fn get_post_neuron_ids(&self) -> Vec<u32> {
        self.post_neuron_ids.clone()
    }
    
    /// Get matrix dimensions
    fn get_shape(&self) -> (usize, usize) {
        (self.weights.len(), self.weights[0].len())
    }
    
    /// Get weight statistics
    fn get_weight_stats(&self) -> PyResult<PyObject> {
        let mut all_weights = Vec::new();
        for row in &self.weights {
            all_weights.extend(row.iter().map(|&w| w as f64));
        }
        
        if all_weights.is_empty() {
            return Err(PyValueError::new_err("No weight data"));
        }
        
        all_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mean = all_weights.iter().sum::<f64>() / all_weights.len() as f64;
        let variance = all_weights.iter()
            .map(|w| (w - mean).powi(2))
            .sum::<f64>() / all_weights.len() as f64;
        let std_dev = variance.sqrt();
        
        let num_positive = all_weights.iter().filter(|&&w| w > 0.0).count();
        let num_negative = all_weights.iter().filter(|&&w| w < 0.0).count();
        let num_zero = all_weights.iter().filter(|&&w| w == 0.0).count();
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("mean", mean)?;
            dict.set_item("std", std_dev)?;
            dict.set_item("min", all_weights[0])?;
            dict.set_item("max", all_weights[all_weights.len() - 1])?;
            dict.set_item("median", all_weights[all_weights.len() / 2])?;
            dict.set_item("num_positive", num_positive)?;
            dict.set_item("num_negative", num_negative)?;
            dict.set_item("num_zero", num_zero)?;
            dict.set_item("sparsity", num_zero as f64 / all_weights.len() as f64)?;
            Ok(dict.to_object(py))
        })
    }
    
    /// Get weight distribution histogram
    fn get_weight_distribution(&self, num_bins: Option<u32>) -> (Vec<f32>, Vec<u32>) {
        let mut all_weights = Vec::new();
        for row in &self.weights {
            all_weights.extend(row.iter().cloned());
        }
        
        if all_weights.is_empty() {
            return (Vec::new(), Vec::new());
        }
        
        let bins = num_bins.unwrap_or(50);
        all_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let min_weight = all_weights[0];
        let max_weight = all_weights[all_weights.len() - 1];
        
        if min_weight == max_weight {
            return (vec![min_weight], vec![all_weights.len() as u32]);
        }
        
        let bin_width = (max_weight - min_weight) / bins as f32;
        let mut bin_edges = Vec::with_capacity(bins as usize);
        let mut bin_counts = vec![0u32; bins as usize];
        
        // Create bin centers
        for i in 0..bins {
            bin_edges.push(min_weight + (i as f32 + 0.5) * bin_width);
        }
        
        // Count weights in each bin
        for &weight in &all_weights {
            let bin_idx = ((weight - min_weight) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins as usize - 1);
            bin_counts[bin_idx] += 1;
        }
        
        (bin_edges, bin_counts)
    }
    
    /// Filter matrix by weight range
    fn filter_by_weight_range(&self, min_weight: f32, max_weight: f32) -> Self {
        let mut filtered_weights = Vec::new();
        
        for row in &self.weights {
            let mut filtered_row = Vec::new();
            for &weight in row {
                if weight >= min_weight && weight <= max_weight {
                    filtered_row.push(weight);
                } else {
                    filtered_row.push(0.0);
                }
            }
            filtered_weights.push(filtered_row);
        }
        
        Self {
            weights: filtered_weights,
            pre_neuron_ids: self.pre_neuron_ids.clone(),
            post_neuron_ids: self.post_neuron_ids.clone(),
        }
    }
    
    /// Export matrix data for matplotlib
    fn to_matplotlib_data(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("weights", &self.weights)?;
            dict.set_item("pre_neuron_ids", &self.pre_neuron_ids)?;
            dict.set_item("post_neuron_ids", &self.post_neuron_ids)?;
            dict.set_item("shape", self.get_shape())?;
            Ok(dict.to_object(py))
        })
    }
    
    fn __repr__(&self) -> String {
        let (rows, cols) = self.get_shape();
        format!("WeightMatrix({}x{})", rows, cols)
    }
}

/// Utility functions for creating visualizations
#[pyfunction]
pub fn create_raster_plot(spikes: &PyList, time_window: Option<(f64, f64)>, neuron_range: Option<(u32, u32)>) -> PyResult<PyRasterPlot> {
    PyRasterPlot::new(spikes, time_window, neuron_range)
}

#[pyfunction]
pub fn create_membrane_trace(
    times: Vec<f64>,
    voltages: Vec<f32>,
    spike_times: Option<Vec<f64>>,
    neuron_id: Option<u32>,
) -> PyResult<PyMembraneTrace> {
    PyMembraneTrace::new(times, voltages, spike_times, neuron_id)
}

#[pyfunction]
pub fn create_weight_matrix(
    weights: Vec<Vec<f32>>,
    pre_neuron_ids: Option<Vec<u32>>,
    post_neuron_ids: Option<Vec<u32>>,
) -> PyResult<PyWeightMatrix> {
    PyWeightMatrix::new(weights, pre_neuron_ids, post_neuron_ids)
}

#[pyfunction]
pub fn plot_spike_raster(
    spikes: &PyList,
    title: Option<&str>,
    xlabel: Option<&str>,
    ylabel: Option<&str>,
    figsize: Option<(f64, f64)>,
    save_path: Option<&str>,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // Try to import matplotlib
        let plt = py.import("matplotlib.pyplot").map_err(|_| {
            PyImportError::new_err("matplotlib not available. Install with: pip install matplotlib")
        })?;
        
        let raster = PyRasterPlot::new(spikes, None, None)?;
        let spike_times = raster.get_spike_times();
        let neuron_ids = raster.get_neuron_ids();
        
        // Create figure
        let fig_args = match figsize {
            Some((width, height)) => {
                let size_tuple = PyTuple::new(py, &[width, height]);
                vec![("figsize", size_tuple.to_object(py))]
            },
            None => vec![],
        };
        
        let fig = plt.call_method("figure", (), Some(&fig_args.into_py_dict(py)))?;
        
        // Create scatter plot
        plt.call_method1("scatter", (spike_times, neuron_ids))?;
        
        // Set labels
        plt.call_method1("xlabel", (xlabel.unwrap_or("Time (s)"),))?;
        plt.call_method1("ylabel", (ylabel.unwrap_or("Neuron ID"),))?;
        plt.call_method1("title", (title.unwrap_or("Spike Raster Plot"),))?;
        
        // Grid
        plt.call_method("grid", (true,), None)?;
        
        // Save or show
        if let Some(path) = save_path {
            plt.call_method1("savefig", (path,))?;
        } else {
            plt.call_method0("show")?;
        }
        
        Ok(fig.to_object(py))
    })
}

#[pyfunction]
pub fn plot_membrane_potential(
    times: Vec<f64>,
    voltages: Vec<f32>,
    spike_times: Option<Vec<f64>>,
    title: Option<&str>,
    save_path: Option<&str>,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let plt = py.import("matplotlib.pyplot").map_err(|_| {
            PyImportError::new_err("matplotlib not available")
        })?;
        
        let fig = plt.call_method0("figure")?;
        
        // Plot membrane potential
        plt.call_method("plot", (times, voltages), Some(&[("label", "Membrane Potential")].into_py_dict(py)))?;
        
        // Add spike markers if provided
        if let Some(spike_times) = spike_times {
            for spike_time in spike_times {
                plt.call_method("axvline", (spike_time,), Some(&[
                    ("color", "red"),
                    ("linestyle", "--"),
                    ("alpha", 0.7),
                ].into_py_dict(py)))?;
            }
        }
        
        plt.call_method1("xlabel", ("Time (s)",))?;
        plt.call_method1("ylabel", ("Voltage (mV)",))?;
        plt.call_method1("title", (title.unwrap_or("Membrane Potential"),))?;
        plt.call_method0("legend")?;
        plt.call_method("grid", (true,), None)?;
        
        if let Some(path) = save_path {
            plt.call_method1("savefig", (path,))?;
        } else {
            plt.call_method0("show")?;
        }
        
        Ok(fig.to_object(py))
    })
}

#[pyfunction]
pub fn plot_weight_matrix(
    weights: Vec<Vec<f32>>,
    title: Option<&str>,
    colormap: Option<&str>,
    save_path: Option<&str>,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let plt = py.import("matplotlib.pyplot").map_err(|_| {
            PyImportError::new_err("matplotlib not available")
        })?;
        
        let fig = plt.call_method0("figure")?;
        
        // Create imshow plot
        let cmap = colormap.unwrap_or("viridis");
        let im = plt.call_method("imshow", (weights,), Some(&[
            ("cmap", cmap),
            ("aspect", "auto"),
        ].into_py_dict(py)))?;
        
        // Add colorbar
        plt.call_method1("colorbar", (im,))?;
        
        plt.call_method1("xlabel", ("Post-synaptic Neuron",))?;
        plt.call_method1("ylabel", ("Pre-synaptic Neuron",))?;
        plt.call_method1("title", (title.unwrap_or("Weight Matrix"),))?;
        
        if let Some(path) = save_path {
            plt.call_method1("savefig", (path,))?;
        } else {
            plt.call_method0("show")?;
        }
        
        Ok(fig.to_object(py))
    })
}

#[pyfunction]
pub fn plot_firing_rate(
    times: Vec<f64>,
    rates: Vec<f64>,
    title: Option<&str>,
    save_path: Option<&str>,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let plt = py.import("matplotlib.pyplot").map_err(|_| {
            PyImportError::new_err("matplotlib not available")
        })?;
        
        let fig = plt.call_method0("figure")?;
        
        plt.call_method("plot", (times, rates), None)?;
        plt.call_method1("xlabel", ("Time (s)",))?;
        plt.call_method1("ylabel", ("Firing Rate (Hz)",))?;
        plt.call_method1("title", (title.unwrap_or("Population Firing Rate"),))?;
        plt.call_method("grid", (true,), None)?;
        
        if let Some(path) = save_path {
            plt.call_method1("savefig", (path,))?;
        } else {
            plt.call_method0("show")?;
        }
        
        Ok(fig.to_object(py))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_raster_plot() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let spikes = PyList::new(py, &[
                PySpike::new(0, 0.001, Some(1.0)),
                PySpike::new(1, 0.002, Some(1.0)),
                PySpike::new(0, 0.003, Some(1.0)),
                PySpike::new(2, 0.004, Some(1.0)),
            ]);
            
            let raster = PyRasterPlot::new(spikes, None, None).unwrap();
            
            assert_eq!(raster.len(), 4);
            assert_eq!(raster.get_spike_times().len(), 4);
            assert_eq!(raster.get_neuron_ids().len(), 4);
            assert_eq!(raster.get_neuron_range(), (0, 2));
            assert!(raster.get_time_window().0 <= 0.001);
            assert!(raster.get_time_window().1 >= 0.004);
            
            // Test filtering
            let filtered = raster.filter_by_time(0.0015, 0.0035);
            assert_eq!(filtered.len(), 2); // Only spikes at 0.002 and 0.003
            
            let neuron_filtered = raster.filter_by_neurons(0, 1);
            assert_eq!(neuron_filtered.get_neuron_ids().iter().max().unwrap(), &1);
            
            // Test binned counts
            let (bin_centers, bin_counts) = raster.get_binned_counts(0.001).unwrap();
            assert!(!bin_centers.is_empty());
            assert_eq!(bin_centers.len(), bin_counts.len());
            assert_eq!(bin_counts.iter().sum::<u32>(), 4);
        });
    }
    
    #[test]
    fn test_membrane_trace() {
        let times = vec![0.0, 0.001, 0.002, 0.003, 0.004];
        let voltages = vec![-70.0, -65.0, -50.0, -70.0, -68.0];
        let spike_times = vec![0.0025];
        
        let trace = PyMembraneTrace::new(times.clone(), voltages.clone(), Some(spike_times.clone()), Some(42)).unwrap();
        
        assert_eq!(trace.get_times(), times);
        assert_eq!(trace.get_voltages(), voltages);
        assert_eq!(trace.get_spike_times(), spike_times);
        assert_eq!(trace.get_neuron_id(), 42);
        assert_eq!(trace.len(), 5);
        
        // Test filtering
        let filtered = trace.filter_by_time(0.001, 0.003);
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.get_spike_times().len(), 1);
        
        // Test voltage statistics
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let stats = trace.get_voltage_stats().unwrap();
            let stats_dict = stats.downcast::<PyDict>(py).unwrap();
            assert!(stats_dict.contains("mean").unwrap());
            assert!(stats_dict.contains("std").unwrap());
            assert!(stats_dict.contains("min").unwrap());
            assert!(stats_dict.contains("max").unwrap());
        });
        
        // Test invalid input
        assert!(PyMembraneTrace::new(vec![0.0], vec![1.0, 2.0], None, None).is_err());
    }
    
    #[test]
    fn test_weight_matrix() {
        let weights = vec![
            vec![0.1, 0.2, 0.0],
            vec![0.3, 0.0, 0.4],
            vec![0.0, 0.5, 0.1],
        ];
        
        let matrix = PyWeightMatrix::new(weights.clone(), None, None).unwrap();
        
        assert_eq!(matrix.get_weights(), weights);
        assert_eq!(matrix.get_shape(), (3, 3));
        assert_eq!(matrix.get_pre_neuron_ids(), vec![0, 1, 2]);
        assert_eq!(matrix.get_post_neuron_ids(), vec![0, 1, 2]);
        
        // Test weight statistics
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let stats = matrix.get_weight_stats().unwrap();
            let stats_dict = stats.downcast::<PyDict>(py).unwrap();
            assert!(stats_dict.contains("mean").unwrap());
            assert!(stats_dict.contains("sparsity").unwrap());
            assert!(stats_dict.contains("num_positive").unwrap());
            assert!(stats_dict.contains("num_zero").unwrap());
        });
        
        // Test weight distribution
        let (bin_centers, bin_counts) = matrix.get_weight_distribution(Some(10));
        assert!(!bin_centers.is_empty());
        assert_eq!(bin_centers.len(), bin_counts.len());
        assert_eq!(bin_counts.iter().sum::<u32>(), 9); // 3x3 matrix
        
        // Test filtering
        let filtered = matrix.filter_by_weight_range(0.1, 0.3);
        let filtered_weights = filtered.get_weights();
        for row in &filtered_weights {
            for &weight in row {
                if weight != 0.0 {
                    assert!(weight >= 0.1 && weight <= 0.3);
                }
            }
        }
        
        // Test invalid input
        let invalid_weights = vec![vec![1.0, 2.0], vec![3.0]]; // Inconsistent row lengths
        assert!(PyWeightMatrix::new(invalid_weights, None, None).is_err());
        
        let empty_weights: Vec<Vec<f32>> = vec![];
        assert!(PyWeightMatrix::new(empty_weights, None, None).is_err());
    }
    
    #[test]
    fn test_utility_functions() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let spikes = PyList::new(py, &[
                PySpike::new(0, 0.001, Some(1.0)),
                PySpike::new(1, 0.002, Some(1.0)),
            ]);
            
            let raster = create_raster_plot(spikes, None, None).unwrap();
            assert_eq!(raster.len(), 2);
            
            let times = vec![0.0, 0.001, 0.002];
            let voltages = vec![-70.0, -65.0, -50.0];
            let trace = create_membrane_trace(times, voltages, None, None).unwrap();
            assert_eq!(trace.len(), 3);
            
            let weights = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
            let matrix = create_weight_matrix(weights, None, None).unwrap();
            assert_eq!(matrix.get_shape(), (2, 2));
        });
    }
}