//! Python bindings for utility functions and data processing
//!
//! This module provides Python interfaces for common utility functions,
//! data conversion, validation, and helper methods for SHNN operations.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyBytes};
use pyo3::exceptions::{PyRuntimeError, PyValueError, PyIOError};

use shnn_core::{
    time::TimeStep,
    spike::{Spike, SpikeTime},
    math::{Statistics, Correlation, FFTProcessor, SignalProcessor},
};

use shnn_ffi::{
    utils::{
        validation::{validate_network_config, validate_spike_data},
        profiling::{ProfilerBuilder, PerformanceReport},
        serialization::{serialize_network_state, deserialize_network_state},
        memory::{MemoryPool, MemoryTracker},
    },
    NetworkConfig, SpikeData,
};

use crate::neuron::PySpike;
use crate::error_conversion::{core_error_to_py_err, ffi_error_to_py_err};

/// Python wrapper for statistics calculator
#[pyclass(name = "Statistics")]
pub struct PyStatistics {
    stats: Statistics,
}

#[pymethods]
impl PyStatistics {
    #[new]
    fn new() -> Self {
        Self {
            stats: Statistics::new(),
        }
    }
    
    /// Calculate mean of data
    fn mean(&self, data: Vec<f64>) -> PyResult<f64> {
        if data.is_empty() {
            return Err(PyValueError::new_err("Data cannot be empty"));
        }
        Ok(self.stats.mean(&data))
    }
    
    /// Calculate standard deviation
    fn std(&self, data: Vec<f64>) -> PyResult<f64> {
        if data.is_empty() {
            return Err(PyValueError::new_err("Data cannot be empty"));
        }
        Ok(self.stats.std(&data))
    }
    
    /// Calculate variance
    fn variance(&self, data: Vec<f64>) -> PyResult<f64> {
        if data.is_empty() {
            return Err(PyValueError::new_err("Data cannot be empty"));
        }
        Ok(self.stats.variance(&data))
    }
    
    /// Calculate median
    fn median(&self, data: Vec<f64>) -> PyResult<f64> {
        if data.is_empty() {
            return Err(PyValueError::new_err("Data cannot be empty"));
        }
        Ok(self.stats.median(&data))
    }
    
    /// Calculate percentile
    fn percentile(&self, data: Vec<f64>, p: f64) -> PyResult<f64> {
        if data.is_empty() {
            return Err(PyValueError::new_err("Data cannot be empty"));
        }
        if p < 0.0 || p > 100.0 {
            return Err(PyValueError::new_err("Percentile must be between 0 and 100"));
        }
        Ok(self.stats.percentile(&data, p))
    }
    
    /// Calculate summary statistics
    fn summary(&self, data: Vec<f64>) -> PyResult<PyObject> {
        if data.is_empty() {
            return Err(PyValueError::new_err("Data cannot be empty"));
        }
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("count", data.len())?;
            dict.set_item("mean", self.stats.mean(&data))?;
            dict.set_item("std", self.stats.std(&data))?;
            dict.set_item("min", self.stats.min(&data))?;
            dict.set_item("max", self.stats.max(&data))?;
            dict.set_item("median", self.stats.median(&data))?;
            dict.set_item("q25", self.stats.percentile(&data, 25.0))?;
            dict.set_item("q75", self.stats.percentile(&data, 75.0))?;
            Ok(dict.to_object(py))
        })
    }
    
    /// Calculate histogram
    fn histogram(&self, data: Vec<f64>, bins: Option<u32>) -> PyResult<(Vec<f64>, Vec<u32>)> {
        if data.is_empty() {
            return Err(PyValueError::new_err("Data cannot be empty"));
        }
        
        let num_bins = bins.unwrap_or(20);
        let (bin_edges, counts) = self.stats.histogram(&data, num_bins)
            .map_err(core_error_to_py_err)?;
        
        Ok((bin_edges, counts))
    }
    
    fn __repr__(&self) -> String {
        "Statistics()".to_string()
    }
}

/// Python wrapper for correlation analysis
#[pyclass(name = "Correlation")]
pub struct PyCorrelation {
    correlation: Correlation,
}

#[pymethods]
impl PyCorrelation {
    #[new]
    fn new() -> Self {
        Self {
            correlation: Correlation::new(),
        }
    }
    
    /// Calculate Pearson correlation coefficient
    fn pearson(&self, x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
        if x.len() != y.len() {
            return Err(PyValueError::new_err("Arrays must have same length"));
        }
        if x.is_empty() {
            return Err(PyValueError::new_err("Arrays cannot be empty"));
        }
        
        self.correlation.pearson(&x, &y)
            .map_err(core_error_to_py_err)
    }
    
    /// Calculate Spearman correlation coefficient
    fn spearman(&self, x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
        if x.len() != y.len() {
            return Err(PyValueError::new_err("Arrays must have same length"));
        }
        if x.is_empty() {
            return Err(PyValueError::new_err("Arrays cannot be empty"));
        }
        
        self.correlation.spearman(&x, &y)
            .map_err(core_error_to_py_err)
    }
    
    /// Calculate cross-correlation
    fn cross_correlation(&self, x: Vec<f64>, y: Vec<f64>, max_lag: Option<i32>) -> PyResult<Vec<f64>> {
        if x.is_empty() || y.is_empty() {
            return Err(PyValueError::new_err("Arrays cannot be empty"));
        }
        
        let lag = max_lag.unwrap_or(x.len().min(y.len()) as i32 / 4);
        self.correlation.cross_correlation(&x, &y, lag)
            .map_err(core_error_to_py_err)
    }
    
    /// Calculate auto-correlation
    fn auto_correlation(&self, x: Vec<f64>, max_lag: Option<i32>) -> PyResult<Vec<f64>> {
        if x.is_empty() {
            return Err(PyValueError::new_err("Array cannot be empty"));
        }
        
        let lag = max_lag.unwrap_or(x.len() as i32 / 4);
        self.correlation.auto_correlation(&x, lag)
            .map_err(core_error_to_py_err)
    }
    
    /// Calculate correlation matrix
    fn correlation_matrix(&self, data: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        if data.is_empty() {
            return Err(PyValueError::new_err("Data cannot be empty"));
        }
        
        // Check all rows have same length
        let expected_len = data[0].len();
        for row in &data {
            if row.len() != expected_len {
                return Err(PyValueError::new_err("All rows must have same length"));
            }
        }
        
        self.correlation.correlation_matrix(&data)
            .map_err(core_error_to_py_err)
    }
    
    fn __repr__(&self) -> String {
        "Correlation()".to_string()
    }
}

/// Python wrapper for FFT processor
#[pyclass(name = "FFTProcessor")]
pub struct PyFFTProcessor {
    fft: FFTProcessor,
}

#[pymethods]
impl PyFFTProcessor {
    #[new]
    fn new(size: usize) -> PyResult<Self> {
        if size == 0 || (size & (size - 1)) != 0 {
            return Err(PyValueError::new_err("FFT size must be a power of 2"));
        }
        
        let fft = FFTProcessor::new(size)
            .map_err(core_error_to_py_err)?;
        
        Ok(Self { fft })
    }
    
    /// Calculate FFT of real signal
    fn fft(&self, signal: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if signal.len() != self.fft.size() {
            return Err(PyValueError::new_err(format!(
                "Signal length {} must match FFT size {}",
                signal.len(),
                self.fft.size()
            )));
        }
        
        let (real, imag) = self.fft.fft(&signal)
            .map_err(core_error_to_py_err)?;
        
        Ok((real, imag))
    }
    
    /// Calculate inverse FFT
    fn ifft(&self, real: Vec<f64>, imag: Vec<f64>) -> PyResult<Vec<f64>> {
        if real.len() != self.fft.size() || imag.len() != self.fft.size() {
            return Err(PyValueError::new_err("Real and imaginary parts must match FFT size"));
        }
        
        self.fft.ifft(&real, &imag)
            .map_err(core_error_to_py_err)
    }
    
    /// Calculate power spectral density
    fn power_spectrum(&self, signal: Vec<f64>) -> PyResult<Vec<f64>> {
        if signal.len() != self.fft.size() {
            return Err(PyValueError::new_err("Signal length must match FFT size"));
        }
        
        self.fft.power_spectrum(&signal)
            .map_err(core_error_to_py_err)
    }
    
    /// Calculate magnitude spectrum
    fn magnitude_spectrum(&self, signal: Vec<f64>) -> PyResult<Vec<f64>> {
        if signal.len() != self.fft.size() {
            return Err(PyValueError::new_err("Signal length must match FFT size"));
        }
        
        self.fft.magnitude_spectrum(&signal)
            .map_err(core_error_to_py_err)
    }
    
    /// Calculate phase spectrum
    fn phase_spectrum(&self, signal: Vec<f64>) -> PyResult<Vec<f64>> {
        if signal.len() != self.fft.size() {
            return Err(PyValueError::new_err("Signal length must match FFT size"));
        }
        
        self.fft.phase_spectrum(&signal)
            .map_err(core_error_to_py_err)
    }
    
    /// Get frequency bins
    fn frequency_bins(&self, sample_rate: f64) -> Vec<f64> {
        self.fft.frequency_bins(sample_rate)
    }
    
    /// Get FFT size
    fn size(&self) -> usize {
        self.fft.size()
    }
    
    fn __repr__(&self) -> String {
        format!("FFTProcessor(size={})", self.fft.size())
    }
}

/// Python wrapper for signal processor
#[pyclass(name = "SignalProcessor")]
pub struct PySignalProcessor {
    processor: SignalProcessor,
}

#[pymethods]
impl PySignalProcessor {
    #[new]
    fn new() -> Self {
        Self {
            processor: SignalProcessor::new(),
        }
    }
    
    /// Apply low-pass filter
    fn low_pass_filter(&self, signal: Vec<f64>, cutoff: f64, sample_rate: f64) -> PyResult<Vec<f64>> {
        if signal.is_empty() {
            return Err(PyValueError::new_err("Signal cannot be empty"));
        }
        if cutoff <= 0.0 || cutoff >= sample_rate / 2.0 {
            return Err(PyValueError::new_err("Cutoff frequency must be between 0 and Nyquist frequency"));
        }
        
        self.processor.low_pass_filter(&signal, cutoff, sample_rate)
            .map_err(core_error_to_py_err)
    }
    
    /// Apply high-pass filter
    fn high_pass_filter(&self, signal: Vec<f64>, cutoff: f64, sample_rate: f64) -> PyResult<Vec<f64>> {
        if signal.is_empty() {
            return Err(PyValueError::new_err("Signal cannot be empty"));
        }
        if cutoff <= 0.0 || cutoff >= sample_rate / 2.0 {
            return Err(PyValueError::new_err("Cutoff frequency must be between 0 and Nyquist frequency"));
        }
        
        self.processor.high_pass_filter(&signal, cutoff, sample_rate)
            .map_err(core_error_to_py_err)
    }
    
    /// Apply band-pass filter
    fn band_pass_filter(&self, signal: Vec<f64>, low_cutoff: f64, high_cutoff: f64, sample_rate: f64) -> PyResult<Vec<f64>> {
        if signal.is_empty() {
            return Err(PyValueError::new_err("Signal cannot be empty"));
        }
        if low_cutoff >= high_cutoff {
            return Err(PyValueError::new_err("Low cutoff must be less than high cutoff"));
        }
        if low_cutoff <= 0.0 || high_cutoff >= sample_rate / 2.0 {
            return Err(PyValueError::new_err("Cutoff frequencies must be between 0 and Nyquist frequency"));
        }
        
        self.processor.band_pass_filter(&signal, low_cutoff, high_cutoff, sample_rate)
            .map_err(core_error_to_py_err)
    }
    
    /// Resample signal
    fn resample(&self, signal: Vec<f64>, original_rate: f64, target_rate: f64) -> PyResult<Vec<f64>> {
        if signal.is_empty() {
            return Err(PyValueError::new_err("Signal cannot be empty"));
        }
        if original_rate <= 0.0 || target_rate <= 0.0 {
            return Err(PyValueError::new_err("Sample rates must be positive"));
        }
        
        self.processor.resample(&signal, original_rate, target_rate)
            .map_err(core_error_to_py_err)
    }
    
    /// Calculate spectrogram
    fn spectrogram(&self, signal: Vec<f64>, window_size: usize, hop_size: usize, sample_rate: f64) -> PyResult<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> {
        if signal.is_empty() {
            return Err(PyValueError::new_err("Signal cannot be empty"));
        }
        if window_size == 0 || hop_size == 0 {
            return Err(PyValueError::new_err("Window size and hop size must be positive"));
        }
        if window_size > signal.len() {
            return Err(PyValueError::new_err("Window size cannot be larger than signal length"));
        }
        
        let (spec, freqs, times) = self.processor.spectrogram(&signal, window_size, hop_size, sample_rate)
            .map_err(core_error_to_py_err)?;
        
        Ok((spec, freqs, times))
    }
    
    fn __repr__(&self) -> String {
        "SignalProcessor()".to_string()
    }
}

/// Python wrapper for performance profiler
#[pyclass(name = "Profiler")]
pub struct PyProfiler {
    profiler: shnn_ffi::utils::profiling::Profiler,
}

#[pymethods]
impl PyProfiler {
    #[new]
    fn new(name: String) -> Self {
        let profiler = ProfilerBuilder::new()
            .with_name(name)
            .build();
        
        Self { profiler }
    }
    
    /// Start profiling a section
    fn start_section(&mut self, section_name: String) {
        self.profiler.start_section(section_name);
    }
    
    /// End profiling current section
    fn end_section(&mut self) {
        self.profiler.end_section();
    }
    
    /// Record custom metric
    fn record_metric(&mut self, name: String, value: f64) {
        self.profiler.record_metric(name, value);
    }
    
    /// Get performance report
    fn get_report(&self) -> PyPerformanceReport {
        let report = self.profiler.get_report();
        PyPerformanceReport { report }
    }
    
    /// Reset profiler
    fn reset(&mut self) {
        self.profiler.reset();
    }
    
    /// Export report to JSON
    fn export_json(&self) -> PyResult<String> {
        self.profiler.export_json()
            .map_err(ffi_error_to_py_err)
    }
    
    /// Save report to file
    fn save_report(&self, filename: String) -> PyResult<()> {
        self.profiler.save_report(&filename)
            .map_err(ffi_error_to_py_err)
    }
    
    fn __repr__(&self) -> String {
        format!("Profiler('{}')", self.profiler.get_name())
    }
}

/// Python wrapper for performance report
#[pyclass(name = "PerformanceReport")]
pub struct PyPerformanceReport {
    report: PerformanceReport,
}

#[pymethods]
impl PyPerformanceReport {
    /// Get total execution time
    fn get_total_time(&self) -> f64 {
        self.report.get_total_time()
    }
    
    /// Get section times
    fn get_section_times(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for (section, time) in self.report.get_section_times() {
                dict.set_item(section, time)?;
            }
            Ok(dict.to_object(py))
        })
    }
    
    /// Get custom metrics
    fn get_metrics(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for (metric, value) in self.report.get_metrics() {
                dict.set_item(metric, value)?;
            }
            Ok(dict.to_object(py))
        })
    }
    
    /// Get summary as dictionary
    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("total_time", self.get_total_time())?;
            dict.set_item("section_times", self.get_section_times()?)?;
            dict.set_item("metrics", self.get_metrics()?)?;
            Ok(dict.to_object(py))
        })
    }
    
    fn __repr__(&self) -> String {
        format!("PerformanceReport(total_time={:.3}s)", self.get_total_time())
    }
}

/// Python wrapper for memory tracker
#[pyclass(name = "MemoryTracker")]
pub struct PyMemoryTracker {
    tracker: MemoryTracker,
}

#[pymethods]
impl PyMemoryTracker {
    #[new]
    fn new() -> Self {
        Self {
            tracker: MemoryTracker::new(),
        }
    }
    
    /// Start tracking memory usage
    fn start_tracking(&mut self) {
        self.tracker.start_tracking();
    }
    
    /// Stop tracking memory usage
    fn stop_tracking(&mut self) {
        self.tracker.stop_tracking();
    }
    
    /// Get current memory usage in bytes
    fn get_current_usage(&self) -> u64 {
        self.tracker.get_current_usage()
    }
    
    /// Get peak memory usage in bytes
    fn get_peak_usage(&self) -> u64 {
        self.tracker.get_peak_usage()
    }
    
    /// Get memory usage statistics
    fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("current_bytes", self.tracker.get_current_usage())?;
            dict.set_item("peak_bytes", self.tracker.get_peak_usage())?;
            dict.set_item("current_mb", self.tracker.get_current_usage() as f64 / (1024.0 * 1024.0))?;
            dict.set_item("peak_mb", self.tracker.get_peak_usage() as f64 / (1024.0 * 1024.0))?;
            dict.set_item("is_tracking", self.tracker.is_tracking())?;
            Ok(dict.to_object(py))
        })
    }
    
    /// Reset tracking
    fn reset(&mut self) {
        self.tracker.reset();
    }
    
    fn __repr__(&self) -> String {
        format!(
            "MemoryTracker(current={:.1}MB, peak={:.1}MB)",
            self.tracker.get_current_usage() as f64 / (1024.0 * 1024.0),
            self.tracker.get_peak_usage() as f64 / (1024.0 * 1024.0)
        )
    }
}

/// Validation utility functions
#[pyfunction]
pub fn validate_spike_train(spikes: &PyList) -> PyResult<bool> {
    let mut spike_vec = Vec::new();
    for spike_obj in spikes.iter() {
        let py_spike: PySpike = spike_obj.extract()?;
        spike_vec.push(py_spike.spike);
    }
    
    let spike_data = SpikeData {
        spikes: spike_vec.into_iter().map(|spike| shnn_ffi::types::SpikeEvent {
            neuron_id: spike.neuron_id,
            timestamp: spike.time.as_secs_f64(),
            amplitude: spike.amplitude,
        }).collect(),
        start_time: 0.0,
        end_time: 1.0,
    };
    
    validate_spike_data(&spike_data)
        .map_err(ffi_error_to_py_err)
        .map(|_| true)
}

#[pyfunction]
pub fn validate_network_parameters(config_dict: &PyDict) -> PyResult<bool> {
    let config = crate::type_conversion::py_dict_to_network_config(config_dict)?;
    
    validate_network_config(&config)
        .map_err(ffi_error_to_py_err)
        .map(|_| true)
}

/// Serialization utility functions
#[pyfunction]
pub fn serialize_network_to_bytes(network_dict: &PyDict) -> PyResult<PyObject> {
    let serialized = serde_json::to_vec(network_dict)
        .map_err(|e| PyRuntimeError::new_err(format!("Serialization failed: {}", e)))?;
    
    Python::with_gil(|py| {
        Ok(PyBytes::new(py, &serialized).to_object(py))
    })
}

#[pyfunction]
pub fn deserialize_network_from_bytes(data: &PyBytes) -> PyResult<PyObject> {
    let bytes = data.as_bytes();
    let network_dict: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("Deserialization failed: {}", e)))?;
    
    Python::with_gil(|py| {
        network_dict.to_object(py)
    })
}

/// Data conversion utilities
#[pyfunction]
pub fn spike_times_to_binary_array(
    spikes: &PyList,
    num_neurons: u32,
    bin_size: f64,
    duration: f64,
) -> PyResult<Vec<Vec<u8>>> {
    let mut spike_vec = Vec::new();
    for spike_obj in spikes.iter() {
        let py_spike: PySpike = spike_obj.extract()?;
        spike_vec.push(py_spike.spike);
    }
    
    let num_bins = (duration / bin_size).ceil() as usize;
    let mut binary_array = vec![vec![0u8; num_bins]; num_neurons as usize];
    
    for spike in spike_vec {
        let neuron_id = spike.neuron_id as usize;
        let bin_idx = (spike.time.as_secs_f64() / bin_size).floor() as usize;
        
        if neuron_id < num_neurons as usize && bin_idx < num_bins {
            binary_array[neuron_id][bin_idx] = 1;
        }
    }
    
    Ok(binary_array)
}

#[pyfunction]
pub fn binary_array_to_spike_times(
    binary_array: Vec<Vec<u8>>,
    bin_size: f64,
) -> Vec<PySpike> {
    let mut spikes = Vec::new();
    
    for (neuron_id, neuron_data) in binary_array.iter().enumerate() {
        for (bin_idx, &value) in neuron_data.iter().enumerate() {
            if value > 0 {
                let time = bin_idx as f64 * bin_size;
                spikes.push(PySpike::new(neuron_id as u32, time, Some(1.0)));
            }
        }
    }
    
    // Sort by time
    spikes.sort_by(|a, b| a.time().partial_cmp(&b.time()).unwrap());
    
    spikes
}

/// Time series utilities
#[pyfunction]
pub fn interpolate_missing_values(
    data: Vec<f64>,
    method: Option<&str>,
) -> PyResult<Vec<f64>> {
    if data.is_empty() {
        return Ok(data);
    }
    
    let interpolation_method = method.unwrap_or("linear");
    
    match interpolation_method {
        "linear" => {
            let mut result = data.clone();
            let mut last_valid_idx = None;
            
            for i in 0..result.len() {
                if result[i].is_finite() {
                    if let Some(start_idx) = last_valid_idx {
                        if i > start_idx + 1 {
                            // Interpolate between start_idx and i
                            let start_val = result[start_idx];
                            let end_val = result[i];
                            let num_points = i - start_idx - 1;
                            
                            for j in 1..=num_points {
                                let t = j as f64 / (num_points + 1) as f64;
                                result[start_idx + j] = start_val + t * (end_val - start_val);
                            }
                        }
                    }
                    last_valid_idx = Some(i);
                }
            }
            
            Ok(result)
        },
        "forward_fill" => {
            let mut result = data.clone();
            let mut last_valid = None;
            
            for i in 0..result.len() {
                if result[i].is_finite() {
                    last_valid = Some(result[i]);
                } else if let Some(val) = last_valid {
                    result[i] = val;
                }
            }
            
            Ok(result)
        },
        "backward_fill" => {
            let mut result = data.clone();
            let mut next_valid = None;
            
            for i in (0..result.len()).rev() {
                if result[i].is_finite() {
                    next_valid = Some(result[i]);
                } else if let Some(val) = next_valid {
                    result[i] = val;
                }
            }
            
            Ok(result)
        },
        _ => Err(PyValueError::new_err(format!("Unknown interpolation method: {}", interpolation_method))),
    }
}

/// Generate test data
#[pyfunction]
pub fn generate_test_spike_data(
    num_neurons: u32,
    duration: f64,
    base_rate: f32,
    noise_level: Option<f32>,
    seed: Option<u64>,
) -> PyResult<Vec<PySpike>> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = match seed {
        Some(seed) => ChaCha8Rng::seed_from_u64(seed),
        None => ChaCha8Rng::from_entropy(),
    };
    
    let noise = noise_level.unwrap_or(0.1);
    let mut spikes = Vec::new();
    
    for neuron_id in 0..num_neurons {
        // Add some variation per neuron
        let neuron_rate = base_rate * (1.0 + (rng.gen::<f32>() - 0.5) * noise);
        
        let mut time = 0.0;
        while time < duration {
            let interval = -((1.0 - rng.gen::<f32>()).ln()) / neuron_rate;
            time += interval;
            
            if time < duration {
                let amplitude = 1.0 + (rng.gen::<f32>() - 0.5) * noise * 0.5;
                spikes.push(PySpike::new(neuron_id, time, Some(amplitude)));
            }
        }
    }
    
    // Sort by time
    spikes.sort_by(|a, b| a.time().partial_cmp(&b.time()).unwrap());
    
    Ok(spikes)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_statistics() {
        let stats = PyStatistics::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_eq!(stats.mean(data.clone()).unwrap(), 3.0);
        assert_eq!(stats.median(data.clone()).unwrap(), 3.0);
        assert!(stats.std(data.clone()).unwrap() > 0.0);
        assert!(stats.variance(data.clone()).unwrap() > 0.0);
        
        // Test percentiles
        assert_eq!(stats.percentile(data.clone(), 0.0).unwrap(), 1.0);
        assert_eq!(stats.percentile(data.clone(), 100.0).unwrap(), 5.0);
        assert_eq!(stats.percentile(data.clone(), 50.0).unwrap(), 3.0);
        
        // Test invalid inputs
        assert!(stats.mean(vec![]).is_err());
        assert!(stats.percentile(data, -1.0).is_err());
        assert!(stats.percentile(vec![1.0], 101.0).is_err());
    }
    
    #[test]
    fn test_correlation() {
        let corr = PyCorrelation::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation
        
        let pearson = corr.pearson(x.clone(), y.clone()).unwrap();
        assert!((pearson - 1.0).abs() < 1e-10);
        
        let spearman = corr.spearman(x.clone(), y.clone()).unwrap();
        assert!((spearman - 1.0).abs() < 1e-10);
        
        // Test auto-correlation
        let auto_corr = corr.auto_correlation(x.clone(), Some(2)).unwrap();
        assert!(!auto_corr.is_empty());
        
        // Test invalid inputs
        assert!(corr.pearson(vec![1.0], vec![1.0, 2.0]).is_err());
        assert!(corr.spearman(vec![], vec![]).is_err());
    }
    
    #[test]
    fn test_fft_processor() {
        let fft = PyFFTProcessor::new(8).unwrap();
        assert_eq!(fft.size(), 8);
        
        // Test FFT with simple signal
        let signal = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let (real, imag) = fft.fft(signal.clone()).unwrap();
        assert_eq!(real.len(), 8);
        assert_eq!(imag.len(), 8);
        
        // Test inverse FFT
        let reconstructed = fft.ifft(real, imag).unwrap();
        for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 1e-10);
        }
        
        // Test power spectrum
        let power = fft.power_spectrum(signal.clone()).unwrap();
        assert_eq!(power.len(), 4); // Half of FFT size for real signals
        
        // Test frequency bins
        let freqs = fft.frequency_bins(1000.0);
        assert_eq!(freqs.len(), 4);
        
        // Test invalid inputs
        assert!(PyFFTProcessor::new(7).is_err()); // Not power of 2
        assert!(fft.fft(vec![1.0; 4]).is_err()); // Wrong size
    }
    
    #[test]
    fn test_signal_processor() {
        let processor = PySignalProcessor::new();
        let signal = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        
        // Test filters
        let filtered = processor.low_pass_filter(signal.clone(), 100.0, 1000.0).unwrap();
        assert_eq!(filtered.len(), signal.len());
        
        let hp_filtered = processor.high_pass_filter(signal.clone(), 10.0, 1000.0).unwrap();
        assert_eq!(hp_filtered.len(), signal.len());
        
        let bp_filtered = processor.band_pass_filter(signal.clone(), 10.0, 100.0, 1000.0).unwrap();
        assert_eq!(bp_filtered.len(), signal.len());
        
        // Test resampling
        let resampled = processor.resample(signal.clone(), 1000.0, 500.0).unwrap();
        assert_eq!(resampled.len(), signal.len() / 2);
        
        // Test invalid inputs
        assert!(processor.low_pass_filter(vec![], 100.0, 1000.0).is_err());
        assert!(processor.band_pass_filter(signal, 200.0, 100.0, 1000.0).is_err()); // low > high
    }
    
    #[test]
    fn test_memory_tracker() {
        let mut tracker = PyMemoryTracker::new();
        
        tracker.start_tracking();
        let initial_usage = tracker.get_current_usage();
        
        // Allocate some memory (simplified)
        let _data = vec![0u8; 1024 * 1024]; // 1MB
        
        tracker.stop_tracking();
        let peak_usage = tracker.get_peak_usage();
        
        assert!(peak_usage >= initial_usage);
        
        tracker.reset();
        // After reset, tracking should be stopped
    }
    
    #[test]
    fn test_data_conversion() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Test spike to binary array conversion
            let spikes = PyList::new(py, &[
                PySpike::new(0, 0.001, Some(1.0)),
                PySpike::new(1, 0.002, Some(1.0)),
                PySpike::new(0, 0.003, Some(1.0)),
            ]);
            
            let binary_array = spike_times_to_binary_array(spikes, 2, 0.001, 0.005).unwrap();
            assert_eq!(binary_array.len(), 2); // 2 neurons
            assert_eq!(binary_array[0].len(), 5); // 5 time bins
            
            // Check spike locations
            assert_eq!(binary_array[0][1], 1); // Neuron 0 at time 0.001
            assert_eq!(binary_array[1][2], 1); // Neuron 1 at time 0.002
            assert_eq!(binary_array[0][3], 1); // Neuron 0 at time 0.003
            
            // Test reverse conversion
            let reconstructed_spikes = binary_array_to_spike_times(binary_array, 0.001);
            assert_eq!(reconstructed_spikes.len(), 3);
        });
    }
    
    #[test]
    fn test_interpolation() {
        let data = vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0];
        
        // Test linear interpolation
        let interpolated = interpolate_missing_values(data.clone(), Some("linear")).unwrap();
        assert_eq!(interpolated[1], 2.0);
        assert_eq!(interpolated[3], 4.0);
        
        // Test forward fill
        let ff_interpolated = interpolate_missing_values(data.clone(), Some("forward_fill")).unwrap();
        assert_eq!(ff_interpolated[1], 1.0);
        assert_eq!(ff_interpolated[3], 3.0);
        
        // Test backward fill
        let bf_interpolated = interpolate_missing_values(data, Some("backward_fill")).unwrap();
        assert_eq!(bf_interpolated[1], 3.0);
        assert_eq!(bf_interpolated[3], 5.0);
        
        // Test invalid method
        assert!(interpolate_missing_values(vec![1.0], Some("invalid")).is_err());
    }
    
    #[test]
    fn test_test_data_generation() {
        let spikes = generate_test_spike_data(10, 0.1, 100.0, Some(0.1), Some(42)).unwrap();
        assert!(!spikes.is_empty());
        
        // Check all spikes are within bounds
        for spike in &spikes {
            assert!(spike.neuron_id() < 10);
            assert!(spike.time() >= 0.0 && spike.time() <= 0.1);
            assert!(spike.amplitude() > 0.0);
        }
        
        // Check spikes are sorted by time
        for i in 1..spikes.len() {
            assert!(spikes[i-1].time() <= spikes[i].time());
        }
    }
}