//! Python bindings for spike operations and analysis
//!
//! This module provides Python interfaces for spike data manipulation,
//! analysis, and visualization of spiking neural network activity.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::{PyRuntimeError, PyValueError, PyIndexError};

use shnn_core::{
    spike::{Spike, SpikeTime, SpikeBuffer, SpikePattern},
    time::TimeStep,
    encoding::{SpikeEncoder, PoissonEncoder, TemporalEncoder, RateEncoder},
};

use crate::neuron::PySpike;
use crate::error_conversion::core_error_to_py_err;

/// Python wrapper for spike buffer
#[pyclass(name = "SpikeBuffer")]
pub struct PySpikeBuffer {
    buffer: SpikeBuffer,
}

#[pymethods]
impl PySpikeBuffer {
    #[new]
    fn new(capacity: Option<usize>) -> Self {
        Self {
            buffer: SpikeBuffer::new(capacity.unwrap_or(10000)),
        }
    }
    
    /// Add a spike to the buffer
    fn add_spike(&mut self, spike: PySpike) {
        self.buffer.add_spike(spike.spike);
    }
    
    /// Add multiple spikes to the buffer
    fn add_spikes(&mut self, spikes: &PyList) -> PyResult<()> {
        for spike_obj in spikes.iter() {
            let spike: PySpike = spike_obj.extract()?;
            self.buffer.add_spike(spike.spike);
        }
        Ok(())
    }
    
    /// Get spikes in time window
    fn get_spikes_in_window(&self, start_time: f64, end_time: f64) -> PyResult<Vec<PySpike>> {
        let start = SpikeTime::from_secs_f64(start_time);
        let end = SpikeTime::from_secs_f64(end_time);
        
        let spikes = self.buffer.get_spikes_in_window(start, end)
            .map_err(core_error_to_py_err)?;
        
        Ok(spikes.into_iter()
            .map(|spike| PySpike::from_spike(spike.clone()))
            .collect())
    }
    
    /// Get spikes for specific neuron
    fn get_spikes_for_neuron(&self, neuron_id: u32) -> Vec<PySpike> {
        self.buffer.get_spikes_for_neuron(neuron_id)
            .into_iter()
            .map(|spike| PySpike::from_spike(spike.clone()))
            .collect()
    }
    
    /// Get all spikes
    fn get_all_spikes(&self) -> Vec<PySpike> {
        self.buffer.get_all_spikes()
            .into_iter()
            .map(|spike| PySpike::from_spike(spike.clone()))
            .collect()
    }
    
    /// Clear the buffer
    fn clear(&mut self) {
        self.buffer.clear();
    }
    
    /// Get number of spikes
    fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Check if buffer is empty
    fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    
    /// Get buffer capacity
    fn capacity(&self) -> usize {
        self.buffer.capacity()
    }
    
    /// Get time range of spikes
    fn get_time_range(&self) -> Option<(f64, f64)> {
        self.buffer.get_time_range()
            .map(|(start, end)| (start.as_secs_f64(), end.as_secs_f64()))
    }
    
    /// Get spike count per neuron
    fn get_spike_counts(&self) -> PyResult<PyObject> {
        let counts = self.buffer.get_spike_counts();
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for (neuron_id, count) in counts {
                dict.set_item(neuron_id, count)?;
            }
            Ok(dict.to_object(py))
        })
    }
    
    /// Get firing rates per neuron
    fn get_firing_rates(&self, time_window: Option<f64>) -> PyResult<PyObject> {
        let window = time_window.map(TimeStep::from_secs_f64);
        let rates = self.buffer.get_firing_rates(window)
            .map_err(core_error_to_py_err)?;
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for (neuron_id, rate) in rates {
                dict.set_item(neuron_id, rate)?;
            }
            Ok(dict.to_object(py))
        })
    }
    
    /// Get inter-spike intervals for a neuron
    fn get_isi(&self, neuron_id: u32) -> PyResult<Vec<f64>> {
        let intervals = self.buffer.get_inter_spike_intervals(neuron_id)
            .map_err(core_error_to_py_err)?;
        
        Ok(intervals.into_iter()
            .map(|interval| interval.as_secs_f64())
            .collect())
    }
    
    /// Calculate spike train correlation
    fn correlation(&self, neuron_id1: u32, neuron_id2: u32, bin_size: f64) -> PyResult<f64> {
        let bin_duration = TimeStep::from_secs_f64(bin_size);
        self.buffer.calculate_correlation(neuron_id1, neuron_id2, bin_duration)
            .map_err(core_error_to_py_err)
    }
    
    /// Sort spikes by time
    fn sort_by_time(&mut self) {
        self.buffer.sort_by_time();
    }
    
    /// Remove spikes before time
    fn remove_before(&mut self, time: f64) {
        let cutoff_time = SpikeTime::from_secs_f64(time);
        self.buffer.remove_before(cutoff_time);
    }
    
    /// Get spikes as list of tuples
    fn to_tuples(&self) -> Vec<(u32, f64, f32)> {
        self.buffer.get_all_spikes()
            .into_iter()
            .map(|spike| (spike.neuron_id, spike.time.as_secs_f64(), spike.amplitude))
            .collect()
    }
    
    /// Create from list of tuples
    #[classmethod]
    fn from_tuples(_cls: &PyType, spike_tuples: &PyList, capacity: Option<usize>) -> PyResult<Self> {
        let mut buffer = Self::new(capacity);
        
        for tuple_obj in spike_tuples.iter() {
            let tuple = tuple_obj.downcast::<PyTuple>()?;
            if tuple.len() < 2 || tuple.len() > 3 {
                return Err(PyValueError::new_err("Spike tuple must be (neuron_id, time) or (neuron_id, time, amplitude)"));
            }
            
            let neuron_id: u32 = tuple.get_item(0)?.extract()?;
            let time: f64 = tuple.get_item(1)?.extract()?;
            let amplitude: f32 = if tuple.len() >= 3 {
                tuple.get_item(2)?.extract()?
            } else {
                1.0
            };
            
            let spike = PySpike::new(neuron_id, time, Some(amplitude));
            buffer.add_spike(spike);
        }
        
        Ok(buffer)
    }
    
    fn __len__(&self) -> usize {
        self.len()
    }
    
    fn __repr__(&self) -> String {
        format!("SpikeBuffer(spikes={}, capacity={})", self.len(), self.capacity())
    }
}

/// Python wrapper for spike pattern
#[pyclass(name = "SpikePattern")]
pub struct PySpikePattern {
    pattern: SpikePattern,
}

#[pymethods]
impl PySpikePattern {
    #[new]
    fn new(name: String) -> Self {
        Self {
            pattern: SpikePattern::new(name),
        }
    }
    
    /// Add spike to pattern
    fn add_spike(&mut self, neuron_id: u32, time: f64, amplitude: Option<f32>) {
        let spike = Spike {
            neuron_id,
            time: SpikeTime::from_secs_f64(time),
            amplitude: amplitude.unwrap_or(1.0),
            payload: Vec::new(),
        };
        self.pattern.add_spike(spike);
    }
    
    /// Get pattern name
    fn get_name(&self) -> &str {
        self.pattern.get_name()
    }
    
    /// Set pattern name
    fn set_name(&mut self, name: String) {
        self.pattern.set_name(name);
    }
    
    /// Get duration of pattern
    fn get_duration(&self) -> f64 {
        self.pattern.get_duration().as_secs_f64()
    }
    
    /// Get spikes in pattern
    fn get_spikes(&self) -> Vec<PySpike> {
        self.pattern.get_spikes()
            .into_iter()
            .map(|spike| PySpike::from_spike(spike.clone()))
            .collect()
    }
    
    /// Repeat pattern
    fn repeat(&self, num_repeats: u32, interval: f64) -> Self {
        let interval_time = TimeStep::from_secs_f64(interval);
        let repeated = self.pattern.repeat(num_repeats, interval_time);
        Self { pattern: repeated }
    }
    
    /// Scale pattern timing
    fn scale_time(&self, factor: f64) -> Self {
        let scaled = self.pattern.scale_time(factor);
        Self { pattern: scaled }
    }
    
    /// Shift pattern in time
    fn shift_time(&self, offset: f64) -> Self {
        let offset_time = TimeStep::from_secs_f64(offset);
        let shifted = self.pattern.shift_time(offset_time);
        Self { pattern: shifted }
    }
    
    /// Merge with another pattern
    fn merge(&self, other: &Self) -> Self {
        let merged = self.pattern.merge(&other.pattern);
        Self { pattern: merged }
    }
    
    /// Get synchrony measure
    fn get_synchrony(&self, time_window: f64) -> PyResult<f64> {
        let window = TimeStep::from_secs_f64(time_window);
        self.pattern.calculate_synchrony(window)
            .map_err(core_error_to_py_err)
    }
    
    /// Convert to spike buffer
    fn to_buffer(&self) -> PySpikeBuffer {
        let mut buffer = PySpikeBuffer::new(Some(self.pattern.get_spikes().len()));
        for spike in self.pattern.get_spikes() {
            buffer.add_spike(PySpike::from_spike(spike.clone()));
        }
        buffer
    }
    
    fn __repr__(&self) -> String {
        format!(
            "SpikePattern('{}', {} spikes, {:.3}s)",
            self.get_name(),
            self.pattern.get_spikes().len(),
            self.get_duration()
        )
    }
}

/// Python wrapper for Poisson encoder
#[pyclass(name = "PoissonEncoder")]
pub struct PyPoissonEncoder {
    encoder: PoissonEncoder,
}

#[pymethods]
impl PyPoissonEncoder {
    #[new]
    fn new(max_rate: f32, seed: Option<u64>) -> Self {
        Self {
            encoder: PoissonEncoder::new(max_rate, seed),
        }
    }
    
    /// Encode a value as Poisson spike train
    fn encode(&mut self, value: f32, duration: f64, neuron_id: u32) -> PyResult<Vec<PySpike>> {
        let duration_time = TimeStep::from_secs_f64(duration);
        let spikes = self.encoder.encode(value, duration_time, neuron_id)
            .map_err(core_error_to_py_err)?;
        
        Ok(spikes.into_iter()
            .map(|spike| PySpike::from_spike(spike))
            .collect())
    }
    
    /// Encode array of values
    fn encode_array(&mut self, values: Vec<f32>, duration: f64, start_neuron_id: u32) -> PyResult<Vec<PySpike>> {
        let duration_time = TimeStep::from_secs_f64(duration);
        let mut all_spikes = Vec::new();
        
        for (i, value) in values.iter().enumerate() {
            let neuron_id = start_neuron_id + i as u32;
            let spikes = self.encoder.encode(*value, duration_time, neuron_id)
                .map_err(core_error_to_py_err)?;
            
            all_spikes.extend(spikes.into_iter().map(|spike| PySpike::from_spike(spike)));
        }
        
        Ok(all_spikes)
    }
    
    /// Get maximum rate
    fn get_max_rate(&self) -> f32 {
        self.encoder.get_max_rate()
    }
    
    /// Set maximum rate
    fn set_max_rate(&mut self, max_rate: f32) -> PyResult<()> {
        if max_rate <= 0.0 {
            return Err(PyValueError::new_err("Maximum rate must be positive"));
        }
        self.encoder.set_max_rate(max_rate);
        Ok(())
    }
    
    fn __repr__(&self) -> String {
        format!("PoissonEncoder(max_rate={:.1} Hz)", self.get_max_rate())
    }
}

/// Python wrapper for temporal encoder
#[pyclass(name = "TemporalEncoder")]
pub struct PyTemporalEncoder {
    encoder: TemporalEncoder,
}

#[pymethods]
impl PyTemporalEncoder {
    #[new]
    fn new(num_neurons: u32, min_delay: f64, max_delay: f64) -> PyResult<Self> {
        if min_delay >= max_delay {
            return Err(PyValueError::new_err("min_delay must be less than max_delay"));
        }
        
        let min_delay_time = TimeStep::from_secs_f64(min_delay);
        let max_delay_time = TimeStep::from_secs_f64(max_delay);
        
        let encoder = TemporalEncoder::new(num_neurons, min_delay_time, max_delay_time)
            .map_err(core_error_to_py_err)?;
        
        Ok(Self { encoder })
    }
    
    /// Encode value as temporal spike pattern
    fn encode(&self, value: f32, neuron_id: u32) -> PyResult<Vec<PySpike>> {
        let spikes = self.encoder.encode(value, neuron_id)
            .map_err(core_error_to_py_err)?;
        
        Ok(spikes.into_iter()
            .map(|spike| PySpike::from_spike(spike))
            .collect())
    }
    
    /// Decode spike times back to value
    fn decode(&self, spikes: &PyList) -> PyResult<f32> {
        let mut spike_vec = Vec::new();
        for spike_obj in spikes.iter() {
            let py_spike: PySpike = spike_obj.extract()?;
            spike_vec.push(py_spike.spike);
        }
        
        self.encoder.decode(&spike_vec)
            .map_err(core_error_to_py_err)
    }
    
    /// Get number of neurons
    fn get_num_neurons(&self) -> u32 {
        self.encoder.get_num_neurons()
    }
    
    /// Get delay range
    fn get_delay_range(&self) -> (f64, f64) {
        let (min, max) = self.encoder.get_delay_range();
        (min.as_secs_f64(), max.as_secs_f64())
    }
    
    fn __repr__(&self) -> String {
        let (min_delay, max_delay) = self.get_delay_range();
        format!(
            "TemporalEncoder({} neurons, delay={:.3}-{:.3}s)",
            self.get_num_neurons(),
            min_delay,
            max_delay
        )
    }
}

/// Python wrapper for rate encoder
#[pyclass(name = "RateEncoder")]
pub struct PyRateEncoder {
    encoder: RateEncoder,
}

#[pymethods]
impl PyRateEncoder {
    #[new]
    fn new(max_rate: f32, time_window: f64) -> PyResult<Self> {
        if max_rate <= 0.0 {
            return Err(PyValueError::new_err("Maximum rate must be positive"));
        }
        if time_window <= 0.0 {
            return Err(PyValueError::new_err("Time window must be positive"));
        }
        
        let window_time = TimeStep::from_secs_f64(time_window);
        let encoder = RateEncoder::new(max_rate, window_time)
            .map_err(core_error_to_py_err)?;
        
        Ok(Self { encoder })
    }
    
    /// Encode value as regular spike train
    fn encode(&self, value: f32, duration: f64, neuron_id: u32) -> PyResult<Vec<PySpike>> {
        let duration_time = TimeStep::from_secs_f64(duration);
        let spikes = self.encoder.encode(value, duration_time, neuron_id)
            .map_err(core_error_to_py_err)?;
        
        Ok(spikes.into_iter()
            .map(|spike| PySpike::from_spike(spike))
            .collect())
    }
    
    /// Decode spikes back to rate
    fn decode(&self, spikes: &PyList) -> PyResult<f32> {
        let mut spike_vec = Vec::new();
        for spike_obj in spikes.iter() {
            let py_spike: PySpike = spike_obj.extract()?;
            spike_vec.push(py_spike.spike);
        }
        
        self.encoder.decode(&spike_vec)
            .map_err(core_error_to_py_err)
    }
    
    /// Get maximum rate
    fn get_max_rate(&self) -> f32 {
        self.encoder.get_max_rate()
    }
    
    /// Get time window
    fn get_time_window(&self) -> f64 {
        self.encoder.get_time_window().as_secs_f64()
    }
    
    fn __repr__(&self) -> String {
        format!(
            "RateEncoder(max_rate={:.1} Hz, window={:.3}s)",
            self.get_max_rate(),
            self.get_time_window()
        )
    }
}

/// Spike analysis functions
#[pyfunction]
pub fn calculate_spike_train_distance(
    spikes1: &PyList,
    spikes2: &PyList,
    cost_factor: Option<f64>,
) -> PyResult<f64> {
    let mut spike_vec1 = Vec::new();
    for spike_obj in spikes1.iter() {
        let py_spike: PySpike = spike_obj.extract()?;
        spike_vec1.push(py_spike.spike);
    }
    
    let mut spike_vec2 = Vec::new();
    for spike_obj in spikes2.iter() {
        let py_spike: PySpike = spike_obj.extract()?;
        spike_vec2.push(py_spike.spike);
    }
    
    let cost = cost_factor.unwrap_or(1.0);
    shnn_core::spike::calculate_spike_distance(&spike_vec1, &spike_vec2, cost)
        .map_err(core_error_to_py_err)
}

#[pyfunction]
pub fn detect_bursts(
    spikes: &PyList,
    max_isi: f64,
    min_spikes: Option<u32>,
) -> PyResult<Vec<(f64, f64, u32)>> {
    let mut spike_vec = Vec::new();
    for spike_obj in spikes.iter() {
        let py_spike: PySpike = spike_obj.extract()?;
        spike_vec.push(py_spike.spike);
    }
    
    let max_interval = TimeStep::from_secs_f64(max_isi);
    let min_burst_spikes = min_spikes.unwrap_or(3);
    
    let bursts = shnn_core::spike::detect_bursts(&spike_vec, max_interval, min_burst_spikes)
        .map_err(core_error_to_py_err)?;
    
    Ok(bursts.into_iter()
        .map(|(start, end, count)| (start.as_secs_f64(), end.as_secs_f64(), count))
        .collect())
}

#[pyfunction]
pub fn calculate_population_rate(
    spikes: &PyList,
    bin_size: f64,
    total_neurons: u32,
) -> PyResult<Vec<(f64, f64)>> {
    let mut spike_vec = Vec::new();
    for spike_obj in spikes.iter() {
        let py_spike: PySpike = spike_obj.extract()?;
        spike_vec.push(py_spike.spike);
    }
    
    let bin_duration = TimeStep::from_secs_f64(bin_size);
    
    let rates = shnn_core::spike::calculate_population_rate(&spike_vec, bin_duration, total_neurons)
        .map_err(core_error_to_py_err)?;
    
    Ok(rates.into_iter()
        .map(|(time, rate)| (time.as_secs_f64(), rate))
        .collect())
}

#[pyfunction]
pub fn calculate_synchrony_index(
    spikes: &PyList,
    time_window: f64,
) -> PyResult<f64> {
    let mut spike_vec = Vec::new();
    for spike_obj in spikes.iter() {
        let py_spike: PySpike = spike_obj.extract()?;
        spike_vec.push(py_spike.spike);
    }
    
    let window = TimeStep::from_secs_f64(time_window);
    
    shnn_core::spike::calculate_synchrony_index(&spike_vec, window)
        .map_err(core_error_to_py_err)
}

#[pyfunction]
pub fn generate_random_spikes(
    num_neurons: u32,
    duration: f64,
    rate: f32,
    seed: Option<u64>,
) -> PyResult<Vec<PySpike>> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = match seed {
        Some(seed) => ChaCha8Rng::seed_from_u64(seed),
        None => ChaCha8Rng::from_entropy(),
    };
    
    let mut spikes = Vec::new();
    
    for neuron_id in 0..num_neurons {
        let mut time = 0.0;
        while time < duration {
            let interval = -((1.0 - rng.gen::<f32>()).ln()) / rate;
            time += interval;
            
            if time < duration {
                spikes.push(PySpike::new(neuron_id, time, Some(1.0)));
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
    fn test_spike_buffer() {
        let mut buffer = PySpikeBuffer::new(Some(100));
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity(), 100);
        
        // Add spike
        let spike = PySpike::new(0, 0.001, Some(1.0));
        buffer.add_spike(spike);
        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());
        
        // Test time range
        let range = buffer.get_time_range();
        assert!(range.is_some());
        let (start, end) = range.unwrap();
        assert_eq!(start, 0.001);
        assert_eq!(end, 0.001);
        
        // Clear buffer
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }
    
    #[test]
    fn test_spike_pattern() {
        let mut pattern = PySpikePattern::new("test_pattern".to_string());
        assert_eq!(pattern.get_name(), "test_pattern");
        
        // Add spikes
        pattern.add_spike(0, 0.001, Some(1.0));
        pattern.add_spike(1, 0.002, Some(1.5));
        
        let spikes = pattern.get_spikes();
        assert_eq!(spikes.len(), 2);
        assert!(pattern.get_duration() >= 0.002);
        
        // Test repeat
        let repeated = pattern.repeat(2, 0.01);
        let repeated_spikes = repeated.get_spikes();
        assert_eq!(repeated_spikes.len(), 4); // Original + 1 repeat
    }
    
    #[test]
    fn test_poisson_encoder() {
        let mut encoder = PyPoissonEncoder::new(100.0, Some(42));
        assert_eq!(encoder.get_max_rate(), 100.0);
        
        let spikes = encoder.encode(0.5, 0.1, 0).unwrap();
        assert!(!spikes.is_empty());
        
        // All spikes should be from neuron 0 and within time window
        for spike in &spikes {
            assert_eq!(spike.neuron_id(), 0);
            assert!(spike.time() >= 0.0 && spike.time() <= 0.1);
        }
        
        // Test array encoding
        let values = vec![0.1, 0.5, 0.9];
        let array_spikes = encoder.encode_array(values, 0.1, 10).unwrap();
        
        // Should have spikes from neurons 10, 11, 12
        let neuron_ids: std::collections::HashSet<u32> = array_spikes.iter()
            .map(|s| s.neuron_id())
            .collect();
        assert!(neuron_ids.contains(&10));
        assert!(neuron_ids.contains(&11));
        assert!(neuron_ids.contains(&12));
    }
    
    #[test]
    fn test_temporal_encoder() {
        let encoder = PyTemporalEncoder::new(10, 0.001, 0.01).unwrap();
        assert_eq!(encoder.get_num_neurons(), 10);
        
        let (min_delay, max_delay) = encoder.get_delay_range();
        assert_eq!(min_delay, 0.001);
        assert_eq!(max_delay, 0.01);
        
        let spikes = encoder.encode(0.5, 0).unwrap();
        assert!(!spikes.is_empty());
        
        // Test invalid parameters
        assert!(PyTemporalEncoder::new(10, 0.01, 0.001).is_err()); // min >= max
    }
    
    #[test]
    fn test_rate_encoder() {
        let encoder = PyRateEncoder::new(50.0, 0.02).unwrap();
        assert_eq!(encoder.get_max_rate(), 50.0);
        assert_eq!(encoder.get_time_window(), 0.02);
        
        let spikes = encoder.encode(0.8, 0.1, 5).unwrap();
        
        // All spikes should be from neuron 5
        for spike in &spikes {
            assert_eq!(spike.neuron_id(), 5);
            assert!(spike.time() >= 0.0 && spike.time() <= 0.1);
        }
        
        // Test invalid parameters
        assert!(PyRateEncoder::new(-1.0, 0.02).is_err()); // negative rate
        assert!(PyRateEncoder::new(50.0, -0.02).is_err()); // negative window
    }
    
    #[test]
    fn test_spike_analysis_functions() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Create test spike data
            let spikes1 = PyList::new(py, &[
                PySpike::new(0, 0.001, Some(1.0)),
                PySpike::new(0, 0.003, Some(1.0)),
                PySpike::new(0, 0.005, Some(1.0)),
            ]);
            
            let spikes2 = PyList::new(py, &[
                PySpike::new(0, 0.002, Some(1.0)),
                PySpike::new(0, 0.004, Some(1.0)),
                PySpike::new(0, 0.006, Some(1.0)),
            ]);
            
            // Test spike train distance
            let distance = calculate_spike_train_distance(spikes1, spikes2, None).unwrap();
            assert!(distance >= 0.0);
            
            // Test burst detection
            let bursts = detect_bursts(spikes1, 0.002, Some(2)).unwrap();
            assert!(!bursts.is_empty());
            
            // Test population rate
            let rates = calculate_population_rate(spikes1, 0.001, 1).unwrap();
            assert!(!rates.is_empty());
            
            // Test synchrony index
            let sync = calculate_synchrony_index(spikes1, 0.001).unwrap();
            assert!(sync >= 0.0 && sync <= 1.0);
        });
    }
    
    #[test]
    fn test_random_spike_generation() {
        let spikes = generate_random_spikes(5, 0.1, 100.0, Some(42)).unwrap();
        assert!(!spikes.is_empty());
        
        // Check spikes are sorted by time
        for i in 1..spikes.len() {
            assert!(spikes[i-1].time() <= spikes[i].time());
        }
        
        // Check neuron IDs are valid
        for spike in &spikes {
            assert!(spike.neuron_id() < 5);
            assert!(spike.time() >= 0.0 && spike.time() <= 0.1);
        }
    }
}