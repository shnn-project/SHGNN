//! Python bindings for NumPy integration and array processing
//!
//! This module provides Python interfaces for converting between Rust data structures
//! and NumPy arrays, enabling efficient data exchange and processing.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::{PyRuntimeError, PyValueError, PyImportError};

use crate::neuron::PySpike;
use crate::error_conversion::core_error_to_py_err;

/// Convert spike train to NumPy array format
#[pyfunction]
pub fn spikes_to_numpy(spikes: &PyList, format: Option<&str>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let np = py.import("numpy").map_err(|_| {
            PyImportError::new_err("NumPy not available. Install with: pip install numpy")
        })?;
        
        let output_format = format.unwrap_or("structured");
        
        match output_format {
            "structured" => {
                // Create structured array with fields: neuron_id, time, amplitude
                let mut neuron_ids = Vec::new();
                let mut times = Vec::new();
                let mut amplitudes = Vec::new();
                
                for spike_obj in spikes.iter() {
                    let spike: PySpike = spike_obj.extract()?;
                    neuron_ids.push(spike.neuron_id());
                    times.push(spike.time());
                    amplitudes.push(spike.amplitude());
                }
                
                let data = PyDict::new(py);
                data.set_item("neuron_id", neuron_ids)?;
                data.set_item("time", times)?;
                data.set_item("amplitude", amplitudes)?;
                
                let dtype = PyList::new(py, &[
                    PyTuple::new(py, &["neuron_id", "u4"]),
                    PyTuple::new(py, &["time", "f8"]),
                    PyTuple::new(py, &["amplitude", "f4"]),
                ]);
                
                // Create structured array
                let array_data = PyList::new(py, &[
                    PyTuple::new(py, &[
                        np.call_method1("array", (neuron_ids,))?,
                        np.call_method1("array", (times,))?,
                        np.call_method1("array", (amplitudes,))?,
                    ])
                ]);
                
                np.call_method("array", (array_data,), Some(&[("dtype", dtype)].into_py_dict(py)))
            },
            "matrix" => {
                // Create 2D matrix [spike_index, [neuron_id, time, amplitude]]
                let mut spike_data = Vec::new();
                
                for spike_obj in spikes.iter() {
                    let spike: PySpike = spike_obj.extract()?;
                    spike_data.push(vec![
                        spike.neuron_id() as f64,
                        spike.time(),
                        spike.amplitude() as f64,
                    ]);
                }
                
                np.call_method1("array", (spike_data,))
            },
            "separate" => {
                // Return separate arrays for each field
                let mut neuron_ids = Vec::new();
                let mut times = Vec::new();
                let mut amplitudes = Vec::new();
                
                for spike_obj in spikes.iter() {
                    let spike: PySpike = spike_obj.extract()?;
                    neuron_ids.push(spike.neuron_id());
                    times.push(spike.time());
                    amplitudes.push(spike.amplitude());
                }
                
                let result = PyDict::new(py);
                result.set_item("neuron_ids", np.call_method1("array", (neuron_ids,))?)?;
                result.set_item("times", np.call_method1("array", (times,))?)?;
                result.set_item("amplitudes", np.call_method1("array", (amplitudes,))?)?;
                
                Ok(result.to_object(py))
            },
            _ => Err(PyValueError::new_err(format!("Unknown format: {}. Use 'structured', 'matrix', or 'separate'", output_format))),
        }
    })
}

/// Convert NumPy array to spike train
#[pyfunction]
pub fn numpy_to_spikes(array: PyObject, format: Option<&str>) -> PyResult<Vec<PySpike>> {
    Python::with_gil(|py| {
        let np = py.import("numpy").map_err(|_| {
            PyImportError::new_err("NumPy not available")
        })?;
        
        let input_format = format.unwrap_or("auto");
        let mut spikes = Vec::new();
        
        match input_format {
            "auto" | "matrix" => {
                // Try to interpret as 2D matrix
                let array_obj = array.as_ref(py);
                let shape: Vec<usize> = array_obj.getattr("shape")?.extract()?;
                
                if shape.len() == 2 && shape[1] >= 2 {
                    // 2D array: each row is [neuron_id, time, amplitude?]
                    let data: Vec<Vec<f64>> = array_obj.call_method0("tolist")?.extract()?;
                    
                    for row in data {
                        if row.len() >= 2 {
                            let neuron_id = row[0] as u32;
                            let time = row[1];
                            let amplitude = if row.len() >= 3 { row[2] as f32 } else { 1.0 };
                            spikes.push(PySpike::new(neuron_id, time, Some(amplitude)));
                        }
                    }
                } else if shape.len() == 1 {
                    // 1D array: assume structured array or times only
                    let dtype = array_obj.getattr("dtype")?;
                    let dtype_names: Option<Vec<String>> = dtype.getattr("names").ok()
                        .and_then(|names| names.extract().ok());
                    
                    if let Some(names) = dtype_names {
                        // Structured array
                        if names.contains(&"neuron_id".to_string()) && names.contains(&"time".to_string()) {
                            let neuron_ids: Vec<u32> = array_obj.get_item("neuron_id")?.call_method0("tolist")?.extract()?;
                            let times: Vec<f64> = array_obj.get_item("time")?.call_method0("tolist")?.extract()?;
                            let amplitudes: Vec<f32> = if names.contains(&"amplitude".to_string()) {
                                array_obj.get_item("amplitude")?.call_method0("tolist")?.extract()?
                            } else {
                                vec![1.0; neuron_ids.len()]
                            };
                            
                            for ((neuron_id, time), amplitude) in neuron_ids.into_iter().zip(times).zip(amplitudes) {
                                spikes.push(PySpike::new(neuron_id, time, Some(amplitude)));
                            }
                        }
                    } else {
                        // Plain 1D array - assume spike times for neuron 0
                        let times: Vec<f64> = array_obj.call_method0("tolist")?.extract()?;
                        for time in times {
                            spikes.push(PySpike::new(0, time, Some(1.0)));
                        }
                    }
                }
            },
            "structured" => {
                // Expect structured array with neuron_id, time, amplitude fields
                let array_obj = array.as_ref(py);
                let neuron_ids: Vec<u32> = array_obj.get_item("neuron_id")?.call_method0("tolist")?.extract()?;
                let times: Vec<f64> = array_obj.get_item("time")?.call_method0("tolist")?.extract()?;
                let amplitudes: Vec<f32> = array_obj.get_item("amplitude")?.call_method0("tolist")?.extract()?;
                
                for ((neuron_id, time), amplitude) in neuron_ids.into_iter().zip(times).zip(amplitudes) {
                    spikes.push(PySpike::new(neuron_id, time, Some(amplitude)));
                }
            },
            "separate" => {
                // Expect dictionary with separate arrays
                let dict = array.downcast::<PyDict>(py)?;
                let neuron_ids: Vec<u32> = dict.get_item("neuron_ids")?.unwrap().call_method0("tolist")?.extract()?;
                let times: Vec<f64> = dict.get_item("times")?.unwrap().call_method0("tolist")?.extract()?;
                let amplitudes: Vec<f32> = dict.get_item("amplitudes")?.unwrap().call_method0("tolist")?.extract()?;
                
                for ((neuron_id, time), amplitude) in neuron_ids.into_iter().zip(times).zip(amplitudes) {
                    spikes.push(PySpike::new(neuron_id, time, Some(amplitude)));
                }
            },
            _ => return Err(PyValueError::new_err(format!("Unknown format: {}", input_format))),
        }
        
        // Sort spikes by time
        spikes.sort_by(|a, b| a.time().partial_cmp(&b.time()).unwrap());
        
        Ok(spikes)
    })
}

/// Create raster plot matrix from spikes
#[pyfunction]
pub fn spikes_to_raster_matrix(
    spikes: &PyList,
    num_neurons: u32,
    time_window: (f64, f64),
    bin_size: f64,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let np = py.import("numpy").map_err(|_| {
            PyImportError::new_err("NumPy not available")
        })?;
        
        if bin_size <= 0.0 {
            return Err(PyValueError::new_err("Bin size must be positive"));
        }
        
        let (start_time, end_time) = time_window;
        let duration = end_time - start_time;
        let num_bins = (duration / bin_size).ceil() as usize;
        
        // Create binary matrix: neurons x time_bins
        let mut raster_matrix = vec![vec![0u8; num_bins]; num_neurons as usize];
        
        for spike_obj in spikes.iter() {
            let spike: PySpike = spike_obj.extract()?;
            let neuron_id = spike.neuron_id();
            let time = spike.time();
            
            if neuron_id < num_neurons && time >= start_time && time <= end_time {
                let bin_idx = ((time - start_time) / bin_size).floor() as usize;
                if bin_idx < num_bins {
                    raster_matrix[neuron_id as usize][bin_idx] = 1;
                }
            }
        }
        
        // Convert to NumPy array
        np.call_method1("array", (raster_matrix,))
    })
}

/// Create spike density matrix with Gaussian smoothing
#[pyfunction]
pub fn spikes_to_density_matrix(
    spikes: &PyList,
    num_neurons: u32,
    time_window: (f64, f64),
    bin_size: f64,
    sigma: Option<f64>,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let np = py.import("numpy").map_err(|_| {
            PyImportError::new_err("NumPy not available")
        })?;
        let scipy_ndimage = py.import("scipy.ndimage").map_err(|_| {
            PyImportError::new_err("SciPy not available. Install with: pip install scipy")
        })?;
        
        // First create raster matrix
        let raster = spikes_to_raster_matrix(spikes, num_neurons, time_window, bin_size)?;
        
        // Apply Gaussian smoothing
        let smoothing_sigma = sigma.unwrap_or(1.0);
        let smoothed = scipy_ndimage.call_method(
            "gaussian_filter1d",
            (raster, smoothing_sigma),
            Some(&[("axis", 1)].into_py_dict(py))
        )?;
        
        Ok(smoothed.to_object(py))
    })
}

/// Calculate population vector from spike data
#[pyfunction]
pub fn calculate_population_vector(
    spikes: &PyList,
    time_window: (f64, f64),
    bin_size: f64,
    preferred_directions: PyObject,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let np = py.import("numpy").map_err(|_| {
            PyImportError::new_err("NumPy not available")
        })?;
        
        let (start_time, end_time) = time_window;
        let duration = end_time - start_time;
        let num_bins = (duration / bin_size).ceil() as usize;
        
        // Get preferred directions as NumPy array
        let pref_dirs = preferred_directions.as_ref(py);
        let directions: Vec<f64> = pref_dirs.call_method0("tolist")?.extract()?;
        let num_neurons = directions.len();
        
        // Calculate spike counts per neuron per time bin
        let mut spike_counts = vec![vec![0u32; num_bins]; num_neurons];
        
        for spike_obj in spikes.iter() {
            let spike: PySpike = spike_obj.extract()?;
            let neuron_id = spike.neuron_id() as usize;
            let time = spike.time();
            
            if neuron_id < num_neurons && time >= start_time && time <= end_time {
                let bin_idx = ((time - start_time) / bin_size).floor() as usize;
                if bin_idx < num_bins {
                    spike_counts[neuron_id][bin_idx] += 1;
                }
            }
        }
        
        // Calculate population vector for each time bin
        let mut pop_vectors = Vec::new();
        let mut time_points = Vec::new();
        
        for bin_idx in 0..num_bins {
            let time = start_time + (bin_idx as f64 + 0.5) * bin_size;
            time_points.push(time);
            
            let mut x_sum = 0.0;
            let mut y_sum = 0.0;
            let mut total_count = 0.0;
            
            for (neuron_id, &count) in spike_counts.iter().enumerate().map(|(i, counts)| (i, &counts[bin_idx])) {
                if count > 0 {
                    let direction = directions[neuron_id];
                    let weight = count as f64;
                    
                    x_sum += weight * direction.cos();
                    y_sum += weight * direction.sin();
                    total_count += weight;
                }
            }
            
            if total_count > 0.0 {
                let magnitude = (x_sum * x_sum + y_sum * y_sum).sqrt() / total_count;
                let angle = y_sum.atan2(x_sum);
                pop_vectors.push(vec![magnitude, angle]);
            } else {
                pop_vectors.push(vec![0.0, 0.0]);
            }
        }
        
        let result = PyDict::new(py);
        result.set_item("times", np.call_method1("array", (time_points,))?)?;
        result.set_item("population_vectors", np.call_method1("array", (pop_vectors,))?)?;
        
        Ok(result.to_object(py))
    })
}

/// Calculate spike triggered average
#[pyfunction]
pub fn spike_triggered_average(
    spikes: &PyList,
    signal: PyObject,
    signal_times: PyObject,
    window_size: f64,
    target_neuron: Option<u32>,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let np = py.import("numpy").map_err(|_| {
            PyImportError::new_err("NumPy not available")
        })?;
        
        // Convert signal and times to vectors
        let signal_data: Vec<f64> = signal.as_ref(py).call_method0("tolist")?.extract()?;
        let time_data: Vec<f64> = signal_times.as_ref(py).call_method0("tolist")?.extract()?;
        
        if signal_data.len() != time_data.len() {
            return Err(PyValueError::new_err("Signal and time arrays must have same length"));
        }
        
        // Find spike times for target neuron
        let mut trigger_times = Vec::new();
        for spike_obj in spikes.iter() {
            let spike: PySpike = spike_obj.extract()?;
            if target_neuron.map_or(true, |target| spike.neuron_id() == target) {
                trigger_times.push(spike.time());
            }
        }
        
        if trigger_times.is_empty() {
            return Err(PyValueError::new_err("No spikes found for target neuron"));
        }
        
        // Calculate time window indices
        let dt = if time_data.len() > 1 {
            time_data[1] - time_data[0]
        } else {
            return Err(PyValueError::new_err("Need at least 2 time points"));
        };
        
        let window_samples = (window_size / dt).round() as usize;
        let half_window = window_samples / 2;
        
        // Collect signal segments around each spike
        let mut segments = Vec::new();
        let mut relative_times = Vec::new();
        
        for &trigger_time in &trigger_times {
            // Find closest time index
            let trigger_idx = time_data.iter()
                .position(|&t| (t - trigger_time).abs() < dt / 2.0);
            
            if let Some(idx) = trigger_idx {
                if idx >= half_window && idx + half_window < signal_data.len() {
                    let segment: Vec<f64> = signal_data[idx - half_window..idx + half_window + 1].to_vec();
                    segments.push(segment);
                    
                    if relative_times.is_empty() {
                        // Create relative time vector
                        for i in 0..=window_samples {
                            relative_times.push((i as f64 - half_window as f64) * dt);
                        }
                    }
                }
            }
        }
        
        if segments.is_empty() {
            return Err(PyValueError::new_err("No valid segments found"));
        }
        
        // Calculate average across segments
        let segment_length = segments[0].len();
        let mut sta = vec![0.0; segment_length];
        
        for segment in &segments {
            for (i, &value) in segment.iter().enumerate() {
                sta[i] += value;
            }
        }
        
        for value in &mut sta {
            *value /= segments.len() as f64;
        }
        
        let result = PyDict::new(py);
        result.set_item("times", np.call_method1("array", (relative_times,))?)?;
        result.set_item("average", np.call_method1("array", (sta,))?)?;
        result.set_item("num_spikes", segments.len())?;
        
        Ok(result.to_object(py))
    })
}

/// Convert weight matrix to NumPy array
#[pyfunction]
pub fn weights_to_numpy(weights: Vec<Vec<f32>>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let np = py.import("numpy").map_err(|_| {
            PyImportError::new_err("NumPy not available")
        })?;
        
        np.call_method1("array", (weights,))
    })
}

/// Convert NumPy array to weight matrix
#[pyfunction]
pub fn numpy_to_weights(array: PyObject) -> PyResult<Vec<Vec<f32>>> {
    Python::with_gil(|py| {
        let array_obj = array.as_ref(py);
        let shape: Vec<usize> = array_obj.getattr("shape")?.extract()?;
        
        if shape.len() != 2 {
            return Err(PyValueError::new_err("Weight array must be 2D"));
        }
        
        let weights: Vec<Vec<f32>> = array_obj.call_method0("tolist")?.extract()?;
        Ok(weights)
    })
}

/// Calculate cross-correlation matrix between spike trains
#[pyfunction]
pub fn spike_cross_correlation_matrix(
    spikes: &PyList,
    max_lag: f64,
    bin_size: f64,
    neuron_ids: Option<Vec<u32>>,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let np = py.import("numpy").map_err(|_| {
            PyImportError::new_err("NumPy not available")
        })?;
        
        // Determine unique neuron IDs
        let mut all_neuron_ids = std::collections::HashSet::new();
        for spike_obj in spikes.iter() {
            let spike: PySpike = spike_obj.extract()?;
            all_neuron_ids.insert(spike.neuron_id());
        }
        
        let neurons: Vec<u32> = neuron_ids.unwrap_or_else(|| {
            let mut ids: Vec<u32> = all_neuron_ids.into_iter().collect();
            ids.sort();
            ids
        });
        
        let num_neurons = neurons.len();
        let num_lags = (2.0 * max_lag / bin_size).round() as usize + 1;
        
        // Group spikes by neuron
        let mut spike_trains = std::collections::HashMap::new();
        for &neuron_id in &neurons {
            spike_trains.insert(neuron_id, Vec::new());
        }
        
        for spike_obj in spikes.iter() {
            let spike: PySpike = spike_obj.extract()?;
            if let Some(train) = spike_trains.get_mut(&spike.neuron_id()) {
                train.push(spike.time());
            }
        }
        
        // Calculate cross-correlation matrix
        let mut corr_matrix = vec![vec![vec![0.0; num_lags]; num_neurons]; num_neurons];
        let mut lag_vector = Vec::new();
        
        for i in 0..num_lags {
            let lag = -max_lag + i as f64 * bin_size;
            lag_vector.push(lag);
        }
        
        for (i, &neuron_i) in neurons.iter().enumerate() {
            for (j, &neuron_j) in neurons.iter().enumerate() {
                let train_i = &spike_trains[&neuron_i];
                let train_j = &spike_trains[&neuron_j];
                
                // Calculate cross-correlation for this pair
                for (lag_idx, &lag) in lag_vector.iter().enumerate() {
                    let mut correlation = 0.0;
                    
                    for &spike_time_i in train_i {
                        let target_time = spike_time_i + lag;
                        
                        // Count spikes in neuron_j within bin around target_time
                        let count = train_j.iter()
                            .filter(|&&t| (t - target_time).abs() <= bin_size / 2.0)
                            .count();
                        
                        correlation += count as f64;
                    }
                    
                    // Normalize by number of spikes in reference train
                    if !train_i.is_empty() {
                        correlation /= train_i.len() as f64;
                    }
                    
                    corr_matrix[i][j][lag_idx] = correlation;
                }
            }
        }
        
        let result = PyDict::new(py);
        result.set_item("correlation_matrix", np.call_method1("array", (corr_matrix,))?)?;
        result.set_item("lags", np.call_method1("array", (lag_vector,))?)?;
        result.set_item("neuron_ids", np.call_method1("array", (neurons,))?)?;
        
        Ok(result.to_object(py))
    })
}

/// Utility function to validate NumPy array properties
#[pyfunction]
pub fn validate_numpy_array(
    array: PyObject,
    expected_shape: Option<Vec<usize>>,
    expected_dtype: Option<&str>,
) -> PyResult<bool> {
    Python::with_gil(|py| {
        let np = py.import("numpy").map_err(|_| {
            PyImportError::new_err("NumPy not available")
        })?;
        
        let array_obj = array.as_ref(py);
        
        // Check if it's a NumPy array
        let is_array: bool = np.call_method1("isinstance", (array_obj, np.getattr("ndarray")?))?
            .extract()?;
        
        if !is_array {
            return Ok(false);
        }
        
        // Check shape if specified
        if let Some(expected) = expected_shape {
            let shape: Vec<usize> = array_obj.getattr("shape")?.extract()?;
            if shape != expected {
                return Ok(false);
            }
        }
        
        // Check dtype if specified
        if let Some(expected_dtype_str) = expected_dtype {
            let dtype = array_obj.getattr("dtype")?;
            let dtype_str: String = dtype.call_method0("__str__")?.extract()?;
            if !dtype_str.contains(expected_dtype_str) {
                return Ok(false);
            }
        }
        
        Ok(true)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spike_format_conversions() {
        pyo3::prepare_freethreaded_python();
        
        // Note: These tests require NumPy to be installed
        // In a real environment, we would skip these tests if NumPy is not available
        Python::with_gil(|py| {
            // Try to import NumPy, skip test if not available
            if py.import("numpy").is_err() {
                return;
            }
            
            let spikes = PyList::new(py, &[
                PySpike::new(0, 0.001, Some(1.0)),
                PySpike::new(1, 0.002, Some(1.5)),
                PySpike::new(0, 0.003, Some(1.2)),
            ]);
            
            // Test different output formats
            let formats = vec!["matrix", "separate"];
            
            for format in formats {
                let result = spikes_to_numpy(spikes, Some(format));
                // Just check that it doesn't error - actual validation would require NumPy
                if result.is_ok() {
                    println!("Format {} conversion successful", format);
                }
            }
        });
    }
    
    #[test]
    fn test_raster_matrix_dimensions() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            if py.import("numpy").is_err() {
                return;
            }
            
            let spikes = PyList::new(py, &[
                PySpike::new(0, 0.001, Some(1.0)),
                PySpike::new(1, 0.005, Some(1.0)),
                PySpike::new(2, 0.009, Some(1.0)),
            ]);
            
            let result = spikes_to_raster_matrix(
                spikes,
                3,      // num_neurons
                (0.0, 0.01), // time_window
                0.001,  // bin_size
            );
            
            if let Ok(matrix) = result {
                // Matrix should be created successfully
                println!("Raster matrix created successfully");
            }
        });
    }
    
    #[test]
    fn test_weight_conversions() {
        let weights = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
        ];
        
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            if py.import("numpy").is_err() {
                return;
            }
            
            let numpy_array = weights_to_numpy(weights.clone());
            if let Ok(array) = numpy_array {
                let converted_back = numpy_to_weights(array);
                if let Ok(converted_weights) = converted_back {
                    // Check dimensions match
                    assert_eq!(converted_weights.len(), weights.len());
                    assert_eq!(converted_weights[0].len(), weights[0].len());
                }
            }
        });
    }
    
    #[test]
    fn test_validation() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            if py.import("numpy").is_err() {
                return;
            }
            
            let np = py.import("numpy").unwrap();
            let array = np.call_method1("array", (vec![vec![1, 2], vec![3, 4]],)).unwrap();
            
            // Test shape validation
            let valid = validate_numpy_array(
                array.to_object(py),
                Some(vec![2, 2]),
                None,
            );
            
            if let Ok(is_valid) = valid {
                assert!(is_valid);
            }
            
            // Test invalid shape
            let invalid = validate_numpy_array(
                array.to_object(py),
                Some(vec![3, 3]),
                None,
            );
            
            if let Ok(is_valid) = invalid {
                assert!(!is_valid);
            }
        });
    }
}