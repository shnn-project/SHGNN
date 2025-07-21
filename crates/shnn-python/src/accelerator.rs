//! Python bindings for hardware accelerator management
//!
//! This module provides Python interfaces for managing and using hardware
//! accelerators including CUDA, OpenCL, FPGA, and neuromorphic chips.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::{PyRuntimeError, PyValueError, PyOSError};

use shnn_ffi::{
    AcceleratorRegistry, AcceleratorType, AcceleratorId, AcceleratorInfo,
    AcceleratorCapabilities, PerformanceMetrics, HardwareStatus,
    NetworkId, SpikeData,
};

use crate::error_conversion::ffi_error_to_py_err;
use crate::network::PyPerformanceMetrics;

/// Python wrapper for accelerator type
#[pyclass(name = "AcceleratorType")]
#[derive(Clone, Debug)]
pub struct PyAcceleratorType {
    pub accel_type: AcceleratorType,
}

#[pymethods]
impl PyAcceleratorType {
    #[new]
    fn new(type_name: &str) -> PyResult<Self> {
        let accel_type = match type_name.to_uppercase().as_str() {
            "CPU" => AcceleratorType::CPU,
            "CUDA" | "GPU" => AcceleratorType::CUDA,
            "OPENCL" => AcceleratorType::OpenCL,
            "FPGA" => AcceleratorType::FPGA,
            "RRAM" => AcceleratorType::RRAM,
            "LOIHI" => AcceleratorType::Loihi,
            "SPINNAKER" => AcceleratorType::SpiNNaker,
            "CUSTOM" => AcceleratorType::Custom,
            _ => return Err(PyValueError::new_err(format!("Unknown accelerator type: {}", type_name))),
        };
        
        Ok(Self { accel_type })
    }
    
    /// Create CPU accelerator type
    #[classmethod]
    fn cpu(_cls: &PyType) -> Self {
        Self { accel_type: AcceleratorType::CPU }
    }
    
    /// Create CUDA accelerator type
    #[classmethod]
    fn cuda(_cls: &PyType) -> Self {
        Self { accel_type: AcceleratorType::CUDA }
    }
    
    /// Create OpenCL accelerator type
    #[classmethod]
    fn opencl(_cls: &PyType) -> Self {
        Self { accel_type: AcceleratorType::OpenCL }
    }
    
    /// Create FPGA accelerator type
    #[classmethod]
    fn fpga(_cls: &PyType) -> Self {
        Self { accel_type: AcceleratorType::FPGA }
    }
    
    /// Create RRAM accelerator type
    #[classmethod]
    fn rram(_cls: &PyType) -> Self {
        Self { accel_type: AcceleratorType::RRAM }
    }
    
    /// Create Intel Loihi accelerator type
    #[classmethod]
    fn loihi(_cls: &PyType) -> Self {
        Self { accel_type: AcceleratorType::Loihi }
    }
    
    /// Create SpiNNaker accelerator type
    #[classmethod]
    fn spinnaker(_cls: &PyType) -> Self {
        Self { accel_type: AcceleratorType::SpiNNaker }
    }
    
    /// Get type name as string
    fn name(&self) -> &str {
        match self.accel_type {
            AcceleratorType::CPU => "CPU",
            AcceleratorType::CUDA => "CUDA",
            AcceleratorType::OpenCL => "OpenCL",
            AcceleratorType::FPGA => "FPGA",
            AcceleratorType::RRAM => "RRAM",
            AcceleratorType::Loihi => "Loihi",
            AcceleratorType::SpiNNaker => "SpiNNaker",
            AcceleratorType::Custom => "Custom",
        }
    }
    
    /// Check if accelerator supports neuromorphic computing
    fn is_neuromorphic(&self) -> bool {
        matches!(self.accel_type, AcceleratorType::Loihi | AcceleratorType::SpiNNaker | AcceleratorType::RRAM)
    }
    
    /// Check if accelerator supports parallel processing
    fn is_parallel(&self) -> bool {
        matches!(self.accel_type, AcceleratorType::CUDA | AcceleratorType::OpenCL | AcceleratorType::FPGA)
    }
    
    fn __repr__(&self) -> String {
        format!("AcceleratorType({})", self.name())
    }
    
    fn __str__(&self) -> String {
        self.name().to_string()
    }
    
    fn __eq__(&self, other: &Self) -> bool {
        std::mem::discriminant(&self.accel_type) == std::mem::discriminant(&other.accel_type)
    }
    
    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        std::mem::discriminant(&self.accel_type).hash(&mut hasher);
        hasher.finish()
    }
}

/// Python wrapper for accelerator capabilities
#[pyclass(name = "AcceleratorCapabilities")]
#[derive(Clone, Debug)]
pub struct PyAcceleratorCapabilities {
    pub capabilities: AcceleratorCapabilities,
}

#[pymethods]
impl PyAcceleratorCapabilities {
    #[getter]
    fn max_neurons(&self) -> u32 {
        self.capabilities.max_neurons
    }
    
    #[getter]
    fn max_synapses(&self) -> u64 {
        self.capabilities.max_synapses
    }
    
    #[getter]
    fn max_spike_rate(&self) -> f64 {
        self.capabilities.max_spike_rate
    }
    
    #[getter]
    fn memory_size(&self) -> u64 {
        self.capabilities.memory_size
    }
    
    #[getter]
    fn compute_units(&self) -> u32 {
        self.capabilities.compute_units
    }
    
    #[getter]
    fn supports_plasticity(&self) -> bool {
        self.capabilities.supports_plasticity
    }
    
    #[getter]
    fn supports_stdp(&self) -> bool {
        self.capabilities.supports_stdp
    }
    
    #[getter]
    fn supports_real_time(&self) -> bool {
        self.capabilities.supports_real_time
    }
    
    #[getter]
    fn precision_bits(&self) -> u8 {
        self.capabilities.precision_bits
    }
    
    /// Get capabilities as dictionary
    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("max_neurons", self.capabilities.max_neurons)?;
            dict.set_item("max_synapses", self.capabilities.max_synapses)?;
            dict.set_item("max_spike_rate", self.capabilities.max_spike_rate)?;
            dict.set_item("memory_size", self.capabilities.memory_size)?;
            dict.set_item("compute_units", self.capabilities.compute_units)?;
            dict.set_item("supports_plasticity", self.capabilities.supports_plasticity)?;
            dict.set_item("supports_stdp", self.capabilities.supports_stdp)?;
            dict.set_item("supports_real_time", self.capabilities.supports_real_time)?;
            dict.set_item("precision_bits", self.capabilities.precision_bits)?;
            Ok(dict.to_object(py))
        })
    }
    
    fn __repr__(&self) -> String {
        format!(
            "AcceleratorCapabilities(neurons={}, synapses={}, memory={}MB)",
            self.capabilities.max_neurons,
            self.capabilities.max_synapses,
            self.capabilities.memory_size / (1024 * 1024)
        )
    }
}

/// Python wrapper for accelerator information
#[pyclass(name = "AcceleratorInfo")]
#[derive(Clone, Debug)]
pub struct PyAcceleratorInfo {
    pub info: AcceleratorInfo,
}

#[pymethods]
impl PyAcceleratorInfo {
    #[getter]
    fn id(&self) -> u64 {
        self.info.id.0
    }
    
    #[getter]
    fn name(&self) -> &str {
        &self.info.name
    }
    
    #[getter]
    fn device_id(&self) -> u32 {
        self.info.device_id
    }
    
    #[getter]
    fn vendor(&self) -> &str {
        &self.info.vendor
    }
    
    #[getter]
    fn driver_version(&self) -> &str {
        &self.info.driver_version
    }
    
    #[getter]
    fn accelerator_type(&self) -> PyAcceleratorType {
        PyAcceleratorType {
            accel_type: self.info.accelerator_type.clone(),
        }
    }
    
    #[getter]
    fn capabilities(&self) -> PyAcceleratorCapabilities {
        PyAcceleratorCapabilities {
            capabilities: self.info.capabilities.clone(),
        }
    }
    
    #[getter]
    fn is_available(&self) -> bool {
        self.info.is_available
    }
    
    /// Get info as dictionary
    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("id", self.info.id.0)?;
            dict.set_item("name", &self.info.name)?;
            dict.set_item("device_id", self.info.device_id)?;
            dict.set_item("vendor", &self.info.vendor)?;
            dict.set_item("driver_version", &self.info.driver_version)?;
            dict.set_item("accelerator_type", self.accelerator_type().name())?;
            dict.set_item("capabilities", self.capabilities().to_dict()?)?;
            dict.set_item("is_available", self.info.is_available)?;
            Ok(dict.to_object(py))
        })
    }
    
    fn __repr__(&self) -> String {
        format!(
            "AcceleratorInfo(name='{}', type={}, available={})",
            self.info.name,
            self.accelerator_type().name(),
            self.info.is_available
        )
    }
}

/// Python wrapper for hardware status
#[pyclass(name = "HardwareStatus")]
#[derive(Clone, Debug)]
pub struct PyHardwareStatus {
    pub status: HardwareStatus,
}

#[pymethods]
impl PyHardwareStatus {
    #[getter]
    fn utilization(&self) -> f32 {
        self.status.utilization
    }
    
    #[getter]
    fn temperature(&self) -> f32 {
        self.status.temperature
    }
    
    #[getter]
    fn power_consumption(&self) -> f32 {
        self.status.power_consumption
    }
    
    #[getter]
    fn memory_used(&self) -> u64 {
        self.status.memory_used
    }
    
    #[getter]
    fn memory_total(&self) -> u64 {
        self.status.memory_total
    }
    
    #[getter]
    fn clock_speed(&self) -> u32 {
        self.status.clock_speed
    }
    
    #[getter]
    fn memory_utilization(&self) -> f32 {
        if self.status.memory_total > 0 {
            (self.status.memory_used as f32 / self.status.memory_total as f32) * 100.0
        } else {
            0.0
        }
    }
    
    /// Check if hardware is overheating
    fn is_overheating(&self, threshold: Option<f32>) -> bool {
        let temp_threshold = threshold.unwrap_or(85.0); // Default 85°C
        self.status.temperature > temp_threshold
    }
    
    /// Check if hardware is under high load
    fn is_high_load(&self, threshold: Option<f32>) -> bool {
        let util_threshold = threshold.unwrap_or(90.0); // Default 90%
        self.status.utilization > util_threshold
    }
    
    /// Get status as dictionary
    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("utilization", self.status.utilization)?;
            dict.set_item("temperature", self.status.temperature)?;
            dict.set_item("power_consumption", self.status.power_consumption)?;
            dict.set_item("memory_used", self.status.memory_used)?;
            dict.set_item("memory_total", self.status.memory_total)?;
            dict.set_item("clock_speed", self.status.clock_speed)?;
            dict.set_item("memory_utilization", self.memory_utilization())?;
            Ok(dict.to_object(py))
        })
    }
    
    fn __repr__(&self) -> String {
        format!(
            "HardwareStatus(util={:.1}%, temp={:.1}°C, mem={:.1}%)",
            self.status.utilization,
            self.status.temperature,
            self.memory_utilization()
        )
    }
}

/// Python wrapper for accelerator registry
#[pyclass(name = "AcceleratorRegistry")]
pub struct PyAcceleratorRegistry;

#[pymethods]
impl PyAcceleratorRegistry {
    #[new]
    fn new() -> Self {
        Self
    }
    
    /// Initialize the accelerator registry
    #[staticmethod]
    fn initialize() -> PyResult<()> {
        AcceleratorRegistry::initialize()
            .map_err(ffi_error_to_py_err)
    }
    
    /// Discover available accelerators
    #[staticmethod]
    fn discover_accelerators() -> PyResult<Vec<PyAcceleratorInfo>> {
        let accelerators = AcceleratorRegistry::discover_accelerators()
            .map_err(ffi_error_to_py_err)?;
        
        Ok(accelerators.into_iter()
            .map(|info| PyAcceleratorInfo { info })
            .collect())
    }
    
    /// Get available accelerators by type
    #[staticmethod]
    fn get_accelerators_by_type(accel_type: PyAcceleratorType) -> PyResult<Vec<PyAcceleratorInfo>> {
        let accelerators = AcceleratorRegistry::get_accelerators_by_type(accel_type.accel_type)
            .map_err(ffi_error_to_py_err)?;
        
        Ok(accelerators.into_iter()
            .map(|info| PyAcceleratorInfo { info })
            .collect())
    }
    
    /// Get accelerator information by ID
    #[staticmethod]
    fn get_accelerator_info(accelerator_id: u64) -> PyResult<PyAcceleratorInfo> {
        let accel_id = AcceleratorId(accelerator_id);
        let info = AcceleratorRegistry::get_accelerator_info(accel_id)
            .map_err(ffi_error_to_py_err)?;
        
        Ok(PyAcceleratorInfo { info })
    }
    
    /// Check if accelerator is available
    #[staticmethod]
    fn is_accelerator_available(accelerator_id: u64) -> PyResult<bool> {
        let accel_id = AcceleratorId(accelerator_id);
        AcceleratorRegistry::is_accelerator_available(accel_id)
            .map_err(ffi_error_to_py_err)
    }
    
    /// Get hardware status
    #[staticmethod]
    fn get_hardware_status(accelerator_id: u64) -> PyResult<PyHardwareStatus> {
        let accel_id = AcceleratorId(accelerator_id);
        let status = AcceleratorRegistry::get_hardware_status(accel_id)
            .map_err(ffi_error_to_py_err)?;
        
        Ok(PyHardwareStatus { status })
    }
    
    /// Get performance metrics
    #[staticmethod]
    fn get_performance_metrics(accelerator_id: u64) -> PyResult<PyPerformanceMetrics> {
        let accel_id = AcceleratorId(accelerator_id);
        let metrics = AcceleratorRegistry::get_performance_metrics(accel_id)
            .map_err(ffi_error_to_py_err)?;
        
        Ok(PyPerformanceMetrics::from_metrics(metrics))
    }
    
    /// Set accelerator power mode
    #[staticmethod]
    fn set_power_mode(accelerator_id: u64, power_mode: &str) -> PyResult<()> {
        let accel_id = AcceleratorId(accelerator_id);
        let mode = match power_mode.to_lowercase().as_str() {
            "low" => shnn_ffi::PowerMode::Low,
            "balanced" => shnn_ffi::PowerMode::Balanced,
            "high" => shnn_ffi::PowerMode::High,
            "max" => shnn_ffi::PowerMode::Max,
            _ => return Err(PyValueError::new_err(format!("Unknown power mode: {}", power_mode))),
        };
        
        AcceleratorRegistry::set_power_mode(accel_id, mode)
            .map_err(ffi_error_to_py_err)
    }
    
    /// Reset accelerator
    #[staticmethod]
    fn reset_accelerator(accelerator_id: u64) -> PyResult<()> {
        let accel_id = AcceleratorId(accelerator_id);
        AcceleratorRegistry::reset_accelerator(accel_id)
            .map_err(ffi_error_to_py_err)
    }
    
    /// Get best accelerator for task
    #[staticmethod]
    fn get_best_accelerator(
        required_neurons: Option<u32>,
        required_memory: Option<u64>,
        prefer_neuromorphic: Option<bool>,
    ) -> PyResult<Option<PyAcceleratorInfo>> {
        let accelerators = AcceleratorRegistry::discover_accelerators()
            .map_err(ffi_error_to_py_err)?;
        
        let mut best = None;
        let mut best_score = 0.0f32;
        
        for accel in accelerators {
            if !accel.is_available {
                continue;
            }
            
            let mut score = 1.0f32;
            
            // Check requirements
            if let Some(neurons) = required_neurons {
                if accel.capabilities.max_neurons < neurons {
                    continue;
                }
                score += (accel.capabilities.max_neurons as f32 / neurons as f32).ln();
            }
            
            if let Some(memory) = required_memory {
                if accel.capabilities.memory_size < memory {
                    continue;
                }
                score += (accel.capabilities.memory_size as f32 / memory as f32).ln();
            }
            
            // Prefer neuromorphic if requested
            if prefer_neuromorphic.unwrap_or(false) {
                match accel.accelerator_type {
                    AcceleratorType::Loihi | AcceleratorType::SpiNNaker => score += 10.0,
                    AcceleratorType::RRAM => score += 5.0,
                    _ => {},
                }
            }
            
            // Prefer faster accelerators
            score += accel.capabilities.max_spike_rate as f32 / 1e6;
            
            if score > best_score {
                best_score = score;
                best = Some(accel);
            }
        }
        
        Ok(best.map(|info| PyAcceleratorInfo { info }))
    }
    
    /// Get system information
    #[staticmethod]
    fn get_system_info() -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            // Get all accelerators
            let accelerators = AcceleratorRegistry::discover_accelerators()
                .map_err(ffi_error_to_py_err)?;
            
            let accel_list = PyList::empty(py);
            for accel in accelerators {
                let py_accel = PyAcceleratorInfo { info: accel };
                accel_list.append(py_accel.to_dict()?)?;
            }
            
            dict.set_item("accelerators", accel_list)?;
            dict.set_item("total_accelerators", accel_list.len())?;
            
            // Count by type
            let mut type_counts = std::collections::HashMap::new();
            for i in 0..accel_list.len() {
                let accel_dict = accel_list.get_item(i)?.downcast::<PyDict>()?;
                let accel_type: String = accel_dict.get_item("accelerator_type")?.unwrap().extract()?;
                *type_counts.entry(accel_type).or_insert(0) += 1;
            }
            
            let type_dict = PyDict::new(py);
            for (accel_type, count) in type_counts {
                type_dict.set_item(accel_type, count)?;
            }
            dict.set_item("accelerator_counts", type_dict)?;
            
            Ok(dict.to_object(py))
        })
    }
    
    /// Monitor accelerator performance
    #[staticmethod]
    fn monitor_performance(
        accelerator_id: u64,
        duration_seconds: f32,
        interval_seconds: Option<f32>,
    ) -> PyResult<Vec<PyPerformanceMetrics>> {
        let accel_id = AcceleratorId(accelerator_id);
        let interval = interval_seconds.unwrap_or(1.0);
        let num_samples = (duration_seconds / interval) as usize;
        
        let mut metrics = Vec::new();
        
        for _ in 0..num_samples {
            let current_metrics = AcceleratorRegistry::get_performance_metrics(accel_id)
                .map_err(ffi_error_to_py_err)?;
            
            metrics.push(PyPerformanceMetrics::from_metrics(current_metrics));
            
            // Sleep for interval (simplified for this example)
            std::thread::sleep(std::time::Duration::from_secs_f32(interval));
        }
        
        Ok(metrics)
    }
}

/// Utility functions for accelerator management
#[pyfunction]
pub fn find_cuda_devices() -> PyResult<Vec<PyAcceleratorInfo>> {
    PyAcceleratorRegistry::get_accelerators_by_type(PyAcceleratorType::cuda(
        Python::with_gil(|py| py.get_type::<PyAcceleratorType>())
    ))
}

#[pyfunction]
pub fn find_neuromorphic_devices() -> PyResult<Vec<PyAcceleratorInfo>> {
    let all_accelerators = AcceleratorRegistry::discover_accelerators()
        .map_err(ffi_error_to_py_err)?;
    
    let neuromorphic: Vec<_> = all_accelerators.into_iter()
        .filter(|accel| matches!(
            accel.accelerator_type,
            AcceleratorType::Loihi | AcceleratorType::SpiNNaker | AcceleratorType::RRAM
        ))
        .map(|info| PyAcceleratorInfo { info })
        .collect();
    
    Ok(neuromorphic)
}

#[pyfunction]
pub fn benchmark_accelerator(accelerator_id: u64, test_duration: Option<f32>) -> PyResult<PyObject> {
    let accel_id = AcceleratorId(accelerator_id);
    let duration = test_duration.unwrap_or(10.0);
    
    Python::with_gil(|py| {
        let result_dict = PyDict::new(py);
        
        // Simple benchmark: measure spike processing throughput
        let start_time = std::time::Instant::now();
        let mut total_spikes = 0u64;
        
        while start_time.elapsed().as_secs_f32() < duration {
            // Generate test spike data
            let test_spikes = SpikeData {
                spikes: vec![
                    shnn_ffi::types::SpikeEvent {
                        neuron_id: 0,
                        timestamp: 0.001,
                        amplitude: 1.0,
                    };
                    1000
                ],
                start_time: 0.0,
                end_time: 0.001,
            };
            
            // This would normally process through the accelerator
            // For now, just count the spikes
            total_spikes += test_spikes.spikes.len() as u64;
        }
        
        let elapsed = start_time.elapsed().as_secs_f32();
        let throughput = total_spikes as f64 / elapsed as f64;
        
        result_dict.set_item("duration_seconds", elapsed)?;
        result_dict.set_item("total_spikes", total_spikes)?;
        result_dict.set_item("throughput_spikes_per_second", throughput)?;
        result_dict.set_item("accelerator_id", accelerator_id)?;
        
        Ok(result_dict.to_object(py))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_accelerator_type() {
        let cuda_type = PyAcceleratorType::new("CUDA").unwrap();
        assert_eq!(cuda_type.name(), "CUDA");
        assert!(!cuda_type.is_neuromorphic());
        assert!(cuda_type.is_parallel());
        
        let loihi_type = PyAcceleratorType::new("LOIHI").unwrap();
        assert_eq!(loihi_type.name(), "Loihi");
        assert!(loihi_type.is_neuromorphic());
        assert!(!loihi_type.is_parallel());
        
        // Test invalid type
        assert!(PyAcceleratorType::new("INVALID").is_err());
    }
    
    #[test]
    fn test_accelerator_type_equality() {
        let cuda1 = PyAcceleratorType::cuda(
            pyo3::prepare_freethreaded_python();
            Python::with_gil(|py| py.get_type::<PyAcceleratorType>())
        );
        let cuda2 = PyAcceleratorType::cuda(
            Python::with_gil(|py| py.get_type::<PyAcceleratorType>())
        );
        let opencl = PyAcceleratorType::opencl(
            Python::with_gil(|py| py.get_type::<PyAcceleratorType>())
        );
        
        assert!(cuda1 == cuda2);
        assert!(cuda1 != opencl);
    }
    
    #[test]
    fn test_hardware_status() {
        let status = PyHardwareStatus {
            status: HardwareStatus {
                utilization: 75.5,
                temperature: 82.3,
                power_consumption: 250.0,
                memory_used: 1024 * 1024 * 1024, // 1GB
                memory_total: 4 * 1024 * 1024 * 1024, // 4GB
                clock_speed: 1500,
            },
        };
        
        assert_eq!(status.utilization(), 75.5);
        assert_eq!(status.temperature(), 82.3);
        assert_eq!(status.memory_utilization(), 25.0);
        assert!(!status.is_overheating(None));
        assert!(status.is_overheating(Some(80.0)));
        assert!(!status.is_high_load(None));
        assert!(status.is_high_load(Some(70.0)));
    }
}