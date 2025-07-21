//! Python bindings for neural network operations
//!
//! This module provides Python classes and functions for creating, configuring,
//! and managing spiking neural networks.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::{PyRuntimeError, PyValueError};

use shnn_ffi::{
    AcceleratorRegistry, NetworkConfig, SpikeData, NetworkId, AcceleratorId,
    AcceleratorType, PerformanceMetrics, NetworkUpdate,
};

use crate::error_conversion::ffi_error_to_py_err;
use crate::type_conversion::{py_list_to_spike_data, spike_data_to_py_list, py_dict_to_network_config};

/// Python wrapper for neural network
#[pyclass(name = "Network")]
pub struct PyNetwork {
    /// Network configuration
    config: NetworkConfig,
    /// Deployed network ID (if deployed to hardware)
    network_id: Option<NetworkId>,
    /// Associated accelerator ID (if using hardware acceleration)
    accelerator_id: Option<AcceleratorId>,
    /// Network metadata
    metadata: std::collections::HashMap<String, String>,
}

#[pymethods]
impl PyNetwork {
    /// Create a new neural network
    #[new]
    #[pyo3(signature = (num_neurons=1000, connectivity=0.1, dt=0.001, **kwargs))]
    fn new(
        num_neurons: u32,
        connectivity: f32,
        dt: f32,
        kwargs: Option<&PyDict>,
    ) -> PyResult<Self> {
        let mut config = NetworkConfig {
            num_neurons,
            connectivity,
            dt,
            num_connections: (num_neurons as f32 * connectivity) as u32,
            input_size: num_neurons / 10,
            output_size: num_neurons / 100,
            ..Default::default()
        };
        
        // Process additional keyword arguments
        if let Some(kwargs) = kwargs {
            if let Some(input_size) = kwargs.get_item("input_size")? {
                config.input_size = input_size.extract()?;
            }
            
            if let Some(output_size) = kwargs.get_item("output_size")? {
                config.output_size = output_size.extract()?;
            }
            
            if let Some(hidden_layers) = kwargs.get_item("hidden_layers")? {
                config.hidden_layers = hidden_layers.extract()?;
            }
            
            if let Some(topology) = kwargs.get_item("topology")? {
                let topology_str: String = topology.extract()?;
                config.topology = match topology_str.to_lowercase().as_str() {
                    "feedforward" => shnn_ffi::types::NetworkTopology::Feedforward,
                    "recurrent" => shnn_ffi::types::NetworkTopology::Recurrent,
                    "convolutional" => shnn_ffi::types::NetworkTopology::Convolutional,
                    "smallworld" => shnn_ffi::types::NetworkTopology::SmallWorld,
                    "scalefree" => shnn_ffi::types::NetworkTopology::ScaleFree,
                    _ => return Err(PyValueError::new_err(format!("Unknown topology: {}", topology_str))),
                };
            }
        }
        
        Ok(Self {
            config,
            network_id: None,
            accelerator_id: None,
            metadata: std::collections::HashMap::new(),
        })
    }
    
    /// Create a feedforward network
    #[classmethod]
    fn feedforward(
        _cls: &PyType,
        layer_sizes: Vec<u32>,
        dt: Option<f32>,
    ) -> PyResult<Self> {
        if layer_sizes.len() < 2 {
            return Err(PyValueError::new_err("Feedforward network needs at least 2 layers"));
        }
        
        let total_neurons: u32 = layer_sizes.iter().sum();
        let mut config = NetworkConfig {
            num_neurons: total_neurons,
            hidden_layers: layer_sizes[1..layer_sizes.len()-1].to_vec(),
            input_size: layer_sizes[0],
            output_size: *layer_sizes.last().unwrap(),
            dt: dt.unwrap_or(0.001),
            topology: shnn_ffi::types::NetworkTopology::Feedforward,
            ..Default::default()
        };
        
        // Estimate connections based on layer structure
        let mut connections = 0u32;
        for i in 0..layer_sizes.len()-1 {
            connections += layer_sizes[i] * layer_sizes[i+1];
        }
        config.num_connections = connections;
        config.connectivity = connections as f32 / (total_neurons * total_neurons) as f32;
        
        Ok(Self {
            config,
            network_id: None,
            accelerator_id: None,
            metadata: std::collections::HashMap::new(),
        })
    }
    
    /// Create a recurrent network
    #[classmethod]
    fn recurrent(
        _cls: &PyType,
        num_neurons: u32,
        connectivity: f32,
        dt: Option<f32>,
    ) -> PyResult<Self> {
        let config = NetworkConfig {
            num_neurons,
            connectivity,
            dt: dt.unwrap_or(0.001),
            num_connections: (num_neurons as f32 * num_neurons as f32 * connectivity) as u32,
            input_size: num_neurons / 10,
            output_size: num_neurons / 10,
            topology: shnn_ffi::types::NetworkTopology::Recurrent,
            ..Default::default()
        };
        
        Ok(Self {
            config,
            network_id: None,
            accelerator_id: None,
            metadata: std::collections::HashMap::new(),
        })
    }
    
    /// Deploy network to hardware accelerator
    fn deploy_to_hardware(&mut self, accelerator_id: u64) -> PyResult<()> {
        let accel_id = AcceleratorId(accelerator_id);
        
        let network_id = AcceleratorRegistry::deploy_network(accel_id, &self.config)
            .map_err(ffi_error_to_py_err)?;
        
        self.network_id = Some(network_id);
        self.accelerator_id = Some(accel_id);
        
        Ok(())
    }
    
    /// Process spikes through the network
    fn process_spikes(&self, input_spikes: &PyList) -> PyResult<PyObject> {
        // Check if network is deployed
        let (network_id, accelerator_id) = match (self.network_id, self.accelerator_id) {
            (Some(net_id), Some(accel_id)) => (net_id, accel_id),
            _ => return Err(PyRuntimeError::new_err("Network not deployed to hardware. Call deploy_to_hardware() first.")),
        };
        
        // Convert input spikes
        let spikes = py_list_to_spike_data(input_spikes)?;
        
        // Process spikes
        let output_spikes = AcceleratorRegistry::process_spikes(accelerator_id, network_id, &spikes)
            .map_err(ffi_error_to_py_err)?;
        
        // Convert output spikes back to Python
        Python::with_gil(|py| {
            spike_data_to_py_list(py, &output_spikes)
        })
    }
    
    /// Process spikes in batch
    fn process_batch(&self, spike_batches: &PyList) -> PyResult<PyObject> {
        let (network_id, accelerator_id) = match (self.network_id, self.accelerator_id) {
            (Some(net_id), Some(accel_id)) => (net_id, accel_id),
            _ => return Err(PyRuntimeError::new_err("Network not deployed to hardware")),
        };
        
        Python::with_gil(|py| {
            let output_batches = PyList::empty(py);
            
            for batch in spike_batches.iter() {
                let batch_list = batch.downcast::<PyList>()?;
                let spikes = py_list_to_spike_data(batch_list)?;
                
                let output_spikes = AcceleratorRegistry::process_spikes(accelerator_id, network_id, &spikes)
                    .map_err(ffi_error_to_py_err)?;
                
                let output_list = spike_data_to_py_list(py, &output_spikes)?;
                output_batches.append(output_list)?;
            }
            
            Ok(output_batches.to_object(py))
        })
    }
    
    /// Update network weights
    fn update_weights(&self, weight_updates: &PyList) -> PyResult<()> {
        let (network_id, accelerator_id) = match (self.network_id, self.accelerator_id) {
            (Some(net_id), Some(accel_id)) => (net_id, accel_id),
            _ => return Err(PyRuntimeError::new_err("Network not deployed to hardware")),
        };
        
        let mut updates = Vec::new();
        for update in weight_updates.iter() {
            let tuple = update.downcast::<pyo3::types::PyTuple>()?;
            if tuple.len() != 3 {
                return Err(PyValueError::new_err("Weight update must be (pre_neuron, post_neuron, new_weight)"));
            }
            
            let pre_neuron: u32 = tuple.get_item(0)?.extract()?;
            let post_neuron: u32 = tuple.get_item(1)?.extract()?;
            let new_weight: f32 = tuple.get_item(2)?.extract()?;
            
            updates.push((pre_neuron, post_neuron, new_weight));
        }
        
        let network_update = NetworkUpdate::UpdateWeights { updates };
        
        AcceleratorRegistry::update_network(accelerator_id, network_id, &network_update)
            .map_err(ffi_error_to_py_err)?;
        
        Ok(())
    }
    
    /// Get network performance metrics
    fn get_performance_metrics(&self) -> PyResult<PyPerformanceMetrics> {
        let accelerator_id = self.accelerator_id
            .ok_or_else(|| PyRuntimeError::new_err("Network not deployed to hardware"))?;
        
        let metrics = AcceleratorRegistry::get_performance_metrics(accelerator_id)
            .map_err(ffi_error_to_py_err)?;
        
        Ok(PyPerformanceMetrics::from_metrics(metrics))
    }
    
    /// Reset network to initial state
    fn reset(&self) -> PyResult<()> {
        let (network_id, accelerator_id) = match (self.network_id, self.accelerator_id) {
            (Some(net_id), Some(accel_id)) => (net_id, accel_id),
            _ => return Err(PyRuntimeError::new_err("Network not deployed to hardware")),
        };
        
        // Reset would be implemented in the accelerator
        // For now, just return success
        Ok(())
    }
    
    /// Get network configuration as dictionary
    fn get_config(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            crate::type_conversion::network_config_to_py_dict(py, &self.config)
        })
    }
    
    /// Update network configuration
    fn update_config(&mut self, config_dict: &PyDict) -> PyResult<()> {
        if self.network_id.is_some() {
            return Err(PyRuntimeError::new_err("Cannot update config of deployed network"));
        }
        
        self.config = py_dict_to_network_config(config_dict)?;
        Ok(())
    }
    
    /// Add metadata to the network
    fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
    
    /// Get metadata from the network
    fn get_metadata(&self, key: String) -> Option<String> {
        self.metadata.get(&key).cloned()
    }
    
    /// Get all metadata
    fn get_all_metadata(&self) -> std::collections::HashMap<String, String> {
        self.metadata.clone()
    }
    
    /// Check if network is deployed
    fn is_deployed(&self) -> bool {
        self.network_id.is_some()
    }
    
    /// Get network statistics
    fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = PyDict::new(py);
            stats.set_item("num_neurons", self.config.num_neurons)?;
            stats.set_item("num_connections", self.config.num_connections)?;
            stats.set_item("connectivity", self.config.connectivity)?;
            stats.set_item("input_size", self.config.input_size)?;
            stats.set_item("output_size", self.config.output_size)?;
            stats.set_item("is_deployed", self.is_deployed())?;
            
            if let Some(network_id) = self.network_id {
                stats.set_item("network_id", network_id.0)?;
            }
            
            if let Some(accelerator_id) = self.accelerator_id {
                stats.set_item("accelerator_id", accelerator_id.0)?;
            }
            
            Ok(stats.to_object(py))
        })
    }
    
    /// Save network configuration to file
    fn save_config(&self, filename: String) -> PyResult<()> {
        let json_config = serde_json::to_string_pretty(&self.config)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization failed: {}", e)))?;
        
        std::fs::write(filename, json_config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write file: {}", e)))?;
        
        Ok(())
    }
    
    /// Load network configuration from file
    #[classmethod]
    fn load_config(_cls: &PyType, filename: String) -> PyResult<Self> {
        let json_config = std::fs::read_to_string(filename)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read file: {}", e)))?;
        
        let config: NetworkConfig = serde_json::from_str(&json_config)
            .map_err(|e| PyRuntimeError::new_err(format!("Deserialization failed: {}", e)))?;
        
        Ok(Self {
            config,
            network_id: None,
            accelerator_id: None,
            metadata: std::collections::HashMap::new(),
        })
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Network(neurons={}, connections={}, deployed={})",
            self.config.num_neurons,
            self.config.num_connections,
            self.is_deployed()
        )
    }
    
    /// String representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python wrapper for network configuration
#[pyclass(name = "NetworkConfig")]
pub struct PyNetworkConfig {
    config: NetworkConfig,
}

#[pymethods]
impl PyNetworkConfig {
    #[new]
    fn new() -> Self {
        Self {
            config: NetworkConfig::default(),
        }
    }
    
    #[getter]
    fn num_neurons(&self) -> u32 {
        self.config.num_neurons
    }
    
    #[setter]
    fn set_num_neurons(&mut self, value: u32) {
        self.config.num_neurons = value;
    }
    
    #[getter]
    fn num_connections(&self) -> u32 {
        self.config.num_connections
    }
    
    #[setter]
    fn set_num_connections(&mut self, value: u32) {
        self.config.num_connections = value;
    }
    
    #[getter]
    fn connectivity(&self) -> f32 {
        self.config.connectivity
    }
    
    #[setter]
    fn set_connectivity(&mut self, value: f32) -> PyResult<()> {
        if value < 0.0 || value > 1.0 {
            return Err(PyValueError::new_err("Connectivity must be between 0.0 and 1.0"));
        }
        self.config.connectivity = value;
        Ok(())
    }
    
    #[getter]
    fn dt(&self) -> f32 {
        self.config.dt
    }
    
    #[setter]
    fn set_dt(&mut self, value: f32) -> PyResult<()> {
        if value <= 0.0 {
            return Err(PyValueError::new_err("Time step must be positive"));
        }
        self.config.dt = value;
        Ok(())
    }
    
    #[getter]
    fn input_size(&self) -> u32 {
        self.config.input_size
    }
    
    #[setter]
    fn set_input_size(&mut self, value: u32) {
        self.config.input_size = value;
    }
    
    #[getter]
    fn output_size(&self) -> u32 {
        self.config.output_size
    }
    
    #[setter]
    fn set_output_size(&mut self, value: u32) {
        self.config.output_size = value;
    }
    
    #[getter]
    fn hidden_layers(&self) -> Vec<u32> {
        self.config.hidden_layers.clone()
    }
    
    #[setter]
    fn set_hidden_layers(&mut self, value: Vec<u32>) {
        self.config.hidden_layers = value;
    }
    
    /// Validate configuration
    fn validate(&self) -> PyResult<()> {
        shnn_ffi::utils::validation::validate_network_config(&self.config)
            .map_err(ffi_error_to_py_err)
    }
    
    /// Get configuration as dictionary
    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            crate::type_conversion::network_config_to_py_dict(py, &self.config)
        })
    }
    
    /// Create from dictionary
    #[classmethod]
    fn from_dict(_cls: &PyType, config_dict: &PyDict) -> PyResult<Self> {
        let config = py_dict_to_network_config(config_dict)?;
        Ok(Self { config })
    }
}

impl PyNetworkConfig {
    /// Get the internal configuration
    pub fn get_config(&self) -> &NetworkConfig {
        &self.config
    }
}

/// Python wrapper for performance metrics
#[pyclass(name = "PerformanceMetrics")]
pub struct PyPerformanceMetrics {
    metrics: PerformanceMetrics,
}

#[pymethods]
impl PyPerformanceMetrics {
    #[getter]
    fn execution_time_ms(&self) -> f64 {
        self.metrics.execution_time_ms
    }
    
    #[getter]
    fn spikes_per_second(&self) -> f64 {
        self.metrics.spikes_per_second
    }
    
    #[getter]
    fn memory_usage(&self) -> u64 {
        self.metrics.memory_usage
    }
    
    #[getter]
    fn power_consumption(&self) -> f32 {
        self.metrics.power_consumption
    }
    
    #[getter]
    fn gpu_utilization(&self) -> f32 {
        self.metrics.gpu_utilization
    }
    
    #[getter]
    fn memory_utilization(&self) -> f32 {
        self.metrics.memory_utilization
    }
    
    /// Get metrics as dictionary
    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("execution_time_ms", self.metrics.execution_time_ms)?;
            dict.set_item("spikes_per_second", self.metrics.spikes_per_second)?;
            dict.set_item("memory_usage", self.metrics.memory_usage)?;
            dict.set_item("power_consumption", self.metrics.power_consumption)?;
            dict.set_item("gpu_utilization", self.metrics.gpu_utilization)?;
            dict.set_item("memory_utilization", self.metrics.memory_utilization)?;
            Ok(dict.to_object(py))
        })
    }
    
    fn __repr__(&self) -> String {
        format!(
            "PerformanceMetrics(execution_time={:.2}ms, throughput={:.0} spikes/s, memory={}MB)",
            self.metrics.execution_time_ms,
            self.metrics.spikes_per_second,
            self.metrics.memory_usage / (1024 * 1024)
        )
    }
}

impl PyPerformanceMetrics {
    pub fn from_metrics(metrics: PerformanceMetrics) -> Self {
        Self { metrics }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_network_creation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let network = PyNetwork::new(1000, 0.1, 0.001, None).unwrap();
            assert_eq!(network.config.num_neurons, 1000);
            assert!((network.config.connectivity - 0.1).abs() < 1e-6);
            assert!((network.config.dt - 0.001).abs() < 1e-6);
            assert!(!network.is_deployed());
        });
    }
    
    #[test]
    fn test_feedforward_network() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let layer_sizes = vec![100, 50, 10];
            let network = PyNetwork::feedforward(
                py.get_type::<PyNetwork>(),
                layer_sizes.clone(),
                Some(0.001),
            ).unwrap();
            
            let total_neurons: u32 = layer_sizes.iter().sum();
            assert_eq!(network.config.num_neurons, total_neurons);
            assert_eq!(network.config.input_size, 100);
            assert_eq!(network.config.output_size, 10);
            assert_eq!(network.config.hidden_layers, vec![50]);
        });
    }
    
    #[test]
    fn test_network_config() {
        let mut config = PyNetworkConfig::new();
        config.set_num_neurons(500);
        config.set_connectivity(0.2).unwrap();
        config.set_dt(0.002).unwrap();
        
        assert_eq!(config.num_neurons(), 500);
        assert!((config.connectivity() - 0.2).abs() < 1e-6);
        assert!((config.dt() - 0.002).abs() < 1e-6);
        
        // Test validation
        assert!(config.validate().is_ok());
        
        // Test invalid values
        assert!(config.set_connectivity(-0.1).is_err());
        assert!(config.set_dt(-0.001).is_err());
    }
}