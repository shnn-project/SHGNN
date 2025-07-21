//! Python bindings for the SHNN (Spiking Hypergraph Neural Network) library
//!
//! This crate provides comprehensive Python bindings for the entire SHNN ecosystem,
//! enabling Python developers to leverage high-performance neuromorphic computing
//! capabilities with hardware acceleration support.
//!
//! # Features
//!
//! - **Complete API Coverage**: Full access to all SHNN functionality from Python
//! - **NumPy Integration**: Seamless conversion between Python/NumPy arrays and Rust data structures
//! - **Hardware Acceleration**: Direct access to CUDA, OpenCL, FPGA, and neuromorphic hardware
//! - **Async Processing**: Python async/await support for concurrent spike processing
//! - **Visualization**: Built-in plotting and visualization tools using matplotlib
//! - **Performance Profiling**: Comprehensive performance monitoring and optimization tools
//! - **Type Safety**: Comprehensive error handling and validation
//!
//! # Quick Start
//!
//! ```python
//! import shnn
//!
//! # Create a simple spiking neural network
//! network = shnn.Network(num_neurons=1000, connectivity=0.1)
//! network.deploy_to_hardware(accelerator_id=0)
//!
//! # Create and process spikes
//! spikes = shnn.generate_poisson_spikes(rate=100, duration=0.1)
//! output = network.process_spikes(spikes)
//!
//! # Visualize results
//! shnn.plot_raster(output, title="Network Output")
//! ```
//!
//! # Architecture
//!
//! The Python bindings are organized into several modules:
//!
//! - `network`: Neural network creation and management
//! - `neuron`: Individual neuron models (LIF, AdEx, Izhikevich)
//! - `spike`: Spike generation, encoding, and analysis
//! - `plasticity`: Learning rules (STDP, homeostatic, Hebbian)
//! - `accelerator`: Hardware acceleration management
//! - `visualization`: Plotting and visualization tools
//! - `utils`: Utility functions and data processing
//! - `numpy_support`: NumPy integration and array conversion

use pyo3::prelude::*;
use pyo3::types::PyModule;

// Import core dependencies
use shnn_core as core;
use shnn_ffi as ffi;
use shnn_async as async_lib;

// Import all submodules
mod error_conversion;
mod type_conversion;
mod network;
mod neuron;
mod spike;
mod plasticity;
mod accelerator;
mod utils;
mod visualization;
mod numpy_support;

// Re-export Python classes and functions
use network::{PyNetwork, PyNetworkConfig, PyPerformanceMetrics};
use neuron::{
    PyNeuronState, PyNeuronParameters, PyLIFNeuron, PyAdExNeuron, 
    PyIzhikevichNeuron, PySpike, create_poisson_spike_train,
    create_regular_spike_train, create_burst_spike_train,
};
use spike::{
    PySpikeBuffer, PySpikePattern, PyPoissonEncoder, PyTemporalEncoder,
    PyRateEncoder, calculate_spike_train_distance, detect_bursts,
    calculate_population_rate, calculate_synchrony_index, generate_random_spikes,
};
use plasticity::{
    PySTDPRule, PyHomeostaticRule, PyBCMRule, PyOjaRule, PyHebbianRule,
    plot_stdp_window, simulate_weight_evolution, calculate_weight_distribution,
};
use accelerator::{
    PyAcceleratorType, PyAcceleratorCapabilities, PyAcceleratorInfo,
    PyHardwareStatus, PyAcceleratorRegistry, find_cuda_devices,
    find_neuromorphic_devices, benchmark_accelerator,
};
use utils::{
    PyStatistics, PyCorrelation, PyFFTProcessor, PySignalProcessor,
    PyProfiler, PyPerformanceReport, PyMemoryTracker,
    validate_spike_train, validate_network_parameters,
    serialize_network_to_bytes, deserialize_network_from_bytes,
    spike_times_to_binary_array, binary_array_to_spike_times,
    interpolate_missing_values, generate_test_spike_data,
};
use visualization::{
    PyRasterPlot, PyMembraneTrace, PyWeightMatrix,
    create_raster_plot, create_membrane_trace, create_weight_matrix,
    plot_spike_raster, plot_membrane_potential, plot_weight_matrix, plot_firing_rate,
};
use numpy_support::{
    spikes_to_numpy, numpy_to_spikes, spikes_to_raster_matrix,
    spikes_to_density_matrix, calculate_population_vector,
    spike_triggered_average, weights_to_numpy, numpy_to_weights,
    spike_cross_correlation_matrix, validate_numpy_array,
};

/// Initialize the accelerator registry on module import
fn initialize_accelerators() -> PyResult<()> {
    ffi::AcceleratorRegistry::initialize()
        .map_err(error_conversion::ffi_error_to_py_err)?;
    Ok(())
}

/// Python module definition
#[pymodule]
fn shnn(_py: Python, m: &PyModule) -> PyResult<()> {
    // Initialize hardware accelerators
    initialize_accelerators()?;
    
    // Module metadata
    m.add("__version__", "0.1.0")?;
    m.add("__doc__", "Spiking Hypergraph Neural Network library with hardware acceleration support")?;
    
    // Core network classes
    m.add_class::<PyNetwork>()?;
    m.add_class::<PyNetworkConfig>()?;
    m.add_class::<PyPerformanceMetrics>()?;
    
    // Neuron classes
    m.add_class::<PyNeuronState>()?;
    m.add_class::<PyNeuronParameters>()?;
    m.add_class::<PyLIFNeuron>()?;
    m.add_class::<PyAdExNeuron>()?;
    m.add_class::<PyIzhikevichNeuron>()?;
    m.add_class::<PySpike>()?;
    
    // Spike processing classes
    m.add_class::<PySpikeBuffer>()?;
    m.add_class::<PySpikePattern>()?;
    m.add_class::<PyPoissonEncoder>()?;
    m.add_class::<PyTemporalEncoder>()?;
    m.add_class::<PyRateEncoder>()?;
    
    // Plasticity classes
    m.add_class::<PySTDPRule>()?;
    m.add_class::<PyHomeostaticRule>()?;
    m.add_class::<PyBCMRule>()?;
    m.add_class::<PyOjaRule>()?;
    m.add_class::<PyHebbianRule>()?;
    
    // Accelerator classes
    m.add_class::<PyAcceleratorType>()?;
    m.add_class::<PyAcceleratorCapabilities>()?;
    m.add_class::<PyAcceleratorInfo>()?;
    m.add_class::<PyHardwareStatus>()?;
    m.add_class::<PyAcceleratorRegistry>()?;
    
    // Utility classes
    m.add_class::<PyStatistics>()?;
    m.add_class::<PyCorrelation>()?;
    m.add_class::<PyFFTProcessor>()?;
    m.add_class::<PySignalProcessor>()?;
    m.add_class::<PyProfiler>()?;
    m.add_class::<PyPerformanceReport>()?;
    m.add_class::<PyMemoryTracker>()?;
    
    // Visualization classes
    m.add_class::<PyRasterPlot>()?;
    m.add_class::<PyMembraneTrace>()?;
    m.add_class::<PyWeightMatrix>()?;
    
    // Spike generation functions
    m.add_function(wrap_pyfunction!(create_poisson_spike_train, m)?)?;
    m.add_function(wrap_pyfunction!(create_regular_spike_train, m)?)?;
    m.add_function(wrap_pyfunction!(create_burst_spike_train, m)?)?;
    m.add_function(wrap_pyfunction!(generate_random_spikes, m)?)?;
    
    // Spike analysis functions
    m.add_function(wrap_pyfunction!(calculate_spike_train_distance, m)?)?;
    m.add_function(wrap_pyfunction!(detect_bursts, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_population_rate, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_synchrony_index, m)?)?;
    
    // Plasticity analysis functions
    m.add_function(wrap_pyfunction!(plot_stdp_window, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_weight_evolution, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_weight_distribution, m)?)?;
    
    // Accelerator utility functions
    m.add_function(wrap_pyfunction!(find_cuda_devices, m)?)?;
    m.add_function(wrap_pyfunction!(find_neuromorphic_devices, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_accelerator, m)?)?;
    
    // Validation functions
    m.add_function(wrap_pyfunction!(validate_spike_train, m)?)?;
    m.add_function(wrap_pyfunction!(validate_network_parameters, m)?)?;
    
    // Serialization functions
    m.add_function(wrap_pyfunction!(serialize_network_to_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize_network_from_bytes, m)?)?;
    
    // Data conversion functions
    m.add_function(wrap_pyfunction!(spike_times_to_binary_array, m)?)?;
    m.add_function(wrap_pyfunction!(binary_array_to_spike_times, m)?)?;
    m.add_function(wrap_pyfunction!(interpolate_missing_values, m)?)?;
    m.add_function(wrap_pyfunction!(generate_test_spike_data, m)?)?;
    
    // Visualization functions
    m.add_function(wrap_pyfunction!(create_raster_plot, m)?)?;
    m.add_function(wrap_pyfunction!(create_membrane_trace, m)?)?;
    m.add_function(wrap_pyfunction!(create_weight_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(plot_spike_raster, m)?)?;
    m.add_function(wrap_pyfunction!(plot_membrane_potential, m)?)?;
    m.add_function(wrap_pyfunction!(plot_weight_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(plot_firing_rate, m)?)?;
    
    // NumPy integration functions
    m.add_function(wrap_pyfunction!(spikes_to_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_to_spikes, m)?)?;
    m.add_function(wrap_pyfunction!(spikes_to_raster_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(spikes_to_density_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_population_vector, m)?)?;
    m.add_function(wrap_pyfunction!(spike_triggered_average, m)?)?;
    m.add_function(wrap_pyfunction!(weights_to_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_to_weights, m)?)?;
    m.add_function(wrap_pyfunction!(spike_cross_correlation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(validate_numpy_array, m)?)?;
    
    // Add version information
    m.add("SHNN_VERSION", env!("CARGO_PKG_VERSION"))?;
    m.add("SHNN_CORE_VERSION", core::VERSION)?;
    
    // Add feature flags
    let features = pyo3::types::PyDict::new(_py);
    features.set_item("cuda", cfg!(feature = "cuda"))?;
    features.set_item("opencl", cfg!(feature = "opencl"))?;
    features.set_item("fpga", cfg!(feature = "fpga"))?;
    features.set_item("loihi", cfg!(feature = "loihi"))?;
    features.set_item("spinnaker", cfg!(feature = "spinnaker"))?;
    features.set_item("rram", cfg!(feature = "rram"))?;
    features.set_item("async", cfg!(feature = "async"))?;
    features.set_item("wasm", cfg!(feature = "wasm"))?;
    features.set_item("embedded", cfg!(feature = "embedded"))?;
    m.add("FEATURES", features)?;
    
    Ok(())
}

// Re-export for easier access
pub use network::*;
pub use neuron::*;
pub use spike::*;
pub use plasticity::*;
pub use accelerator::*;
pub use utils::*;
pub use visualization::*;
pub use numpy_support::*;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_initialization() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = PyModule::new(py, "test_shnn").unwrap();
            shnn(py, module).unwrap();
            
            // Test that basic classes are available
            assert!(module.getattr("Network").is_ok());
            assert!(module.getattr("LIFNeuron").is_ok());
            assert!(module.getattr("AcceleratorRegistry").is_ok());
        });
    }
    
    #[test]
    fn test_spike_creation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let spike = PySpike::new(42, 0.001, Some(1.5));
            assert_eq!(spike.neuron_id(), 42);
            assert_eq!(spike.time(), 0.001);
            assert_eq!(spike.amplitude(), 1.5);
        });
    }
    
    #[test]
    fn test_network_creation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let network = PyNetwork::new(1000, 0.1, 0.001, None).unwrap();
            assert!(!network.is_deployed());
        });
    }
    
    #[test]
    fn test_accelerator_types() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let cuda_type = PyAcceleratorType::cuda(py.get_type::<PyAcceleratorType>());
            assert_eq!(cuda_type.name(), "CUDA");
            assert!(!cuda_type.is_neuromorphic());
            assert!(cuda_type.is_parallel());
            
            let loihi_type = PyAcceleratorType::loihi(py.get_type::<PyAcceleratorType>());
            assert_eq!(loihi_type.name(), "Loihi");
            assert!(loihi_type.is_neuromorphic());
        });
    }
    
    #[test]
    fn test_plasticity_rules() {
        let mut stdp = PySTDPRule::new(0.01, 0.01, 20.0, 20.0, "additive", 0.0, 1.0, None).unwrap();
        assert_eq!(stdp.a_plus(), 0.01);
        assert_eq!(stdp.a_minus(), 0.01);
        
        stdp.set_a_plus(0.02).unwrap();
        assert_eq!(stdp.a_plus(), 0.02);
    }
}