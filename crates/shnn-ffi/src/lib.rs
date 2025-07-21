//! # SHNN FFI - Foreign Function Interface for Hardware Acceleration
//!
//! This crate provides FFI bindings and hardware acceleration interfaces for the
//! Spiking Hypergraph Neural Network (SHNN) library, enabling integration with
//! various hardware acceleration platforms and programming languages.
//!
//! ## Supported Hardware Platforms
//!
//! - **CUDA**: NVIDIA GPU acceleration for parallel spike processing
//! - **OpenCL**: Cross-platform parallel computing
//! - **FPGA**: Field-Programmable Gate Array deployment
//! - **RRAM/Memristor**: Resistive memory-based neuromorphic computing
//! - **Intel Loihi**: Neuromorphic research chip
//! - **SpiNNaker**: Large-scale neuromorphic computing platform
//!
//! ## Language Bindings
//!
//! - **C/C++**: Native bindings for system integration
//! - **Python**: High-level scripting and research
//! - **MATLAB**: Scientific computing integration
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use shnn_ffi::{HardwareAccelerator, AcceleratorType, NetworkConfig};
//!
//! // Initialize CUDA accelerator
//! let mut accelerator = HardwareAccelerator::new(AcceleratorType::CUDA)?;
//! 
//! // Configure neural network
//! let config = NetworkConfig {
//!     num_neurons: 1000,
//!     connectivity: 0.1,
//!     dt: 0.001,
//! };
//! 
//! // Deploy network to hardware
//! let network_id = accelerator.deploy_network(&config)?;
//! 
//! // Process spikes on hardware
//! let input_spikes = vec![/* spike data */];
//! let output_spikes = accelerator.process_spikes(network_id, &input_spikes)?;
//! 
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs, rustdoc::broken_intra_doc_links)]
#![deny(unsafe_op_in_unsafe_fn)]

// Core modules
pub mod error;
pub mod types;
pub mod hardware;

// Hardware acceleration backends
#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(feature = "fpga")]
pub mod fpga;

#[cfg(feature = "rram")]
pub mod rram;

#[cfg(feature = "intel-loihi")]
pub mod loihi;

#[cfg(feature = "spiNNaker")]
pub mod spinnaker;

// Language bindings
#[cfg(feature = "c-bindings")]
pub mod c_bindings;

#[cfg(feature = "python-bindings")]
pub mod python;

#[cfg(feature = "cpp-bindings")]
pub mod cpp_bindings;

// Utility modules
pub mod utils;
pub mod profiling;

// Re-exports
pub use error::{FFIError, FFIResult};
pub use types::{
    NetworkConfig, SpikeData, NeuronState, AcceleratorType,
    HardwareCapabilities, PerformanceMetrics
};
pub use hardware::{HardwareAccelerator, AcceleratorManager};

use shnn_core::{spike::Spike, time::Time};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Global accelerator registry for managing hardware instances
static ACCELERATOR_REGISTRY: once_cell::sync::Lazy<Arc<Mutex<AcceleratorRegistry>>> =
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(AcceleratorRegistry::new())));

/// Registry for managing multiple hardware accelerators
#[derive(Debug)]
pub struct AcceleratorRegistry {
    accelerators: HashMap<AcceleratorId, Box<dyn HardwareAccelerator + Send + Sync>>,
    next_id: AcceleratorId,
}

impl AcceleratorRegistry {
    /// Create a new accelerator registry
    pub fn new() -> Self {
        Self {
            accelerators: HashMap::new(),
            next_id: AcceleratorId(0),
        }
    }
    
    /// Register a new hardware accelerator
    pub fn register_accelerator(
        &mut self,
        accelerator: Box<dyn HardwareAccelerator + Send + Sync>
    ) -> AcceleratorId {
        let id = self.next_id;
        self.accelerators.insert(id, accelerator);
        self.next_id = AcceleratorId(self.next_id.0 + 1);
        id
    }
    
    /// Get a reference to an accelerator by ID
    pub fn get_accelerator(&self, id: AcceleratorId) -> Option<&dyn HardwareAccelerator> {
        self.accelerators.get(&id).map(|acc| acc.as_ref())
    }
    
    /// Get a mutable reference to an accelerator by ID
    pub fn get_accelerator_mut(&mut self, id: AcceleratorId) -> Option<&mut dyn HardwareAccelerator> {
        self.accelerators.get_mut(&id).map(|acc| acc.as_mut())
    }
    
    /// Remove an accelerator from the registry
    pub fn unregister_accelerator(&mut self, id: AcceleratorId) -> Option<Box<dyn HardwareAccelerator + Send + Sync>> {
        self.accelerators.remove(&id)
    }
    
    /// List all registered accelerators
    pub fn list_accelerators(&self) -> Vec<AcceleratorId> {
        self.accelerators.keys().copied().collect()
    }
}

/// Unique identifier for hardware accelerators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AcceleratorId(pub u64);

/// Network identifier for deployed neural networks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NetworkId(pub u64);

/// Global API functions for C/Python bindings
impl AcceleratorRegistry {
    /// Initialize the SHNN FFI library
    pub fn initialize() -> FFIResult<()> {
        // Initialize logging
        #[cfg(feature = "profiling")]
        {
            tracing_subscriber::fmt::init();
        }
        
        // Discover available hardware
        Self::discover_hardware()?;
        
        Ok(())
    }
    
    /// Discover available hardware accelerators
    pub fn discover_hardware() -> FFIResult<Vec<AcceleratorType>> {
        let mut available = Vec::new();
        
        #[cfg(feature = "cuda")]
        {
            if cuda::is_available() {
                available.push(AcceleratorType::CUDA);
            }
        }
        
        #[cfg(feature = "opencl")]
        {
            if opencl::is_available() {
                available.push(AcceleratorType::OpenCL);
            }
        }
        
        #[cfg(feature = "fpga")]
        {
            if fpga::is_available() {
                available.push(AcceleratorType::FPGA);
            }
        }
        
        #[cfg(feature = "rram")]
        {
            if rram::is_available() {
                available.push(AcceleratorType::RRAM);
            }
        }
        
        #[cfg(feature = "intel-loihi")]
        {
            if loihi::is_available() {
                available.push(AcceleratorType::IntelLoihi);
            }
        }
        
        #[cfg(feature = "spiNNaker")]
        {
            if spinnaker::is_available() {
                available.push(AcceleratorType::SpiNNaker);
            }
        }
        
        Ok(available)
    }
    
    /// Create a new hardware accelerator instance
    pub fn create_accelerator(accelerator_type: AcceleratorType) -> FFIResult<AcceleratorId> {
        let mut registry = ACCELERATOR_REGISTRY.lock().unwrap();
        
        let accelerator: Box<dyn HardwareAccelerator + Send + Sync> = match accelerator_type {
            #[cfg(feature = "cuda")]
            AcceleratorType::CUDA => Box::new(cuda::CudaAccelerator::new()?),
            
            #[cfg(feature = "opencl")]
            AcceleratorType::OpenCL => Box::new(opencl::OpenCLAccelerator::new()?),
            
            #[cfg(feature = "fpga")]
            AcceleratorType::FPGA => Box::new(fpga::FPGAAccelerator::new()?),
            
            #[cfg(feature = "rram")]
            AcceleratorType::RRAM => Box::new(rram::RRAMAccelerator::new()?),
            
            #[cfg(feature = "intel-loihi")]
            AcceleratorType::IntelLoihi => Box::new(loihi::LoihiAccelerator::new()?),
            
            #[cfg(feature = "spiNNaker")]
            AcceleratorType::SpiNNaker => Box::new(spinnaker::SpiNNakerAccelerator::new()?),
            
            _ => return Err(FFIError::UnsupportedHardware(accelerator_type)),
        };
        
        Ok(registry.register_accelerator(accelerator))
    }
    
    /// Deploy a neural network to hardware
    pub fn deploy_network(
        accelerator_id: AcceleratorId,
        config: &NetworkConfig
    ) -> FFIResult<NetworkId> {
        let mut registry = ACCELERATOR_REGISTRY.lock().unwrap();
        
        if let Some(accelerator) = registry.get_accelerator_mut(accelerator_id) {
            accelerator.deploy_network(config)
        } else {
            Err(FFIError::InvalidAcceleratorId(accelerator_id))
        }
    }
    
    /// Process spikes on hardware accelerator
    pub fn process_spikes(
        accelerator_id: AcceleratorId,
        network_id: NetworkId,
        input_spikes: &[SpikeData],
    ) -> FFIResult<Vec<SpikeData>> {
        let mut registry = ACCELERATOR_REGISTRY.lock().unwrap();
        
        if let Some(accelerator) = registry.get_accelerator_mut(accelerator_id) {
            accelerator.process_spikes(network_id, input_spikes)
        } else {
            Err(FFIError::InvalidAcceleratorId(accelerator_id))
        }
    }
    
    /// Get hardware capabilities
    pub fn get_capabilities(accelerator_id: AcceleratorId) -> FFIResult<HardwareCapabilities> {
        let registry = ACCELERATOR_REGISTRY.lock().unwrap();
        
        if let Some(accelerator) = registry.get_accelerator(accelerator_id) {
            Ok(accelerator.get_capabilities())
        } else {
            Err(FFIError::InvalidAcceleratorId(accelerator_id))
        }
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(
        accelerator_id: AcceleratorId
    ) -> FFIResult<PerformanceMetrics> {
        let registry = ACCELERATOR_REGISTRY.lock().unwrap();
        
        if let Some(accelerator) = registry.get_accelerator(accelerator_id) {
            Ok(accelerator.get_performance_metrics())
        } else {
            Err(FFIError::InvalidAcceleratorId(accelerator_id))
        }
    }
    
    /// Release a hardware accelerator
    pub fn release_accelerator(accelerator_id: AcceleratorId) -> FFIResult<()> {
        let mut registry = ACCELERATOR_REGISTRY.lock().unwrap();
        
        if registry.unregister_accelerator(accelerator_id).is_some() {
            Ok(())
        } else {
            Err(FFIError::InvalidAcceleratorId(accelerator_id))
        }
    }
}

/// Convert between SHNN core types and FFI types
pub mod conversion {
    use super::*;
    use shnn_core::spike::Spike;
    
    /// Convert FFI SpikeData to core Spike
    pub fn spike_data_to_core(spike_data: &SpikeData) -> Spike {
        Spike::new(
            spike_data.neuron_id.into(),
            Time::from_milliseconds(spike_data.timestamp),
            spike_data.amplitude,
        )
    }
    
    /// Convert core Spike to FFI SpikeData
    pub fn core_to_spike_data(spike: &Spike) -> SpikeData {
        SpikeData {
            neuron_id: spike.source_id().into(),
            timestamp: spike.timestamp().as_milliseconds(),
            amplitude: spike.amplitude(),
        }
    }
    
    /// Convert multiple spikes
    pub fn spikes_to_core(spike_data: &[SpikeData]) -> Vec<Spike> {
        spike_data.iter().map(spike_data_to_core).collect()
    }
    
    /// Convert multiple spikes from core
    pub fn spikes_from_core(spikes: &[Spike]) -> Vec<SpikeData> {
        spikes.iter().map(core_to_spike_data).collect()
    }
}

/// Utility functions for FFI operations
pub mod ffi_utils {
    use super::*;
    
    /// Initialize SHNN FFI with default settings
    pub fn init() -> FFIResult<()> {
        AcceleratorRegistry::initialize()
    }
    
    /// Cleanup and shutdown SHNN FFI
    pub fn shutdown() -> FFIResult<()> {
        // Release all accelerators
        let mut registry = ACCELERATOR_REGISTRY.lock().unwrap();
        let accelerator_ids: Vec<_> = registry.list_accelerators();
        
        for id in accelerator_ids {
            registry.unregister_accelerator(id);
        }
        
        Ok(())
    }
    
    /// Get library version information
    pub fn get_version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
    
    /// Get supported features
    pub fn get_supported_features() -> Vec<&'static str> {
        let mut features = vec!["core"];
        
        #[cfg(feature = "cuda")]
        features.push("cuda");
        
        #[cfg(feature = "opencl")]
        features.push("opencl");
        
        #[cfg(feature = "fpga")]
        features.push("fpga");
        
        #[cfg(feature = "rram")]
        features.push("rram");
        
        #[cfg(feature = "intel-loihi")]
        features.push("intel-loihi");
        
        #[cfg(feature = "spiNNaker")]
        features.push("spinnaker");
        
        #[cfg(feature = "python-bindings")]
        features.push("python");
        
        #[cfg(feature = "c-bindings")]
        features.push("c-bindings");
        
        #[cfg(feature = "cpp-bindings")]
        features.push("cpp-bindings");
        
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_accelerator_registry() {
        let mut registry = AcceleratorRegistry::new();
        assert_eq!(registry.list_accelerators().len(), 0);
        
        // Mock accelerator for testing
        struct MockAccelerator;
        impl HardwareAccelerator for MockAccelerator {
            fn deploy_network(&mut self, _config: &NetworkConfig) -> FFIResult<NetworkId> {
                Ok(NetworkId(1))
            }
            
            fn process_spikes(
                &mut self,
                _network_id: NetworkId,
                _input_spikes: &[SpikeData],
            ) -> FFIResult<Vec<SpikeData>> {
                Ok(vec![])
            }
            
            fn get_capabilities(&self) -> HardwareCapabilities {
                HardwareCapabilities::default()
            }
            
            fn get_performance_metrics(&self) -> PerformanceMetrics {
                PerformanceMetrics::default()
            }
        }
        
        let accelerator = Box::new(MockAccelerator);
        let id = registry.register_accelerator(accelerator);
        
        assert_eq!(registry.list_accelerators().len(), 1);
        assert!(registry.get_accelerator(id).is_some());
        
        registry.unregister_accelerator(id);
        assert_eq!(registry.list_accelerators().len(), 0);
    }
    
    #[test]
    fn test_spike_conversion() {
        let spike_data = SpikeData {
            neuron_id: 42,
            timestamp: 1000.0,
            amplitude: 1.5,
        };
        
        let spike = conversion::spike_data_to_core(&spike_data);
        let converted_back = conversion::core_to_spike_data(&spike);
        
        assert_eq!(spike_data.neuron_id, converted_back.neuron_id);
        assert!((spike_data.timestamp - converted_back.timestamp).abs() < 0.001);
        assert!((spike_data.amplitude - converted_back.amplitude).abs() < 0.001);
    }
    
    #[test]
    fn test_library_info() {
        let version = ffi_utils::get_version();
        assert!(!version.is_empty());
        
        let features = ffi_utils::get_supported_features();
        assert!(features.contains(&"core"));
    }
}