//! Data types for FFI operations
//!
//! This module defines C-compatible data structures and enums used for
//! communication between Rust and other languages/hardware platforms.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Hardware accelerator types supported by SHNN
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AcceleratorType {
    /// CPU-based software implementation
    CPU = 0,
    /// NVIDIA CUDA GPU acceleration
    CUDA = 1,
    /// OpenCL cross-platform acceleration
    OpenCL = 2,
    /// FPGA implementation
    FPGA = 3,
    /// RRAM/Memristor-based neuromorphic hardware
    RRAM = 4,
    /// Intel Loihi neuromorphic research chip
    IntelLoihi = 5,
    /// SpiNNaker large-scale neuromorphic platform
    SpiNNaker = 6,
    /// Custom hardware accelerator
    Custom = 99,
}

impl AcceleratorType {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::CPU => "CPU",
            Self::CUDA => "CUDA",
            Self::OpenCL => "OpenCL",
            Self::FPGA => "FPGA",
            Self::RRAM => "RRAM",
            Self::IntelLoihi => "Intel Loihi",
            Self::SpiNNaker => "SpiNNaker",
            Self::Custom => "Custom",
        }
    }
    
    /// Check if this accelerator type supports specific features
    pub fn supports_feature(&self, feature: AcceleratorFeature) -> bool {
        match (self, feature) {
            (Self::CPU, _) => true, // CPU supports all features in software
            (Self::CUDA, AcceleratorFeature::ParallelProcessing) => true,
            (Self::CUDA, AcceleratorFeature::FloatingPoint) => true,
            (Self::CUDA, AcceleratorFeature::LargeMemory) => true,
            (Self::OpenCL, AcceleratorFeature::ParallelProcessing) => true,
            (Self::OpenCL, AcceleratorFeature::CrossPlatform) => true,
            (Self::FPGA, AcceleratorFeature::LowLatency) => true,
            (Self::FPGA, AcceleratorFeature::CustomLogic) => true,
            (Self::FPGA, AcceleratorFeature::LowPower) => true,
            (Self::RRAM, AcceleratorFeature::InMemoryCompute) => true,
            (Self::RRAM, AcceleratorFeature::LowPower) => true,
            (Self::RRAM, AcceleratorFeature::NonVolatile) => true,
            (Self::IntelLoihi, AcceleratorFeature::EventDriven) => true,
            (Self::IntelLoihi, AcceleratorFeature::LowPower) => true,
            (Self::IntelLoihi, AcceleratorFeature::OnChipLearning) => true,
            (Self::SpiNNaker, AcceleratorFeature::MassivelyParallel) => true,
            (Self::SpiNNaker, AcceleratorFeature::EventDriven) => true,
            (Self::SpiNNaker, AcceleratorFeature::ScalableArchitecture) => true,
            _ => false,
        }
    }
}

/// Hardware accelerator features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceleratorFeature {
    /// Parallel processing capabilities
    ParallelProcessing,
    /// Cross-platform support
    CrossPlatform,
    /// Low latency execution
    LowLatency,
    /// Custom logic implementation
    CustomLogic,
    /// Low power consumption
    LowPower,
    /// In-memory computation
    InMemoryCompute,
    /// Non-volatile storage
    NonVolatile,
    /// Event-driven processing
    EventDriven,
    /// On-chip learning capabilities
    OnChipLearning,
    /// Massively parallel architecture
    MassivelyParallel,
    /// Scalable architecture
    ScalableArchitecture,
    /// Floating-point support
    FloatingPoint,
    /// Large memory capacity
    LargeMemory,
}

/// Network configuration for hardware deployment
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Number of neurons in the network
    pub num_neurons: u32,
    /// Number of hyperedges/connections
    pub num_connections: u32,
    /// Connectivity density (0.0 to 1.0)
    pub connectivity: f32,
    /// Simulation time step in milliseconds
    pub dt: f32,
    /// Input layer size
    pub input_size: u32,
    /// Output layer size
    pub output_size: u32,
    /// Hidden layer sizes
    pub hidden_layers: Vec<u32>,
    /// Network topology type
    pub topology: NetworkTopology,
    /// Neuron model parameters
    pub neuron_config: NeuronConfig,
    /// Plasticity configuration
    pub plasticity_config: PlasticityConfig,
    /// Memory allocation hints
    pub memory_hints: MemoryHints,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            num_neurons: 1000,
            num_connections: 10000,
            connectivity: 0.1,
            dt: 0.001,
            input_size: 100,
            output_size: 10,
            hidden_layers: vec![500, 200],
            topology: NetworkTopology::Feedforward,
            neuron_config: NeuronConfig::default(),
            plasticity_config: PlasticityConfig::default(),
            memory_hints: MemoryHints::default(),
        }
    }
}

/// Network topology types
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkTopology {
    /// Feedforward network
    Feedforward = 0,
    /// Recurrent network
    Recurrent = 1,
    /// Convolutional spiking network
    Convolutional = 2,
    /// Small-world network
    SmallWorld = 3,
    /// Scale-free network
    ScaleFree = 4,
    /// Custom topology
    Custom = 99,
}

/// Neuron model configuration
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronConfig {
    /// Neuron model type
    pub model_type: NeuronModelType,
    /// Membrane time constant (ms)
    pub tau_m: f32,
    /// Resting potential (mV)
    pub v_rest: f32,
    /// Threshold potential (mV)
    pub v_thresh: f32,
    /// Reset potential (mV)
    pub v_reset: f32,
    /// Refractory period (ms)
    pub tau_ref: f32,
    /// Additional model-specific parameters
    pub model_params: HashMap<String, f32>,
}

impl Default for NeuronConfig {
    fn default() -> Self {
        Self {
            model_type: NeuronModelType::LIF,
            tau_m: 20.0,
            v_rest: -70.0,
            v_thresh: -55.0,
            v_reset: -75.0,
            tau_ref: 2.0,
            model_params: HashMap::new(),
        }
    }
}

/// Neuron model types
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronModelType {
    /// Leaky Integrate-and-Fire
    LIF = 0,
    /// Adaptive Exponential Integrate-and-Fire
    AdEx = 1,
    /// Izhikevich model
    Izhikevich = 2,
    /// Hodgkin-Huxley model
    HodgkinHuxley = 3,
    /// Custom neuron model
    Custom = 99,
}

/// Plasticity configuration
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityConfig {
    /// Enable/disable plasticity
    pub enabled: bool,
    /// Plasticity rule type
    pub rule_type: PlasticityRuleType,
    /// Learning rate
    pub learning_rate: f32,
    /// STDP time window (ms)
    pub stdp_window: f32,
    /// Weight bounds
    pub weight_min: f32,
    pub weight_max: f32,
    /// Plasticity-specific parameters
    pub rule_params: HashMap<String, f32>,
}

impl Default for PlasticityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rule_type: PlasticityRuleType::STDP,
            learning_rate: 0.01,
            stdp_window: 20.0,
            weight_min: 0.0,
            weight_max: 1.0,
            rule_params: HashMap::new(),
        }
    }
}

/// Plasticity rule types
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlasticityRuleType {
    /// Spike-Timing Dependent Plasticity
    STDP = 0,
    /// Rate-based Hebbian learning
    Hebbian = 1,
    /// Homeostatic plasticity
    Homeostatic = 2,
    /// Custom plasticity rule
    Custom = 99,
}

/// Memory allocation hints for hardware optimization
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHints {
    /// Preferred memory location
    pub preferred_location: MemoryLocation,
    /// Memory access pattern
    pub access_pattern: MemoryAccessPattern,
    /// Expected memory usage (bytes)
    pub expected_usage: u64,
    /// Memory alignment requirements
    pub alignment: u32,
    /// Enable memory pooling
    pub use_pooling: bool,
}

impl Default for MemoryHints {
    fn default() -> Self {
        Self {
            preferred_location: MemoryLocation::Device,
            access_pattern: MemoryAccessPattern::Sequential,
            expected_usage: 1024 * 1024, // 1MB default
            alignment: 256,
            use_pooling: true,
        }
    }
}

/// Memory location preferences
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryLocation {
    /// Host/system memory
    Host = 0,
    /// Device memory (GPU/accelerator)
    Device = 1,
    /// Unified memory (accessible by both)
    Unified = 2,
}

/// Memory access patterns
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryAccessPattern {
    /// Sequential access
    Sequential = 0,
    /// Random access
    Random = 1,
    /// Streaming access
    Streaming = 2,
    /// Coalesced access
    Coalesced = 3,
}

/// Spike data structure for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpikeData {
    /// Source neuron ID
    pub neuron_id: u32,
    /// Spike timestamp (ms)
    pub timestamp: f64,
    /// Spike amplitude
    pub amplitude: f32,
}

impl SpikeData {
    /// Create a new spike
    pub fn new(neuron_id: u32, timestamp: f64, amplitude: f32) -> Self {
        Self {
            neuron_id,
            timestamp,
            amplitude,
        }
    }
    
    /// Create a binary spike (amplitude = 1.0)
    pub fn binary(neuron_id: u32, timestamp: f64) -> Self {
        Self::new(neuron_id, timestamp, 1.0)
    }
}

/// Neuron state for monitoring and debugging
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NeuronState {
    /// Neuron ID
    pub neuron_id: u32,
    /// Current membrane potential (mV)
    pub membrane_potential: f32,
    /// Last spike time (ms)
    pub last_spike_time: f64,
    /// Spike count
    pub spike_count: u32,
    /// Refractory state
    pub is_refractory: bool,
    /// Input current (nA)
    pub input_current: f32,
}

/// Hardware capabilities description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    /// Accelerator type
    pub accelerator_type: AcceleratorType,
    /// Maximum number of neurons supported
    pub max_neurons: u32,
    /// Maximum number of connections
    pub max_connections: u64,
    /// Available memory (bytes)
    pub memory_size: u64,
    /// Processing units (cores, CUs, etc.)
    pub processing_units: u32,
    /// Supported neuron models
    pub supported_models: Vec<NeuronModelType>,
    /// Supported features
    pub features: Vec<String>,
    /// Performance characteristics
    pub performance: PerformanceCharacteristics,
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self {
            accelerator_type: AcceleratorType::CPU,
            max_neurons: 1_000_000,
            max_connections: 10_000_000,
            memory_size: 8 * 1024 * 1024 * 1024, // 8GB
            processing_units: 8,
            supported_models: vec![
                NeuronModelType::LIF,
                NeuronModelType::AdEx,
                NeuronModelType::Izhikevich,
            ],
            features: vec![
                "parallel_processing".to_string(),
                "floating_point".to_string(),
                "large_memory".to_string(),
            ],
            performance: PerformanceCharacteristics::default(),
        }
    }
}

/// Performance characteristics of hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristics {
    /// Peak operations per second
    pub peak_ops_per_sec: f64,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
    /// Latency per spike (microseconds)
    pub spike_latency_us: f64,
    /// Power consumption (watts)
    pub power_consumption: f32,
    /// Thermal design power (watts)
    pub tdp: f32,
}

impl Default for PerformanceCharacteristics {
    fn default() -> Self {
        Self {
            peak_ops_per_sec: 1e12, // 1 TOPS
            memory_bandwidth: 100.0, // 100 GB/s
            spike_latency_us: 1.0,
            power_consumption: 50.0,
            tdp: 100.0,
        }
    }
}

/// Runtime performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total execution time (ms)
    pub execution_time_ms: f64,
    /// Spikes processed per second
    pub spikes_per_second: f64,
    /// Memory usage (bytes)
    pub memory_usage: u64,
    /// Power consumption (watts)
    pub power_consumption: f32,
    /// GPU utilization (0.0 to 1.0)
    pub gpu_utilization: f32,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f32,
    /// Network accuracy (if applicable)
    pub accuracy: Option<f32>,
    /// Error counts by type
    pub error_counts: HashMap<String, u32>,
    /// Timing breakdown
    pub timing_breakdown: TimingBreakdown,
}

/// Detailed timing breakdown
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingBreakdown {
    /// Network initialization time (ms)
    pub initialization_ms: f64,
    /// Spike processing time (ms)
    pub processing_ms: f64,
    /// Memory transfer time (ms)
    pub memory_transfer_ms: f64,
    /// Plasticity update time (ms)
    pub plasticity_update_ms: f64,
    /// Synchronization time (ms)
    pub synchronization_ms: f64,
}

/// Execution context for hardware operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    /// Batch size for processing
    pub batch_size: u32,
    /// Number of simulation steps
    pub num_steps: u32,
    /// Parallel execution streams
    pub num_streams: u32,
    /// Memory optimization level
    pub optimization_level: OptimizationLevel,
    /// Profiling enabled
    pub enable_profiling: bool,
    /// Debug mode
    pub debug_mode: bool,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            num_steps: 1000,
            num_streams: 1,
            optimization_level: OptimizationLevel::Balanced,
            enable_profiling: false,
            debug_mode: false,
        }
    }
}

/// Optimization levels for different use cases
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// Debug mode (slow but comprehensive checks)
    Debug = 0,
    /// Balanced performance and memory usage
    Balanced = 1,
    /// Maximum performance (may use more memory)
    Performance = 2,
    /// Minimum memory usage (may be slower)
    Memory = 3,
    /// Low power consumption
    Power = 4,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_accelerator_type_features() {
        assert!(AcceleratorType::CUDA.supports_feature(AcceleratorFeature::ParallelProcessing));
        assert!(AcceleratorType::FPGA.supports_feature(AcceleratorFeature::LowLatency));
        assert!(AcceleratorType::RRAM.supports_feature(AcceleratorFeature::InMemoryCompute));
        assert!(!AcceleratorType::CPU.supports_feature(AcceleratorFeature::CustomLogic));
    }
    
    #[test]
    fn test_network_config_default() {
        let config = NetworkConfig::default();
        assert_eq!(config.num_neurons, 1000);
        assert_eq!(config.topology, NetworkTopology::Feedforward);
        assert_eq!(config.neuron_config.model_type, NeuronModelType::LIF);
    }
    
    #[test]
    fn test_spike_data_creation() {
        let spike = SpikeData::new(42, 1.5, 0.8);
        assert_eq!(spike.neuron_id, 42);
        assert_eq!(spike.timestamp, 1.5);
        assert_eq!(spike.amplitude, 0.8);
        
        let binary_spike = SpikeData::binary(24, 2.0);
        assert_eq!(binary_spike.neuron_id, 24);
        assert_eq!(binary_spike.amplitude, 1.0);
    }
    
    #[test]
    fn test_serialization() {
        let config = NetworkConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: NetworkConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(config.num_neurons, deserialized.num_neurons);
    }
    
    #[test]
    fn test_hardware_capabilities() {
        let caps = HardwareCapabilities::default();
        assert_eq!(caps.accelerator_type, AcceleratorType::CPU);
        assert!(caps.max_neurons > 0);
        assert!(!caps.supported_models.is_empty());
    }
}