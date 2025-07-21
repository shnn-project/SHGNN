//! # SHNN Embedded
//!
//! No-std implementation of Spiking Hypergraph Neural Networks for embedded systems.
//!
//! This crate provides a lightweight, deterministic implementation of neuromorphic
//! computing primitives optimized for microcontrollers and real-time systems.
//!
//! ## Features
//!
//! - **No-std Compatible**: Runs on bare metal without heap allocation
//! - **Real-time Deterministic**: Guaranteed execution times for real-time systems
//! - **Memory Efficient**: Optimized for constrained memory environments
//! - **Fixed-point Arithmetic**: Deterministic computation without floating point
//! - **Hardware Abstraction**: Support for various embedded platforms
//! - **RTIC Integration**: Real-Time Interrupt-driven Concurrency support
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! #![no_std]
//! #![no_main]
//! 
//! use shnn_embedded::{EmbeddedNetwork, FixedNeuron, FixedSpike};
//! use heapless::Vec;
//! 
//! // Create a small network suitable for microcontrollers
//! let mut network: EmbeddedNetwork<16, 32> = EmbeddedNetwork::new();
//! 
//! // Add neurons with fixed-point configuration
//! let neuron = FixedNeuron::new(0, Default::default());
//! network.add_neuron(neuron).unwrap();
//! 
//! // Process spikes deterministically
//! let spike = FixedSpike::new(1, 1000, 1.0);
//! let outputs = network.process_spike(spike).unwrap();
//! ```

#![no_std]
#![deny(missing_docs)]
#![warn(clippy::all)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Optional standard library support for testing
#[cfg(any(test, feature = "std"))]
extern crate std;

// Optional allocator support
#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

// Re-export core functionality (no-std compatible)
pub use shnn_core::{
    spike::NeuronId,
    time::{Time, Duration},
    error::SHNNError,
};

// Error handling
pub mod error;

// Core embedded modules
pub mod fixed_point;
pub mod embedded_neuron;
pub mod embedded_network;
pub mod embedded_memory;
pub mod hal;

// Optional modules based on features
#[cfg(feature = "rtic")]
pub mod rtic_support;

// Hardware-specific optimizations (future extensions)
#[cfg(feature = "arm-math")]
pub mod arm_optimizations;

#[cfg(feature = "risc-v")]
pub mod riscv_optimizations;

// Re-export commonly used types
pub use crate::{
    error::{EmbeddedError, EmbeddedResult},
    fixed_point::{FixedPoint, Q16_16, FixedSpike},
    embedded_neuron::{
        EmbeddedNeuron, EmbeddedLIFNeuron, EmbeddedIzhikevichNeuron,
        EmbeddedSynapse, EmbeddedNeuronPopulation
    },
    embedded_network::{
        EmbeddedNetwork, EmbeddedSNN, EmbeddedTopology, EmbeddedNetworkBuilder,
        NetworkStatistics, EmbeddedNeuronWrapper
    },
    embedded_memory::{
        EmbeddedHypergraph, EmbeddedSpikeBuffer, EmbeddedWeightMatrix,
        EmbeddedTimeSeriesBuffer, HyperedgeType
    },
    hal::{
        EmbeddedHAL, HardwareTimer, GpioPin, AnalogToDigital, PulseWidthModulation,
        HALFactory, PlatformInfo
    },
};

#[cfg(feature = "rtic")]
pub use crate::rtic_support::{
    RTICScheduler, RTICTaskConfig, RTICTaskType, RTICStatistics,
    RTConstraints, InterruptConfig
};

// Version information
/// The version of this crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build configuration for embedded targets
#[derive(Debug, Clone)]
pub struct EmbeddedBuildConfig {
    /// Target architecture
    pub target_arch: &'static str,
    /// Enabled features
    pub features: &'static [&'static str],
    /// Memory constraints
    pub memory_config: MemoryConfig,
}

/// Memory configuration for embedded systems
#[derive(Debug, Clone, Copy)]
pub struct MemoryConfig {
    /// Maximum number of neurons
    pub max_neurons: usize,
    /// Maximum number of connections
    pub max_connections: usize,
    /// Spike buffer size
    pub spike_buffer_size: usize,
    /// Stack size in bytes
    pub stack_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_neurons: 64,
            max_connections: 256,
            spike_buffer_size: 128,
            stack_size: 4096,
        }
    }
}

/// Default build configuration
pub const BUILD_CONFIG: EmbeddedBuildConfig = EmbeddedBuildConfig {
    target_arch: env!("CARGO_CFG_TARGET_ARCH"),
    features: &[
        #[cfg(feature = "cortex-m")]
        "cortex-m",
        #[cfg(feature = "rtic")]
        "rtic",
        #[cfg(feature = "alloc")]
        "alloc",
        #[cfg(feature = "serde")]
        "serde",
        #[cfg(feature = "fixed-point")]
        "fixed-point",
        #[cfg(feature = "arm-math")]
        "arm-math",
        #[cfg(feature = "risc-v")]
        "risc-v",
        #[cfg(feature = "real-time")]
        "real-time",
    ],
    memory_config: MemoryConfig {
        max_neurons: 64,
        max_connections: 256,
        spike_buffer_size: 128,
        stack_size: 4096,
    },
};

/// Initialize the embedded SHNN library
///
/// This function should be called once during system initialization.
/// It sets up memory management, timers, and other hardware resources.
pub fn init() -> EmbeddedResult<()> {
    init_with_config(EmbeddedInitConfig::default())
}

/// Initialize with custom configuration
pub fn init_with_config(config: EmbeddedInitConfig) -> EmbeddedResult<()> {
    // Initialize memory management (basic validation)
    if config.init_memory {
        if config.memory_config.max_neurons == 0 {
            return Err(EmbeddedError::InvalidConfiguration);
        }
    }
    
    // Initialize hardware abstraction layer
    if config.init_hal {
        // HAL initialization would be platform-specific
        // For now, just validate the configuration
        if config.memory_config.stack_size < 1024 {
            return Err(EmbeddedError::InvalidConfiguration);
        }
    }
    
    // Initialize timers (basic validation)
    if config.init_timers {
        // Timer initialization would be platform-specific
        // For now, just validate the configuration
        if config.memory_config.spike_buffer_size == 0 {
            return Err(EmbeddedError::InvalidConfiguration);
        }
    }
    
    Ok(())
}

/// Configuration for embedded initialization
#[derive(Debug, Clone)]
pub struct EmbeddedInitConfig {
    /// Initialize memory management
    pub init_memory: bool,
    /// Initialize hardware abstraction layer
    pub init_hal: bool,
    /// Initialize timers
    pub init_timers: bool,
    /// Memory configuration
    pub memory_config: MemoryConfig,
}

impl Default for EmbeddedInitConfig {
    fn default() -> Self {
        Self {
            init_memory: true,
            init_hal: true,
            init_timers: true,
            memory_config: MemoryConfig::default(),
        }
    }
}

/// Panic handler for no-std environments
#[cfg(not(test))]
#[panic_handler]
fn panic_handler(_info: &core::panic::PanicInfo) -> ! {
    // In a real implementation, this would:
    // 1. Log the panic information
    // 2. Safely shutdown the neural network
    // 3. Reset the system or enter a safe state
    
    loop {
        // Wait for watchdog reset or manual intervention
        core::hint::spin_loop();
    }
}

/// Critical section implementation for embedded systems
pub mod critical_section {
    use core::sync::atomic::{AtomicBool, Ordering};
    
    static CRITICAL_SECTION: AtomicBool = AtomicBool::new(false);
    
    /// Enter critical section (disable interrupts)
    pub fn enter() -> CriticalSection {
        // Disable interrupts
        #[cfg(feature = "cortex-m")]
        {
            cortex_m::interrupt::disable();
        }
        
        CRITICAL_SECTION.store(true, Ordering::SeqCst);
        CriticalSection
    }
    
    /// Critical section guard
    pub struct CriticalSection;
    
    impl Drop for CriticalSection {
        fn drop(&mut self) {
            CRITICAL_SECTION.store(false, Ordering::SeqCst);
            
            // Re-enable interrupts
            #[cfg(feature = "cortex-m")]
            unsafe {
                cortex_m::interrupt::enable();
            }
        }
    }
    
    /// Check if currently in critical section
    pub fn is_critical() -> bool {
        CRITICAL_SECTION.load(Ordering::SeqCst)
    }
}

/// Utility macros for embedded development
#[macro_export]
macro_rules! embedded_assert {
    ($cond:expr) => {
        if !$cond {
            // In embedded systems, we might want to handle assertions
            // differently than panicking
            #[cfg(debug_assertions)]
            panic!("Assertion failed: {}", stringify!($cond));
        }
    };
    ($cond:expr, $msg:expr) => {
        if !$cond {
            #[cfg(debug_assertions)]
            panic!("Assertion failed: {}: {}", stringify!($cond), $msg);
        }
    };
}

/// Compile-time configuration validation
#[macro_export]
macro_rules! validate_config {
    ($config:expr, $field:ident, $max:expr) => {
        const _: () = {
            if $config.$field > $max {
                panic!(concat!(
                    "Configuration error: ",
                    stringify!($field),
                    " exceeds maximum value of ",
                    stringify!($max)
                ));
            }
        };
    };
}

// Compile-time validation of default configuration
validate_config!(BUILD_CONFIG.memory_config, max_neurons, 1024);
validate_config!(BUILD_CONFIG.memory_config, max_connections, 4096);
validate_config!(BUILD_CONFIG.memory_config, spike_buffer_size, 1024);

/// Common constants for embedded systems
pub mod constants {
    /// Maximum number of neurons in embedded systems
    pub const MAX_NEURONS: usize = 128;
    
    /// Maximum number of synapses per neuron
    pub const MAX_SYNAPSES_PER_NEURON: usize = 32;
    
    /// Maximum spike buffer size
    pub const MAX_SPIKE_BUFFER: usize = 256;
    
    /// Maximum number of hyperedges
    pub const MAX_HYPEREDGES: usize = 128;
    
    /// Maximum nodes per hyperedge
    pub const MAX_NODES_PER_EDGE: usize = 8;
    
    /// Default fixed-point scale (Q16.16)
    pub const FIXED_POINT_SCALE: i32 = 1 << 16;
}

/// Prelude module for embedded SHNN development
pub mod prelude {
    //! Common imports for embedded SHNN development
    
    pub use crate::{
        EmbeddedError, EmbeddedResult,
        FixedPoint, Q16_16, FixedSpike,
        EmbeddedNeuron, EmbeddedLIFNeuron,
        EmbeddedNetwork, EmbeddedSNN, EmbeddedTopology,
        EmbeddedNetworkBuilder,
        EmbeddedHypergraph, EmbeddedSpikeBuffer,
        MemoryConfig, EmbeddedInitConfig,
        init, init_with_config,
        embedded_assert, validate_config,
        constants::*,
    };
    
    pub use heapless::{Vec, FnvIndexMap, FnvIndexSet};
    
    #[cfg(feature = "rtic")]
    pub use crate::rtic_support::{RTICScheduler, RTICTaskType};
    
    #[cfg(feature = "cortex-m")]
    pub use cortex_m;
    
    // Re-export essential types from core
    pub use shnn_core::{
        spike::NeuronId,
        time::{Time, Duration},
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_build_config() {
        let config = BUILD_CONFIG;
        assert!(!config.target_arch.is_empty());
        assert!(config.memory_config.max_neurons > 0);
        assert!(config.memory_config.max_connections > 0);
    }
    
    #[test]
    fn test_memory_config_validation() {
        let config = MemoryConfig::default();
        assert!(config.max_neurons <= 1024);
        assert!(config.max_connections <= 4096);
        assert!(config.spike_buffer_size <= 1024);
    }
    
    #[test]
    fn test_init_config() {
        let config = EmbeddedInitConfig::default();
        assert!(config.init_memory);
        assert!(config.init_hal);
        assert!(config.init_timers);
    }
    
    #[test]
    fn test_critical_section() {
        assert!(!critical_section::is_critical());
        
        {
            let _guard = critical_section::enter();
            assert!(critical_section::is_critical());
        }
        
        assert!(!critical_section::is_critical());
    }
    
    #[test]
    fn test_embedded_assert() {
        embedded_assert!(true);
        embedded_assert!(1 + 1 == 2, "Math is broken");
        
        // These would panic in debug mode:
        // embedded_assert!(false);
        // embedded_assert!(1 + 1 == 3, "This should fail");
    }
}