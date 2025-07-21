//! # SHNN Core
//!
//! Core neuromorphic primitives for Spiking Hypergraph Neural Networks.
//!
//! This crate provides the fundamental building blocks for neuromorphic computing:
//!
//! - **Neuron Models**: Biologically realistic neuron implementations
//! - **Hypergraph Structures**: Multi-synaptic connection representations
//! - **Spike Processing**: Event-driven spike routing and processing
//! - **Plasticity Mechanisms**: Learning rules and synaptic adaptation
//! - **Temporal Encoding**: Information encoding in spike timing patterns
//!
//! ## Quick Start
//!
//! ```rust
//! use shnn_core::prelude::*;
//!
//! // Create a basic neuron
//! let mut neuron = LIFNeuron::new(NeuronId(0), LIFConfig::default());
//!
//! // Process a spike
//! let spike = Spike {
//!     source: NeuronId(1),
//!     timestamp: 1000,
//!     amplitude: 1.0,
//! };
//!
//! if let Some(output_spike) = neuron.process_spike(&spike, 1000) {
//!     println!("Neuron fired at time {}", output_spike.timestamp);
//! }
//! ```
//!
//! ## Feature Flags
//!
//! - `std` (default): Enable standard library support
//! - `no-std`: Enable no-std embedded support
//! - `async`: Enable asynchronous processing capabilities
//! - `math`: Enable advanced mathematical operations
//! - `serde`: Enable serialization support
//! - `simd`: Enable SIMD optimizations
//! - `parallel`: Enable parallel processing
//! - `hardware-accel`: Enable hardware acceleration support

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::type_complexity)]

// Re-export important types for convenience
pub use crate::{
    neuron::{Neuron, NeuronId, NeuronType},
    spike::{Spike, SpikeTarget, TimedSpike},
    hypergraph::{Hyperedge, HyperedgeId, HypergraphNetwork},
    plasticity::{PlasticityRule, STDPRule},
    time::Time,
    error::{SHNNError, Result},
};

// Core modules
pub mod neuron;
pub mod spike;
pub mod hypergraph;
pub mod plasticity;
pub mod encoding;
pub mod time;
pub mod memory;
pub mod error;

// Conditional modules based on features
#[cfg(feature = "async")]
pub mod async_processing;

#[cfg(feature = "math")]
pub mod math;

#[cfg(feature = "serde")]
pub mod serialization;

#[cfg(test)]
pub mod test_helpers;

// Prelude module for common imports
pub mod prelude {
    //! Common imports for SHNN users
    
    pub use crate::{
        neuron::*,
        spike::*,
        hypergraph::*,
        plasticity::*,
        encoding::*,
        time::*,
        error::*,
    };
    
    #[cfg(feature = "async")]
    pub use crate::async_processing::*;
    
    #[cfg(feature = "math")]
    pub use crate::math::*;
}

// Platform-specific imports
#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate core as std;

// Version information
/// The version of this crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build information
pub const BUILD_INFO: BuildInfo = BuildInfo {
    version: VERSION,
    git_hash: option_env!("GIT_HASH"),
    build_timestamp: match option_env!("BUILD_TIMESTAMP") {
        Some(ts) => ts,
        None => "unknown",
    },
    features: &[
        #[cfg(feature = "std")]
        "std",
        #[cfg(feature = "no-std")]
        "no-std",
        #[cfg(feature = "async")]
        "async",
        #[cfg(feature = "math")]
        "math",
        #[cfg(feature = "serde")]
        "serde",
        #[cfg(feature = "simd")]
        "simd",
        #[cfg(feature = "parallel")]
        "parallel",
        #[cfg(feature = "hardware-accel")]
        "hardware-accel",
    ],
};

/// Build information structure
#[derive(Debug, Clone)]
pub struct BuildInfo {
    /// Version string
    pub version: &'static str,
    /// Git commit hash
    pub git_hash: Option<&'static str>,
    /// Build timestamp
    pub build_timestamp: &'static str,
    /// Enabled features
    pub features: &'static [&'static str],
}

impl BuildInfo {
    /// Get a formatted build string
    pub fn build_string(&self) -> String {
        format!(
            "SHNN Core v{} ({}), built on {} with features: [{}]",
            self.version,
            self.git_hash.unwrap_or("unknown"),
            self.build_timestamp,
            self.features.join(", ")
        )
    }
}

/// Initialize the SHNN Core library
///
/// This function should be called once at the beginning of your application.
/// It sets up logging and any other global state required by the library.
pub fn init() -> Result<()> {
    #[cfg(feature = "std")]
    {
        // Initialize logging if available
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", "info");
        }
        env_logger::try_init().ok();
        
        log::info!("{}", BUILD_INFO.build_string());
    }
    
    Ok(())
}

/// Initialize the library with custom configuration
pub fn init_with_config(config: InitConfig) -> Result<()> {
    #[cfg(feature = "std")]
    {
        if let Some(log_level) = config.log_level {
            std::env::set_var("RUST_LOG", log_level);
        }
        
        if config.enable_logging {
            env_logger::try_init().ok();
        }
        
        if config.print_build_info {
            log::info!("{}", BUILD_INFO.build_string());
        }
    }
    
    Ok(())
}

/// Configuration for library initialization
#[derive(Debug, Default)]
pub struct InitConfig {
    /// Log level to set (e.g., "debug", "info", "warn", "error")
    pub log_level: Option<&'static str>,
    /// Whether to enable logging
    pub enable_logging: bool,
    /// Whether to print build information
    pub print_build_info: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_build_info() {
        let build_string = BUILD_INFO.build_string();
        assert!(build_string.contains("SHNN Core"));
        assert!(build_string.contains(VERSION));
    }
    
    #[test]
    fn test_init() {
        assert!(init().is_ok());
    }
    
    #[test]
    fn test_init_with_config() {
        let config = InitConfig {
            log_level: Some("debug"),
            enable_logging: true,
            print_build_info: true,
        };
        assert!(init_with_config(config).is_ok());
    }
}