//! # SHNN Async
//!
//! Asynchronous processing extensions for Spiking Hypergraph Neural Networks.
//!
//! This crate provides advanced async capabilities for neuromorphic computation,
//! including distributed processing, real-time streaming, and high-performance
//! concurrent spike processing.
//!
//! ## Features
//!
//! - **Async Runtimes**: Support for Tokio and async-std
//! - **Distributed Processing**: Multi-node neural network computation
//! - **Real-time Streaming**: Continuous spike stream processing
//! - **Load Balancing**: Intelligent workload distribution
//! - **Monitoring**: Comprehensive metrics and tracing
//! - **Fault Tolerance**: Resilient processing with error recovery
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use shnn_async::{AsyncNetworkManager, ProcessingConfig};
//! use shnn_core::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = ProcessingConfig::default();
//!     let mut manager = AsyncNetworkManager::new(config).await?;
//!     
//!     // Create and configure network
//!     let network_id = manager.create_network().await?;
//!     
//!     // Process spikes asynchronously
//!     let spike = Spike::binary(NeuronId::new(0), Time::from_millis(1))?;
//!     let results = manager.process_spike(network_id, spike).await?;
//!     
//!     println!("Processed {} output spikes", results.len());
//!     Ok(())
//! }
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Re-export core functionality
pub use shnn_core::{
    error::{Result, SHNNError},
    spike::{NeuronId, Spike, TimedSpike, SpikeTrain},
    time::{Time, Duration},
    neuron::{Neuron, LIFNeuron, LIFConfig},
    hypergraph::{HypergraphNetwork, Hyperedge, HyperedgeId},
    plasticity::{STDPRule, STDPConfig},
    encoding::{SpikeEncoder, RateEncoder, TemporalEncoder},
};

// Core async modules
pub mod runtime;
pub mod network;
pub mod streaming;
pub mod distributed;
pub mod monitoring;
pub mod channels;
pub mod scheduling;
pub mod load_balancing;
pub mod error;

// Re-export commonly used types
pub use crate::{
    runtime::{AsyncRuntime, RuntimeConfig},
    network::{AsyncNetworkManager, NetworkHandle, ProcessingConfig},
    streaming::{SpikeStream, StreamProcessor, StreamConfig},
    distributed::{DistributedNetwork, NodeConfig, ClusterConfig},
    monitoring::{NetworkMonitor, MonitoringConfig, ProcessingMetrics},
    channels::{SpikeChannel, ChannelConfig, MessageType},
    scheduling::{TaskScheduler, SchedulingPolicy, TaskPriority},
    load_balancing::{LoadBalancer, BalancingStrategy, WorkloadMetrics},
    error::{AsyncError, AsyncResult},
};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build configuration
pub const BUILD_CONFIG: BuildConfig = BuildConfig {
    version: VERSION,
    features: &[
        #[cfg(feature = "tokio-runtime")]
        "tokio-runtime",
        #[cfg(feature = "async-std-runtime")]
        "async-std-runtime",
        #[cfg(feature = "flume-channels")]
        "flume-channels",
        #[cfg(feature = "metrics")]
        "metrics",
        #[cfg(feature = "tracing")]
        "tracing",
        #[cfg(feature = "serde")]
        "serde",
        #[cfg(feature = "parallel")]
        "parallel",
    ],
    async_runtime: {
        #[cfg(feature = "tokio-runtime")]
        { "tokio" }
        #[cfg(all(feature = "async-std-runtime", not(feature = "tokio-runtime")))]
        { "async-std" }
        #[cfg(not(any(feature = "tokio-runtime", feature = "async-std-runtime")))]
        { "none" }
    },
};

/// Build configuration information
#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Crate version
    pub version: &'static str,
    /// Enabled features
    pub features: &'static [&'static str],
    /// Selected async runtime
    pub async_runtime: &'static str,
}

impl BuildConfig {
    /// Get a formatted build string
    pub fn build_string(&self) -> String {
        format!(
            "SHNN-Async v{} (runtime: {}, features: [{}])",
            self.version,
            self.async_runtime,
            self.features.join(", ")
        )
    }
    
    /// Check if a feature is enabled
    pub fn has_feature(&self, feature: &str) -> bool {
        self.features.contains(&feature)
    }
}

/// Global initialization for SHNN-Async
///
/// This function should be called once at the beginning of your application
/// to set up logging, metrics, and other global state.
pub async fn init() -> AsyncResult<()> {
    init_with_config(InitConfig::default()).await
}

/// Initialize with custom configuration
pub async fn init_with_config(config: InitConfig) -> AsyncResult<()> {
    // Initialize core SHNN
    shnn_core::init_with_config(shnn_core::InitConfig {
        log_level: config.log_level,
        enable_logging: config.enable_logging,
        print_build_info: false, // We'll print our own
    })?;
    
    if config.print_build_info {
        #[cfg(feature = "std")]
        println!("{}", BUILD_CONFIG.build_string());
    }
    
    Ok(())
}

/// Configuration for library initialization
#[derive(Debug, Clone)]
pub struct InitConfig {
    /// Log level to set
    pub log_level: Option<&'static str>,
    /// Whether to enable logging
    pub enable_logging: bool,
    /// Whether to print build information
    pub print_build_info: bool,
}

impl Default for InitConfig {
    fn default() -> Self {
        Self {
            log_level: Some("info"),
            enable_logging: true,
            print_build_info: true,
        }
    }
}

/// Prelude module for common imports
pub mod prelude {
    //! Common imports for SHNN-Async users
    
    // Re-export core prelude
    pub use shnn_core::prelude::*;
    
    // Async-specific exports
    pub use crate::{
        AsyncNetworkManager, ProcessingConfig,
        SpikeStream, StreamProcessor, StreamConfig,
        DistributedNetwork, NodeConfig, ClusterConfig,
        NetworkMonitor, MonitoringConfig,
        SpikeChannel, ChannelConfig,
        TaskScheduler, SchedulingPolicy,
        LoadBalancer, BalancingStrategy,
        AsyncError, AsyncResult,
        init, init_with_config, InitConfig,
    };
    
    // Runtime selection
    #[cfg(feature = "tokio-runtime")]
    pub use tokio;
    
    #[cfg(feature = "async-std-runtime")]
    pub use async_std;
    
    // Common async utilities
    pub use futures::prelude::*;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_build_config() {
        let config = BUILD_CONFIG;
        assert!(!config.version.is_empty());
        assert!(!config.async_runtime.is_empty());
        
        let build_string = config.build_string();
        assert!(build_string.contains("SHNN-Async"));
        assert!(build_string.contains(config.version));
    }
    
    #[tokio::test]
    async fn test_init() {
        let result = init().await;
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_version_constant() {
        assert!(!VERSION.is_empty());
        assert!(VERSION.chars().any(|c| c.is_ascii_digit()));
    }
}