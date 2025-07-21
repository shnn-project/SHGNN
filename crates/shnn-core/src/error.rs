//! Error types and handling for SHNN Core
//!
//! This module provides comprehensive error handling for all neuromorphic operations,
//! designed to work in both std and no-std environments.

use core::fmt;

#[cfg(feature = "std")]
use std::error::Error as StdError;

/// Result type alias for SHNN operations
pub type Result<T> = core::result::Result<T, SHNNError>;

/// Main error type for SHNN operations
#[derive(Debug, Clone, PartialEq)]
pub enum SHNNError {
    /// Invalid neuron configuration
    InvalidNeuronConfig {
        /// The neuron ID that had invalid configuration
        neuron_id: u32,
        /// Description of the configuration issue
        reason: &'static str,
    },
    
    /// Invalid spike parameters
    InvalidSpike {
        /// Description of the spike issue
        reason: &'static str,
    },
    
    /// Hypergraph structure errors
    HypergraphError {
        /// Type of hypergraph error
        kind: HypergraphErrorKind,
        /// Additional context
        context: &'static str,
    },
    
    /// Plasticity rule errors
    PlasticityError {
        /// Description of the plasticity issue
        reason: &'static str,
    },
    
    /// Time-related errors
    TimeError {
        /// Description of the time issue
        reason: &'static str,
    },
    
    /// Memory allocation errors (no-std compatible)
    MemoryError {
        /// Description of the memory issue
        reason: &'static str,
    },
    
    /// Encoding/decoding errors
    EncodingError {
        /// Description of the encoding issue
        reason: &'static str,
    },
    
    /// Hardware acceleration errors
    #[cfg(feature = "hardware-accel")]
    HardwareError {
        /// Description of the hardware issue
        reason: &'static str,
    },
    
    /// Serialization errors
    #[cfg(feature = "serde")]
    SerializationError {
        /// Description of the serialization issue
        reason: String,
    },
    
    /// Async processing errors
    #[cfg(feature = "async")]
    AsyncError {
        /// Description of the async issue
        reason: &'static str,
    },
    
    /// Generic error for cases not covered by specific variants
    Generic {
        /// Error message
        message: &'static str,
    },
}

/// Specific hypergraph error kinds
#[derive(Debug, Clone, PartialEq)]
pub enum HypergraphErrorKind {
    /// Node not found in the hypergraph
    NodeNotFound,
    /// Hyperedge not found in the hypergraph
    HyperedgeNotFound,
    /// Invalid hyperedge structure
    InvalidHyperedge,
    /// Cycle detected in the hypergraph
    CycleDetected,
    /// Maximum capacity exceeded
    CapacityExceeded,
    /// Invalid routing configuration
    InvalidRouting,
}

impl fmt::Display for SHNNError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SHNNError::InvalidNeuronConfig { neuron_id, reason } => {
                write!(f, "Invalid neuron configuration for neuron {}: {}", neuron_id, reason)
            }
            SHNNError::InvalidSpike { reason } => {
                write!(f, "Invalid spike: {}", reason)
            }
            SHNNError::HypergraphError { kind, context } => {
                write!(f, "Hypergraph error ({:?}): {}", kind, context)
            }
            SHNNError::PlasticityError { reason } => {
                write!(f, "Plasticity error: {}", reason)
            }
            SHNNError::TimeError { reason } => {
                write!(f, "Time error: {}", reason)
            }
            SHNNError::MemoryError { reason } => {
                write!(f, "Memory error: {}", reason)
            }
            SHNNError::EncodingError { reason } => {
                write!(f, "Encoding error: {}", reason)
            }
            #[cfg(feature = "hardware-accel")]
            SHNNError::HardwareError { reason } => {
                write!(f, "Hardware error: {}", reason)
            }
            #[cfg(feature = "serde")]
            SHNNError::SerializationError { reason } => {
                write!(f, "Serialization error: {}", reason)
            }
            #[cfg(feature = "async")]
            SHNNError::AsyncError { reason } => {
                write!(f, "Async error: {}", reason)
            }
            SHNNError::Generic { message } => {
                write!(f, "Error: {}", message)
            }
        }
    }
}

impl fmt::Display for HypergraphErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HypergraphErrorKind::NodeNotFound => write!(f, "Node not found"),
            HypergraphErrorKind::HyperedgeNotFound => write!(f, "Hyperedge not found"),
            HypergraphErrorKind::InvalidHyperedge => write!(f, "Invalid hyperedge"),
            HypergraphErrorKind::CycleDetected => write!(f, "Cycle detected"),
            HypergraphErrorKind::CapacityExceeded => write!(f, "Capacity exceeded"),
            HypergraphErrorKind::InvalidRouting => write!(f, "Invalid routing"),
        }
    }
}

#[cfg(feature = "std")]
impl StdError for SHNNError {}

#[cfg(feature = "std")]
impl StdError for HypergraphErrorKind {}

// Convenience constructors for common errors
impl SHNNError {
    /// Create an invalid neuron configuration error
    pub fn invalid_neuron_config(neuron_id: u32, reason: &'static str) -> Self {
        Self::InvalidNeuronConfig { neuron_id, reason }
    }
    
    /// Create an invalid spike error
    pub fn invalid_spike(reason: &'static str) -> Self {
        Self::InvalidSpike { reason }
    }
    
    /// Create a hypergraph error
    pub fn hypergraph_error(kind: HypergraphErrorKind, context: &'static str) -> Self {
        Self::HypergraphError { kind, context }
    }
    
    /// Create a plasticity error
    pub fn plasticity_error(reason: &'static str) -> Self {
        Self::PlasticityError { reason }
    }
    
    /// Create a time error
    pub fn time_error(reason: &'static str) -> Self {
        Self::TimeError { reason }
    }
    
    /// Create a memory error
    pub fn memory_error(reason: &'static str) -> Self {
        Self::MemoryError { reason }
    }
    
    /// Create an encoding error
    pub fn encoding_error(reason: &'static str) -> Self {
        Self::EncodingError { reason }
    }
    
    /// Create a hardware error
    #[cfg(feature = "hardware-accel")]
    pub fn hardware_error(reason: &'static str) -> Self {
        Self::HardwareError { reason }
    }
    
    /// Create a serialization error
    #[cfg(feature = "serde")]
    pub fn serialization_error(reason: String) -> Self {
        Self::SerializationError { reason }
    }
    
    /// Create an async error
    #[cfg(feature = "async")]
    pub fn async_error(reason: &'static str) -> Self {
        Self::AsyncError { reason }
    }
    
    /// Create a generic error
    pub fn generic(message: &'static str) -> Self {
        Self::Generic { message }
    }
}

/// Macro for creating errors with formatted messages
#[macro_export]
macro_rules! shnn_error {
    ($kind:ident, $msg:expr) => {
        $crate::error::SHNNError::$kind { reason: $msg }
    };
    ($kind:ident, $msg:expr, $($arg:expr),+) => {
        $crate::error::SHNNError::$kind { reason: &format!($msg, $($arg),+) }
    };
}

/// Macro for creating results
#[macro_export]
macro_rules! shnn_result {
    ($expr:expr) => {
        Ok($expr)
    };
    (Err $kind:ident, $msg:expr) => {
        Err($crate::error::SHNNError::$kind { reason: $msg })
    };
}

/// Trait for converting errors from external crates
pub trait IntoSHNNError {
    /// Convert into an SHNN error
    fn into_shnn_error(self) -> SHNNError;
}

// Implement conversions for common error types
#[cfg(feature = "std")]
impl IntoSHNNError for std::io::Error {
    fn into_shnn_error(self) -> SHNNError {
        SHNNError::generic("I/O error")
    }
}

#[cfg(feature = "serde")]
impl IntoSHNNError for serde_json::Error {
    fn into_shnn_error(self) -> SHNNError {
        SHNNError::serialization_error("JSON serialization error")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let error = SHNNError::invalid_neuron_config(42, "test reason");
        assert_eq!(
            format!("{}", error),
            "Invalid neuron configuration for neuron 42: test reason"
        );
    }
    
    #[test]
    fn test_error_equality() {
        let error1 = SHNNError::invalid_spike("test");
        let error2 = SHNNError::invalid_spike("test");
        let error3 = SHNNError::invalid_spike("different");
        
        assert_eq!(error1, error2);
        assert_ne!(error1, error3);
    }
    
    #[test]
    fn test_hypergraph_error() {
        let error = SHNNError::hypergraph_error(
            HypergraphErrorKind::NodeNotFound,
            "test context"
        );
        
        assert!(format!("{}", error).contains("Node not found"));
        assert!(format!("{}", error).contains("test context"));
    }
    
    #[test]
    fn test_result_type() {
        let success: Result<i32> = Ok(42);
        let failure: Result<i32> = Err(SHNNError::generic("test error"));
        
        assert!(success.is_ok());
        assert!(failure.is_err());
    }
}