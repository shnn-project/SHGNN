//! Error handling for embedded SHNN systems
//!
//! This module provides lightweight error handling optimized for no-std environments.

use core::fmt;

/// Result type for embedded operations
pub type EmbeddedResult<T> = Result<T, EmbeddedError>;

/// Embedded error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddedError {
    /// Memory allocation failed
    OutOfMemory,
    /// Invalid configuration
    InvalidConfig,
    /// Hardware error
    HardwareError,
    /// Timeout error
    Timeout,
    /// Invalid neuron ID
    InvalidNeuronId,
    /// Buffer overflow
    BufferOverflow,
    /// Arithmetic overflow
    ArithmeticOverflow,
    /// Real-time constraint violation
    RealTimeViolation,
}

impl fmt::Display for EmbeddedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory => write!(f, "Out of memory"),
            Self::InvalidConfig => write!(f, "Invalid configuration"),
            Self::HardwareError => write!(f, "Hardware error"),
            Self::Timeout => write!(f, "Operation timed out"),
            Self::InvalidNeuronId => write!(f, "Invalid neuron ID"),
            Self::BufferOverflow => write!(f, "Buffer overflow"),
            Self::ArithmeticOverflow => write!(f, "Arithmetic overflow"),
            Self::RealTimeViolation => write!(f, "Real-time constraint violation"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EmbeddedError {}