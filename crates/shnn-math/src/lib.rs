//! # SHNN Math Library
//! 
//! Zero-dependency math library optimized for neuromorphic computations.
//! Provides SIMD-accelerated operations with deterministic performance.

#![cfg_attr(not(feature = "std"), no_std)]
// Note: SIMD features require nightly Rust - disabled for stable compilation
// #![cfg_attr(feature = "simd", feature(portable_simd))]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

/// Vector operations module
pub mod vector;

/// Matrix operations module  
pub mod matrix;

/// Neural activation functions
pub mod activation;

/// SIMD-optimized operations
#[cfg(feature = "simd")]
pub mod simd;

/// Sparse data structures for synaptic connectivity
pub mod sparse;

/// Fast approximate functions
pub mod approx;

/// Statistical functions
pub mod stats;

/// Custom math functions for no-std compatibility
pub mod math;

// Re-exports for convenience
pub use vector::{Vector, VectorOps};
pub use matrix::{Matrix, MatrixOps};
pub use activation::{ActivationFunction, sigmoid, tanh, relu, leaky_relu};
pub use sparse::{SparseMatrix, SparseVector};

/// Common floating-point type for neural computations
pub type Float = f32;

/// Neuromorphic-specific constants
pub mod constants {
    use super::Float;

    /// Membrane potential threshold (mV)
    pub const THRESHOLD: Float = -50.0;
    
    /// Resting potential (mV)
    pub const RESTING_POTENTIAL: Float = -70.0;
    
    /// Reset potential (mV)
    pub const RESET_POTENTIAL: Float = -80.0;
    
    /// Membrane time constant (ms)
    pub const TAU_M: Float = 20.0;
    
    /// Synaptic time constant (ms)
    pub const TAU_S: Float = 5.0;
    
    /// Refractory period (ms)
    pub const T_REF: Float = 2.0;
    
    /// Small epsilon for numerical stability
    pub const EPSILON: Float = 1e-8;
    
    /// Maximum spike frequency (Hz)
    pub const MAX_SPIKE_FREQ: Float = 1000.0;
}

/// Error types for math operations
#[derive(Debug, Clone, PartialEq)]
pub enum MathError {
    /// Dimension mismatch in operations
    DimensionMismatch { expected: usize, got: usize },
    /// Division by zero
    DivisionByZero,
    /// Invalid input value
    InvalidInput { reason: &'static str },
    /// Memory allocation failed
    OutOfMemory,
    /// Index out of bounds
    IndexOutOfBounds { index: usize, len: usize },
    /// Singular matrix (determinant is zero)
    SingularMatrix,
    /// Unsupported operation
    UnsupportedOperation,
}

#[cfg(feature = "std")]
impl std::fmt::Display for MathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MathError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            MathError::DivisionByZero => write!(f, "Division by zero"),
            MathError::InvalidInput { reason } => write!(f, "Invalid input: {}", reason),
            MathError::OutOfMemory => write!(f, "Out of memory"),
            MathError::IndexOutOfBounds { index, len } => {
                write!(f, "Index {} out of bounds for length {}", index, len)
            }
            MathError::SingularMatrix => write!(f, "Matrix is singular"),
            MathError::UnsupportedOperation => write!(f, "Operation not supported"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MathError {}

/// Result type for math operations
pub type Result<T> = core::result::Result<T, MathError>;

/// Performance timing utilities
#[cfg(feature = "std")]
pub mod perf {
    use std::time::Instant;
    
    /// Simple benchmark helper
    pub fn time_it<F, R>(name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        println!("{}: {:?}", name, elapsed);
        result
    }
    
    /// Benchmark with iterations
    pub fn bench_iterations<F>(name: &str, iterations: usize, mut f: F)
    where
        F: FnMut(),
    {
        let start = Instant::now();
        for _ in 0..iterations {
            f();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iterations as u32;
        println!("{}: {} iterations in {:?} ({:?}/iter)", 
                 name, iterations, elapsed, per_iter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert!(constants::THRESHOLD > constants::RESTING_POTENTIAL);
        assert!(constants::RESTING_POTENTIAL > constants::RESET_POTENTIAL);
        assert!(constants::TAU_M > 0.0);
        assert!(constants::EPSILON > 0.0);
    }

    #[test]
    fn test_error_display() {
        let err = MathError::DimensionMismatch { expected: 5, got: 3 };
        #[cfg(feature = "std")]
        {
            let display = format!("{}", err);
            assert!(display.contains("5"));
            assert!(display.contains("3"));
        }
    }
}