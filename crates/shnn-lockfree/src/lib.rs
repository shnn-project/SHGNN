//! # SHNN Lock-Free Concurrency Library
//! 
//! Zero-dependency lock-free concurrency primitives optimized for neuromorphic computations.
//! Provides high-performance atomic operations and data structures for spike processing.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::all)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

/// Atomic operations and primitives
pub mod atomic;

/// Lock-free queue implementations
pub mod queue;

/// Lock-free stack implementations
pub mod stack;

/// Memory ordering utilities
pub mod ordering;

/// Epoch-based memory management
pub mod epoch;

// Re-exports for convenience
pub use atomic::{AtomicFloat, AtomicCounter, AtomicFlag};
pub use queue::{SPSCQueue, MPSCQueue, MPMCQueue};
pub use stack::LockFreeStack;
pub use ordering::MemoryOrdering;

/// Common error types for lock-free operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockFreeError {
    /// Queue is full
    QueueFull,
    /// Queue is empty
    QueueEmpty,
    /// Stack is empty
    StackEmpty,
    /// Invalid capacity
    InvalidCapacity,
    /// Memory allocation failed
    AllocationFailed,
    /// Concurrent modification detected
    ConcurrentModification,
}

/// Result type for lock-free operations
pub type Result<T> = core::result::Result<T, LockFreeError>;

/// Neuromorphic-specific constants for lock-free operations
pub mod constants {
    /// Default queue capacity for spike events
    pub const DEFAULT_SPIKE_QUEUE_SIZE: usize = 4096;
    
    /// Default buffer size for neural state vectors
    pub const DEFAULT_STATE_BUFFER_SIZE: usize = 1024;
    
    /// Maximum retry attempts for CAS operations
    pub const MAX_CAS_RETRIES: usize = 16;
    
    /// Cache line size for alignment optimization
    pub const CACHE_LINE_SIZE: usize = 64;
    
    /// Default backoff delay in nanoseconds
    pub const DEFAULT_BACKOFF_NS: u64 = 100;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert!(constants::DEFAULT_SPIKE_QUEUE_SIZE.is_power_of_two());
        assert!(constants::DEFAULT_STATE_BUFFER_SIZE.is_power_of_two());
        assert!(constants::MAX_CAS_RETRIES > 0);
        assert!(constants::CACHE_LINE_SIZE == 64);
    }

    #[test]
    fn test_error_types() {
        let err = LockFreeError::QueueFull;
        assert_eq!(err, LockFreeError::QueueFull);
        
        let result: Result<()> = Err(LockFreeError::QueueEmpty);
        assert!(result.is_err());
    }
}