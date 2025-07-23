//! Memory ordering utilities for neuromorphic concurrency
//! 
//! Provides helper types and functions for managing memory ordering
//! in lock-free data structures optimized for spike processing.

use core::sync::atomic::Ordering;

/// Memory ordering wrapper with neuromorphic-specific semantics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrdering {
    /// Relaxed ordering - no synchronization constraints
    Relaxed,
    /// Acquire ordering - synchronize with release operations
    Acquire,
    /// Release ordering - synchronize with acquire operations
    Release,
    /// AcqRel ordering - both acquire and release semantics
    AcqRel,
    /// SeqCst ordering - sequential consistency
    SeqCst,
}

impl MemoryOrdering {
    /// Convert to standard atomic ordering
    pub fn to_atomic(self) -> Ordering {
        match self {
            MemoryOrdering::Relaxed => Ordering::Relaxed,
            MemoryOrdering::Acquire => Ordering::Acquire,
            MemoryOrdering::Release => Ordering::Release,
            MemoryOrdering::AcqRel => Ordering::AcqRel,
            MemoryOrdering::SeqCst => Ordering::SeqCst,
        }
    }

    /// Get recommended ordering for spike event operations
    pub fn spike_event() -> Self {
        MemoryOrdering::AcqRel
    }

    /// Get recommended ordering for neural state updates
    pub fn neural_state() -> Self {
        MemoryOrdering::Release
    }

    /// Get recommended ordering for learning rate updates
    pub fn learning_rate() -> Self {
        MemoryOrdering::SeqCst
    }

    /// Get recommended ordering for weight matrix updates
    pub fn weight_update() -> Self {
        MemoryOrdering::AcqRel
    }

    /// Get recommended ordering for threshold operations
    pub fn threshold_check() -> Self {
        MemoryOrdering::Acquire
    }

    /// Get relaxed ordering for performance-critical paths
    pub fn performance_critical() -> Self {
        MemoryOrdering::Relaxed
    }
}

impl From<MemoryOrdering> for Ordering {
    fn from(ordering: MemoryOrdering) -> Self {
        ordering.to_atomic()
    }
}

impl Default for MemoryOrdering {
    fn default() -> Self {
        MemoryOrdering::AcqRel
    }
}

/// Backoff strategy for reducing contention in lock-free operations
pub struct Backoff {
    step: usize,
    max_step: usize,
}

impl Backoff {
    /// Create new backoff strategy
    pub const fn new() -> Self {
        Self {
            step: 0,
            max_step: 6, // Maximum 2^6 = 64 iterations
        }
    }

    /// Perform backoff operation
    pub fn backoff(&mut self) {
        if self.step <= self.max_step {
            for _ in 0..(1 << self.step) {
                core::hint::spin_loop();
            }
            self.step += 1;
        } else {
            // Yield to scheduler if available
            #[cfg(feature = "std")]
            std::thread::yield_now();
            
            #[cfg(not(feature = "std"))]
            for _ in 0..100 {
                core::hint::spin_loop();
            }
        }
    }

    /// Reset backoff to initial state
    pub fn reset(&mut self) {
        self.step = 0;
    }

    /// Check if we should yield to other threads
    pub fn should_yield(&self) -> bool {
        self.step > self.max_step
    }
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory fence utilities for neuromorphic synchronization
pub struct MemoryFence;

impl MemoryFence {
    /// Full memory fence for critical synchronization points
    #[inline]
    pub fn full() {
        core::sync::atomic::fence(Ordering::SeqCst);
    }

    /// Acquire fence for loading shared state
    #[inline]
    pub fn acquire() {
        core::sync::atomic::fence(Ordering::Acquire);
    }

    /// Release fence for publishing shared state
    #[inline]
    pub fn release() {
        core::sync::atomic::fence(Ordering::Release);
    }

    /// Compiler fence to prevent reordering
    #[inline]
    pub fn compiler() {
        core::sync::atomic::compiler_fence(Ordering::SeqCst);
    }

    /// Spike synchronization barrier
    #[inline]
    pub fn spike_barrier() {
        core::sync::atomic::fence(Ordering::AcqRel);
    }

    /// Neural state synchronization
    #[inline]
    pub fn neural_sync() {
        core::sync::atomic::fence(Ordering::Release);
    }
}

/// Cache line padding to prevent false sharing
#[repr(align(64))]
pub struct CacheAligned<T> {
    value: T,
}

impl<T> CacheAligned<T> {
    /// Create new cache-aligned value
    pub const fn new(value: T) -> Self {
        Self { value }
    }

    /// Get reference to inner value
    pub const fn get(&self) -> &T {
        &self.value
    }

    /// Get mutable reference to inner value
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.value
    }

    /// Consume and return inner value
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T: Default> Default for CacheAligned<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: Clone> Clone for CacheAligned<T> {
    fn clone(&self) -> Self {
        Self::new(self.value.clone())
    }
}

/// Atomic operation wrapper with neuromorphic semantics
pub struct AtomicOp;

impl AtomicOp {
    /// Perform compare-and-swap with appropriate backoff
    pub fn cas_with_backoff<T, F>(
        operation: F,
        max_retries: usize,
    ) -> Result<T, crate::LockFreeError>
    where
        F: Fn() -> Result<T, ()>,
    {
        let mut backoff = Backoff::new();
        
        for _ in 0..max_retries {
            match operation() {
                Ok(result) => return Ok(result),
                Err(_) => {
                    backoff.backoff();
                    continue;
                }
            }
        }
        
        Err(crate::LockFreeError::ConcurrentModification)
    }

    /// Perform operation with memory ordering for spike events
    pub fn spike_operation<T, F>(operation: F) -> T
    where
        F: FnOnce(Ordering) -> T,
    {
        operation(MemoryOrdering::spike_event().to_atomic())
    }

    /// Perform operation with memory ordering for neural state
    pub fn neural_state_operation<T, F>(operation: F) -> T
    where
        F: FnOnce(Ordering) -> T,
    {
        operation(MemoryOrdering::neural_state().to_atomic())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_memory_ordering() {
        assert_eq!(MemoryOrdering::Relaxed.to_atomic(), Ordering::Relaxed);
        assert_eq!(MemoryOrdering::Acquire.to_atomic(), Ordering::Acquire);
        assert_eq!(MemoryOrdering::Release.to_atomic(), Ordering::Release);
        assert_eq!(MemoryOrdering::AcqRel.to_atomic(), Ordering::AcqRel);
        assert_eq!(MemoryOrdering::SeqCst.to_atomic(), Ordering::SeqCst);
    }

    #[test]
    fn test_neuromorphic_orderings() {
        assert_eq!(MemoryOrdering::spike_event(), MemoryOrdering::AcqRel);
        assert_eq!(MemoryOrdering::neural_state(), MemoryOrdering::Release);
        assert_eq!(MemoryOrdering::learning_rate(), MemoryOrdering::SeqCst);
        assert_eq!(MemoryOrdering::performance_critical(), MemoryOrdering::Relaxed);
    }

    #[test]
    fn test_backoff() {
        let mut backoff = Backoff::new();
        assert!(!backoff.should_yield());
        
        // Perform several backoffs
        for _ in 0..10 {
            backoff.backoff();
        }
        
        assert!(backoff.should_yield());
        
        backoff.reset();
        assert!(!backoff.should_yield());
    }

    #[test]
    fn test_cache_aligned() {
        let aligned = CacheAligned::new(42);
        assert_eq!(*aligned.get(), 42);
        
        let mut aligned = CacheAligned::new(100);
        *aligned.get_mut() = 200;
        assert_eq!(aligned.into_inner(), 200);
    }

    #[test]
    fn test_atomic_op() {
        let counter = AtomicUsize::new(0);
        
        let result = AtomicOp::cas_with_backoff(
            || {
                match counter.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed) {
                    Ok(_) => Ok(1),
                    Err(_) => Err(()),
                }
            },
            10,
        );
        
        assert_eq!(result.unwrap(), 1);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_spike_operation() {
        let counter = AtomicUsize::new(5);
        
        let result = AtomicOp::spike_operation(|ordering| {
            counter.load(ordering)
        });
        
        assert_eq!(result, 5);
    }
}