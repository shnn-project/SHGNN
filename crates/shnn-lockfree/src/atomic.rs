//! Atomic operations and primitives optimized for neuromorphic computations
//! 
//! Provides zero-dependency atomic types with enhanced operations for
//! neural network state management and spike processing.

use core::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use crate::{Result, LockFreeError, constants::MAX_CAS_RETRIES};

/// Atomic floating-point operations for neural state values
#[repr(transparent)]
pub struct AtomicFloat {
    inner: AtomicU64,
}

impl AtomicFloat {
    /// Create new atomic float with initial value
    pub const fn new(value: f32) -> Self {
        Self {
            inner: AtomicU64::new(value.to_bits() as u64),
        }
    }

    /// Load the current value
    #[inline]
    pub fn load(&self, ordering: Ordering) -> f32 {
        f32::from_bits(self.inner.load(ordering) as u32)
    }

    /// Store a new value
    #[inline]
    pub fn store(&self, value: f32, ordering: Ordering) {
        self.inner.store(value.to_bits() as u64, ordering);
    }

    /// Compare-and-swap operation
    #[inline]
    pub fn compare_exchange(&self, current: f32, new: f32, success: Ordering, failure: Ordering) -> core::result::Result<f32, f32> {
        match self.inner.compare_exchange(
            current.to_bits() as u64,
            new.to_bits() as u64,
            success,
            failure,
        ) {
            Ok(val) => Ok(f32::from_bits(val as u32)),
            Err(val) => Err(f32::from_bits(val as u32)),
        }
    }

    /// Atomic add operation with retry logic
    pub fn add(&self, value: f32) -> Result<f32> {
        for _ in 0..MAX_CAS_RETRIES {
            let current = self.load(Ordering::Acquire);
            let new_value = current + value;
            
            match self.compare_exchange(current, new_value, Ordering::Release, Ordering::Relaxed) {
                Ok(_) => return Ok(new_value),
                Err(_) => continue,
            }
        }
        Err(LockFreeError::ConcurrentModification)
    }

    /// Atomic multiply operation for learning rate adjustments
    pub fn multiply(&self, factor: f32) -> Result<f32> {
        for _ in 0..MAX_CAS_RETRIES {
            let current = self.load(Ordering::Acquire);
            let new_value = current * factor;
            
            match self.compare_exchange(current, new_value, Ordering::Release, Ordering::Relaxed) {
                Ok(_) => return Ok(new_value),
                Err(_) => continue,
            }
        }
        Err(LockFreeError::ConcurrentModification)
    }

    /// Atomic max operation for threshold updates
    pub fn fetch_max(&self, value: f32) -> Result<f32> {
        for _ in 0..MAX_CAS_RETRIES {
            let current = self.load(Ordering::Acquire);
            let new_value = current.max(value);
            
            if current >= value {
                return Ok(current);
            }
            
            match self.compare_exchange(current, new_value, Ordering::Release, Ordering::Relaxed) {
                Ok(_) => return Ok(current),
                Err(_) => continue,
            }
        }
        Err(LockFreeError::ConcurrentModification)
    }
}

/// Atomic counter optimized for spike counting
pub struct AtomicCounter {
    inner: AtomicUsize,
}

impl AtomicCounter {
    /// Create new counter with initial value
    pub const fn new(value: usize) -> Self {
        Self {
            inner: AtomicUsize::new(value),
        }
    }

    /// Load current count
    #[inline]
    pub fn load(&self, ordering: Ordering) -> usize {
        self.inner.load(ordering)
    }

    /// Store new count
    #[inline]
    pub fn store(&self, value: usize, ordering: Ordering) {
        self.inner.store(value, ordering);
    }

    /// Increment counter and return new value
    #[inline]
    pub fn increment(&self) -> usize {
        self.inner.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Decrement counter and return new value
    #[inline]
    pub fn decrement(&self) -> usize {
        self.inner.fetch_sub(1, Ordering::AcqRel).saturating_sub(1)
    }

    /// Add value to counter
    #[inline]
    pub fn add(&self, value: usize) -> usize {
        self.inner.fetch_add(value, Ordering::AcqRel) + value
    }

    /// Reset counter to zero and return previous value
    #[inline]
    pub fn reset(&self) -> usize {
        self.inner.swap(0, Ordering::AcqRel)
    }

    /// Compare and swap operation
    #[inline]
    pub fn compare_exchange(&self, current: usize, new: usize) -> core::result::Result<usize, usize> {
        self.inner.compare_exchange(current, new, Ordering::AcqRel, Ordering::Relaxed)
    }
}

/// Atomic flag for neural state signaling
pub struct AtomicFlag {
    inner: AtomicBool,
}

impl AtomicFlag {
    /// Create new flag with initial state
    pub const fn new(value: bool) -> Self {
        Self {
            inner: AtomicBool::new(value),
        }
    }

    /// Load current flag state
    #[inline]
    pub fn load(&self, ordering: Ordering) -> bool {
        self.inner.load(ordering)
    }

    /// Store new flag state
    #[inline]
    pub fn store(&self, value: bool, ordering: Ordering) {
        self.inner.store(value, ordering);
    }

    /// Set flag to true and return previous state
    #[inline]
    pub fn set(&self) -> bool {
        self.inner.swap(true, Ordering::AcqRel)
    }

    /// Clear flag to false and return previous state
    #[inline]
    pub fn clear(&self) -> bool {
        self.inner.swap(false, Ordering::AcqRel)
    }

    /// Toggle flag and return new state
    #[inline]
    pub fn toggle(&self) -> bool {
        let current = self.inner.load(Ordering::Acquire);
        self.inner.store(!current, Ordering::Release);
        !current
    }

    /// Test and set operation - set to true if currently false
    #[inline]
    pub fn test_and_set(&self) -> bool {
        self.inner.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed).is_err()
    }
}

/// Memory barrier utilities for neuromorphic synchronization
pub struct MemoryBarrier;

impl MemoryBarrier {
    /// Full memory barrier
    #[inline]
    pub fn full() {
        core::sync::atomic::fence(Ordering::SeqCst);
    }

    /// Acquire barrier
    #[inline]
    pub fn acquire() {
        core::sync::atomic::fence(Ordering::Acquire);
    }

    /// Release barrier
    #[inline]
    pub fn release() {
        core::sync::atomic::fence(Ordering::Release);
    }

    /// Compiler barrier (prevents reordering)
    #[inline]
    pub fn compiler() {
        core::sync::atomic::compiler_fence(Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_float() {
        let af = AtomicFloat::new(1.5);
        assert_eq!(af.load(Ordering::Relaxed), 1.5);
        
        af.store(2.5, Ordering::Relaxed);
        assert_eq!(af.load(Ordering::Relaxed), 2.5);
        
        let result = af.add(1.0).unwrap();
        assert_eq!(result, 3.5);
        assert_eq!(af.load(Ordering::Relaxed), 3.5);
    }

    #[test]
    fn test_atomic_counter() {
        let counter = AtomicCounter::new(10);
        assert_eq!(counter.load(Ordering::Relaxed), 10);
        
        assert_eq!(counter.increment(), 11);
        assert_eq!(counter.decrement(), 10);
        assert_eq!(counter.add(5), 15);
        assert_eq!(counter.reset(), 15);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_atomic_flag() {
        let flag = AtomicFlag::new(false);
        assert!(!flag.load(Ordering::Relaxed));
        
        assert!(!flag.set());
        assert!(flag.load(Ordering::Relaxed));
        
        assert!(flag.clear());
        assert!(!flag.load(Ordering::Relaxed));
        
        assert!(!flag.test_and_set());
        assert!(flag.test_and_set());
    }
}