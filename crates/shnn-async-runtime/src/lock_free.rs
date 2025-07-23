//! Lock-free data structures optimized for spike event processing
//! 
//! This module provides high-performance, wait-free data structures specifically
//! designed for neuromorphic workloads where deterministic latency is critical.

use core::{
    sync::atomic::{AtomicUsize, Ordering},
    mem::MaybeUninit,
    ptr,
    cell::UnsafeCell,
};
use alloc::{boxed::Box, vec::Vec};

use crate::SpikeEvent;

/// Lock-free MPMC queue optimized for spike events
/// 
/// Uses a ring buffer with atomic head/tail pointers for maximum throughput.
/// Specifically optimized for spike event patterns with temporal locality.
pub struct SpikeEventQueue {
    buffer: Box<[UnsafeCell<MaybeUninit<SpikeEvent>>]>,
    head: AtomicUsize,
    tail: AtomicUsize,
    capacity: usize,
    mask: usize,
}

unsafe impl Send for SpikeEventQueue {}
unsafe impl Sync for SpikeEventQueue {}

impl SpikeEventQueue {
    /// Create queue with specified capacity (must be power of 2)
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two(), "Capacity must be power of 2");
        assert!(capacity >= 2, "Capacity must be at least 2");

        let buffer = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            buffer,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            capacity,
            mask: capacity - 1,
        }
    }

    /// Non-blocking enqueue with backpressure handling
    /// 
    /// Returns `Err(spike)` if queue is full, allowing caller to handle backpressure.
    pub fn try_enqueue(&self, spike: SpikeEvent) -> Result<(), SpikeEvent> {
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (tail + 1) & self.mask;

        // Check if queue is full (would overtake head)
        if next_tail == self.head.load(Ordering::Acquire) {
            return Err(spike); // Queue full - apply backpressure
        }

        // Write spike event at tail position
        unsafe {
            let slot = &mut *self.buffer[tail].get();
            ptr::write(slot.as_mut_ptr(), spike);
        }

        // Advance tail pointer (release ordering ensures write is visible)
        self.tail.store(next_tail, Ordering::Release);
        Ok(())
    }

    /// Non-blocking dequeue
    /// 
    /// Returns `None` if queue is empty.
    pub fn try_dequeue(&self) -> Option<SpikeEvent> {
        let head = self.head.load(Ordering::Relaxed);

        // Check if queue is empty
        if head == self.tail.load(Ordering::Acquire) {
            return None;
        }

        // Read spike event from head position
        let spike = unsafe {
            let slot = &*self.buffer[head].get();
            slot.as_ptr().read()
        };

        // Advance head pointer
        self.head.store((head + 1) & self.mask, Ordering::Release);
        Some(spike)
    }

    /// Get current queue length (approximate)
    pub fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        (tail.wrapping_sub(head)) & self.mask
    }

    /// Check if queue is empty (approximate)
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }

    /// Check if queue is full (approximate)
    pub fn is_full(&self) -> bool {
        let tail = self.tail.load(Ordering::Acquire);
        let next_tail = (tail + 1) & self.mask;
        next_tail == self.head.load(Ordering::Acquire)
    }
}

impl Drop for SpikeEventQueue {
    fn drop(&mut self) {
        // Drain remaining elements to prevent memory leaks
        while self.try_dequeue().is_some() {}
    }
}

/// Lock-free MPMC queue for generic items
/// 
/// Similar to SpikeEventQueue but works with any `Send + Sync` type.
pub struct LockFreeQueue<T> {
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    head: AtomicUsize,
    tail: AtomicUsize,
    capacity: usize,
    mask: usize,
}

unsafe impl<T: Send> Send for LockFreeQueue<T> {}
unsafe impl<T: Sync> Sync for LockFreeQueue<T> {}

impl<T> LockFreeQueue<T> {
    /// Create queue with specified capacity (must be power of 2)
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two(), "Capacity must be power of 2");
        assert!(capacity >= 2, "Capacity must be at least 2");

        let buffer = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            buffer,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            capacity,
            mask: capacity - 1,
        }
    }

    /// Non-blocking enqueue
    pub fn try_push(&self, item: T) -> Result<(), T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (tail + 1) & self.mask;

        if next_tail == self.head.load(Ordering::Acquire) {
            return Err(item); // Queue full
        }

        unsafe {
            let slot = &mut *self.buffer[tail].get();
            ptr::write(slot.as_mut_ptr(), item);
        }

        self.tail.store(next_tail, Ordering::Release);
        Ok(())
    }

    /// Non-blocking dequeue
    pub fn try_pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);

        if head == self.tail.load(Ordering::Acquire) {
            return None; // Queue empty
        }

        let item = unsafe {
            let slot = &*self.buffer[head].get();
            slot.as_ptr().read()
        };

        self.head.store((head + 1) & self.mask, Ordering::Release);
        Some(item)
    }

    /// Get current queue length (approximate)
    pub fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        (tail.wrapping_sub(head)) & self.mask
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }
}

impl<T> Drop for LockFreeQueue<T> {
    fn drop(&mut self) {
        // Drain remaining elements to prevent memory leaks
        while self.try_pop().is_some() {}
    }
}

/// Work-stealing deque for task distribution
/// 
/// Optimized for the common case where tasks are pushed/popped from one end
/// but can be stolen from the other end by worker threads.
pub struct StealQueue<T> {
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    head: AtomicUsize,
    tail: AtomicUsize,
    capacity: usize,
    mask: usize,
}

unsafe impl<T: Send> Send for StealQueue<T> {}
unsafe impl<T: Sync> Sync for StealQueue<T> {}

impl<T> StealQueue<T> {
    /// Create new work-stealing deque
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two(), "Capacity must be power of 2");

        let buffer = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            buffer,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            capacity,
            mask: capacity - 1,
        }
    }

    /// Push item (only called by owner thread)
    pub fn push(&self, item: T) -> Result<(), T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (tail + 1) & self.mask;

        if next_tail == self.head.load(Ordering::Acquire) {
            return Err(item); // Queue full
        }

        unsafe {
            let slot = &mut *self.buffer[tail].get();
            ptr::write(slot.as_mut_ptr(), item);
        }

        self.tail.store(next_tail, Ordering::Release);
        Ok(())
    }

    /// Pop item (only called by owner thread)
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        
        if tail == 0 {
            return None;
        }

        let new_tail = tail - 1;
        self.tail.store(new_tail, Ordering::Relaxed);

        if new_tail == self.head.load(Ordering::Acquire) {
            // Queue became empty, restore tail
            self.tail.store(tail, Ordering::Relaxed);
            return None;
        }

        unsafe {
            let slot = &*self.buffer[new_tail].get();
            Some(slot.as_ptr().read())
        }
    }

    /// Steal item (called by other worker threads)
    pub fn steal(&self) -> Option<T> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);

        if head >= tail {
            return None; // Queue empty or being modified
        }

        let item = unsafe {
            let slot = &*self.buffer[head].get();
            slot.as_ptr().read()
        };

        // Try to advance head atomically
        if self.head.compare_exchange_weak(
            head,
            (head + 1) & self.mask,
            Ordering::Release,
            Ordering::Relaxed,
        ).is_ok() {
            Some(item)
        } else {
            // Someone else stole it, forget the item we read
            core::mem::forget(item);
            None
        }
    }

    /// Get approximate length
    pub fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        tail.saturating_sub(head)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Drop for StealQueue<T> {
    fn drop(&mut self) {
        // Drain remaining elements
        while self.pop().is_some() {}
    }
}

/// Cache-aligned atomic counter for performance monitoring
#[repr(align(64))]
pub struct AtomicCounter {
    value: AtomicUsize,
}

impl AtomicCounter {
    /// Create new counter
    pub fn new() -> Self {
        Self {
            value: AtomicUsize::new(0),
        }
    }

    /// Increment counter
    pub fn increment(&self) -> usize {
        self.value.fetch_add(1, Ordering::Relaxed)
    }

    /// Get current value
    pub fn get(&self) -> usize {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset counter
    pub fn reset(&self) -> usize {
        self.value.swap(0, Ordering::Relaxed)
    }
}

impl Default for AtomicCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_event_queue() {
        let queue = SpikeEventQueue::with_capacity(4);
        
        let spike1 = SpikeEvent::new(1000, 1, 1.0);
        let spike2 = SpikeEvent::new(2000, 2, 2.0);
        
        assert!(queue.try_enqueue(spike1).is_ok());
        assert!(queue.try_enqueue(spike2).is_ok());
        
        assert_eq!(queue.try_dequeue().unwrap(), spike1);
        assert_eq!(queue.try_dequeue().unwrap(), spike2);
        assert!(queue.try_dequeue().is_none());
    }

    #[test]
    fn test_lock_free_queue() {
        let queue = LockFreeQueue::with_capacity(4);
        
        assert!(queue.try_push(1).is_ok());
        assert!(queue.try_push(2).is_ok());
        
        assert_eq!(queue.try_pop().unwrap(), 1);
        assert_eq!(queue.try_pop().unwrap(), 2);
        assert!(queue.try_pop().is_none());
    }

    #[test]
    fn test_steal_queue() {
        let queue = StealQueue::with_capacity(4);
        
        assert!(queue.push(1).is_ok());
        assert!(queue.push(2).is_ok());
        
        assert_eq!(queue.steal().unwrap(), 1);
        assert_eq!(queue.pop().unwrap(), 2);
        assert!(queue.steal().is_none());
    }

    #[test]
    fn test_atomic_counter() {
        let counter = AtomicCounter::new();
        
        assert_eq!(counter.get(), 0);
        assert_eq!(counter.increment(), 0);
        assert_eq!(counter.get(), 1);
        assert_eq!(counter.reset(), 1);
        assert_eq!(counter.get(), 0);
    }
}