//! Lock-free stack implementation for neuromorphic computations
//! 
//! Provides a Treiber stack optimized for neural state management
//! and temporary storage of spike events.

use core::sync::atomic::{AtomicPtr, Ordering};
use core::ptr;
use crate::{Result, LockFreeError, constants::MAX_CAS_RETRIES};
#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

#[cfg(feature = "std")]
use std::boxed::Box;

/// Lock-free stack using Treiber's algorithm
pub struct LockFreeStack<T> {
    head: AtomicPtr<Node<T>>,
}

struct Node<T> {
    data: T,
    next: *mut Node<T>,
}

impl<T> Node<T> {
    fn new(data: T) -> Box<Self> {
        Box::new(Self {
            data,
            next: ptr::null_mut(),
        })
    }
}

impl<T> LockFreeStack<T> {
    /// Create new empty stack
    pub const fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Push element onto stack
    pub fn push(&self, item: T) -> Result<()> {
        let new_node = Box::into_raw(Node::new(item));
        
        for _ in 0..MAX_CAS_RETRIES {
            let head = self.head.load(Ordering::Acquire);
            unsafe {
                (*new_node).next = head;
            }
            
            match self.head.compare_exchange_weak(
                head,
                new_node,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(_) => continue,
            }
        }
        
        // Failed to push after retries, clean up
        unsafe {
            drop(Box::from_raw(new_node));
        }
        Err(LockFreeError::ConcurrentModification)
    }

    /// Pop element from stack
    pub fn pop(&self) -> Result<T> {
        for _ in 0..MAX_CAS_RETRIES {
            let head = self.head.load(Ordering::Acquire);
            
            if head.is_null() {
                return Err(LockFreeError::StackEmpty);
            }
            
            let next = unsafe { (*head).next };
            
            match self.head.compare_exchange_weak(
                head,
                next,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let data = unsafe {
                        let node = Box::from_raw(head);
                        node.data
                    };
                    return Ok(data);
                }
                Err(_) => continue,
            }
        }
        
        Err(LockFreeError::ConcurrentModification)
    }

    /// Check if stack is empty
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire).is_null()
    }

    /// Peek at top element without removing it
    /// Note: This is inherently racy in concurrent environments
    pub fn peek(&self) -> Option<&T> {
        let head = self.head.load(Ordering::Acquire);
        if head.is_null() {
            None
        } else {
            unsafe { Some(&(*head).data) }
        }
    }

    /// Clear all elements from stack
    pub fn clear(&self) {
        while self.pop().is_ok() {}
    }

    /// Get approximate size (expensive operation)
    pub fn len(&self) -> usize {
        let mut count = 0;
        let mut current = self.head.load(Ordering::Acquire);
        
        while !current.is_null() {
            count += 1;
            current = unsafe { (*current).next };
            
            // Prevent infinite loops in case of corruption
            if count > 1000000 {
                break;
            }
        }
        
        count
    }
}

impl<T> Default for LockFreeStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: Send> Send for LockFreeStack<T> {}
unsafe impl<T: Send> Sync for LockFreeStack<T> {}

impl<T> Drop for LockFreeStack<T> {
    fn drop(&mut self) {
        // Clean up all remaining nodes
        self.clear();
    }
}

/// Stack-based memory pool for efficient allocation
pub struct StackPool<T> {
    free_list: LockFreeStack<Box<T>>,
    capacity: usize,
}

impl<T> StackPool<T> {
    /// Create new stack pool with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            free_list: LockFreeStack::new(),
            capacity,
        }
    }

    /// Allocate object from pool or create new one
    pub fn allocate(&self) -> Box<T> 
    where
        T: Default,
    {
        match self.free_list.pop() {
            Ok(item) => item,
            Err(_) => Box::new(T::default()),
        }
    }

    /// Return object to pool for reuse
    pub fn deallocate(&self, item: Box<T>) -> Result<()> {
        if self.free_list.len() < self.capacity {
            self.free_list.push(item)
        } else {
            // Pool is full, just drop the item
            drop(item);
            Ok(())
        }
    }

    /// Get current pool size
    pub fn size(&self) -> usize {
        self.free_list.len()
    }

    /// Check if pool is empty
    pub fn is_empty(&self) -> bool {
        self.free_list.is_empty()
    }

    /// Clear all items from pool
    pub fn clear(&self) {
        self.free_list.clear();
    }
}

/// Lock-free LIFO buffer for spike events
pub struct SpikeBuffer<T> {
    stack: LockFreeStack<T>,
    max_size: usize,
}

impl<T> SpikeBuffer<T> {
    /// Create new spike buffer with maximum size
    pub fn new(max_size: usize) -> Self {
        Self {
            stack: LockFreeStack::new(),
            max_size,
        }
    }

    /// Add spike event to buffer
    pub fn add_spike(&self, spike: T) -> Result<()> {
        if self.stack.len() >= self.max_size {
            // Remove oldest spike to make room
            let _ = self.stack.pop();
        }
        self.stack.push(spike)
    }

    /// Get most recent spike
    pub fn get_recent_spike(&self) -> Result<T> {
        self.stack.pop()
    }

    /// Peek at most recent spike without removing
    pub fn peek_recent_spike(&self) -> Option<&T> {
        self.stack.peek()
    }

    /// Get all spikes (drains the buffer)
    pub fn drain_spikes(&self) -> Vec<T> 
    where
        T: Clone,
    {
        let mut spikes = Vec::new();
        while let Ok(spike) = self.stack.pop() {
            spikes.push(spike);
        }
        spikes.reverse(); // Restore chronological order
        spikes
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.stack.len().min(self.max_size)
    }

    /// Clear all spikes from buffer
    pub fn clear(&self) {
        self.stack.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_free_stack() {
        let stack = LockFreeStack::new();
        
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
        
        // Push some items
        for i in 0..10 {
            stack.push(i).unwrap();
        }
        
        assert!(!stack.is_empty());
        assert_eq!(stack.len(), 10);
        
        // Peek at top
        assert_eq!(stack.peek(), Some(&9));
        
        // Pop items (LIFO order)
        for i in (0..10).rev() {
            assert_eq!(stack.pop().unwrap(), i);
        }
        
        assert!(stack.is_empty());
        assert!(stack.pop().is_err());
    }

    #[test]
    fn test_stack_pool() {
        let pool = StackPool::<i32>::new(5);
        
        assert!(pool.is_empty());
        
        // Allocate some items
        let item1 = pool.allocate();
        let item2 = pool.allocate();
        
        // Return to pool
        pool.deallocate(item1).unwrap();
        pool.deallocate(item2).unwrap();
        
        assert_eq!(pool.size(), 2);
        
        // Allocate again (should reuse)
        let _reused = pool.allocate();
        assert_eq!(pool.size(), 1);
    }

    #[test]
    fn test_spike_buffer() {
        let buffer = SpikeBuffer::new(3);
        
        assert!(buffer.is_empty());
        
        // Add spikes
        buffer.add_spike(1).unwrap();
        buffer.add_spike(2).unwrap();
        buffer.add_spike(3).unwrap();
        
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.peek_recent_spike(), Some(&3));
        
        // Add one more (should evict oldest)
        buffer.add_spike(4).unwrap();
        assert_eq!(buffer.len(), 3);
        
        // Get recent spike
        assert_eq!(buffer.get_recent_spike().unwrap(), 4);
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_spike_buffer_drain() {
        let buffer = SpikeBuffer::new(5);
        
        for i in 1..=3 {
            buffer.add_spike(i).unwrap();
        }
        
        let spikes = buffer.drain_spikes();
        assert_eq!(spikes, vec![1, 2, 3]);
        assert!(buffer.is_empty());
    }
}