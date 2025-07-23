//! Synchronization primitives for async runtime
//! 
//! Provides channels and other sync primitives optimized for neuromorphic computation
//! with zero external dependencies.

use alloc::collections::VecDeque;
use core::sync::atomic::{AtomicBool, Ordering};
use alloc::sync::Arc;
use crate::{Result, AsyncError};

/// Multi-producer, single-consumer channel
pub struct Channel<T> {
    sender: Sender<T>,
    receiver: Receiver<T>,
}

/// Sender half of a channel
pub struct Sender<T> {
    inner: Arc<ChannelInner<T>>,
}

/// Receiver half of a channel
pub struct Receiver<T> {
    inner: Arc<ChannelInner<T>>,
}

struct ChannelInner<T> {
    queue: spin::Mutex<VecDeque<T>>,
    closed: AtomicBool,
    capacity: usize,
}

impl<T> Channel<T> {
    /// Create a new bounded channel
    pub fn bounded(capacity: usize) -> (Sender<T>, Receiver<T>) {
        let inner = Arc::new(ChannelInner {
            queue: spin::Mutex::new(VecDeque::with_capacity(capacity)),
            closed: AtomicBool::new(false),
            capacity,
        });

        let sender = Sender { inner: inner.clone() };
        let receiver = Receiver { inner };

        (sender, receiver)
    }

    /// Create a new unbounded channel
    pub fn unbounded() -> (Sender<T>, Receiver<T>) {
        Self::bounded(usize::MAX)
    }
}

impl<T> Sender<T> {
    /// Send a value through the channel
    pub fn send(&self, value: T) -> Result<()> {
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(AsyncError::ChannelClosed);
        }

        let mut queue = self.inner.queue.lock();
        if queue.len() >= self.inner.capacity {
            return Err(AsyncError::ChannelFull);
        }

        queue.push_back(value);
        Ok(())
    }

    /// Try to send a value without blocking
    pub fn try_send(&self, value: T) -> Result<()> {
        self.send(value)
    }

    /// Close the channel
    pub fn close(&self) {
        self.inner.closed.store(true, Ordering::Release);
    }
}

impl<T> Receiver<T> {
    /// Receive a value from the channel
    pub fn recv(&self) -> Result<T> {
        let mut queue = self.inner.queue.lock();
        
        if let Some(value) = queue.pop_front() {
            return Ok(value);
        }

        if self.inner.closed.load(Ordering::Acquire) {
            return Err(AsyncError::ChannelClosed);
        }

        Err(AsyncError::ChannelEmpty)
    }

    /// Try to receive a value without blocking
    pub fn try_recv(&self) -> Result<T> {
        self.recv()
    }

    /// Check if the channel is closed
    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::Acquire)
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Create a bounded channel
pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    Channel::<T>::bounded(capacity)
}

/// Create an unbounded channel
pub fn unbounded<T>() -> (Sender<T>, Receiver<T>) {
    Channel::<T>::unbounded()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_send_recv() {
        let (sender, receiver) = channel::<i32>(10);
        
        sender.send(42).unwrap();
        let value = receiver.recv().unwrap();
        assert_eq!(value, 42);
    }

    #[test]
    fn test_channel_close() {
        let (sender, receiver) = channel::<i32>(10);
        
        sender.close();
        assert!(sender.send(42).is_err());
        assert!(receiver.is_closed());
    }
}