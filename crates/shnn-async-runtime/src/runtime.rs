//! Async runtime for neuromorphic computation
//! 
//! Provides a complete async runtime optimized for spike-based neural networks
//! with zero external dependencies.

use crate::{SHNNRuntime, executor::Executor, task::Task};

/// Re-export of the main runtime
pub type Runtime = SHNNRuntime;

/// Runtime builder for configuration
pub struct RuntimeBuilder {
    worker_threads: usize,
    spike_queue_capacity: usize,
    enable_work_stealing: bool,
}

impl RuntimeBuilder {
    /// Create new runtime builder
    pub fn new() -> Self {
        Self {
            worker_threads: 1,
            spike_queue_capacity: 1024,
            enable_work_stealing: true,
        }
    }

    /// Set number of worker threads
    pub fn worker_threads(mut self, threads: usize) -> Self {
        self.worker_threads = threads;
        self
    }

    /// Set spike queue capacity
    pub fn spike_queue_capacity(mut self, capacity: usize) -> Self {
        self.spike_queue_capacity = capacity;
        self
    }

    /// Enable or disable work stealing
    pub fn enable_work_stealing(mut self, enable: bool) -> Self {
        self.enable_work_stealing = enable;
        self
    }

    /// Build the runtime
    pub fn build(self) -> Runtime {
        Runtime::new(self.worker_threads, self.spike_queue_capacity)
    }
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self::new()
    }
}