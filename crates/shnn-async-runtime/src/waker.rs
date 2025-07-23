//! Custom waker implementation for the SHNN async runtime
//! 
//! Provides efficient waking mechanisms optimized for neuromorphic workloads.

use core::{
    task::{RawWaker, RawWakerVTable, Waker},
    sync::atomic::{AtomicUsize, Ordering},
};
use alloc::{sync::Arc, boxed::Box, vec::Vec};

use crate::{TaskId, LockFreeQueue};

/// Custom waker alias for compatibility
pub type CustomWaker = SHNNWaker;

/// SHNN-specific waker for task notifications
pub struct SHNNWaker {
    /// Task ID to wake
    task_id: TaskId,
    /// Waker queue for batched notifications
    waker_queue: Arc<LockFreeQueue<TaskId>>,
    /// Wake counter for debugging
    wake_count: AtomicUsize,
}

impl SHNNWaker {
    /// Create new SHNN waker
    pub fn new(task_id: TaskId, waker_queue: Arc<LockFreeQueue<TaskId>>) -> Self {
        Self {
            task_id,
            waker_queue,
            wake_count: AtomicUsize::new(0),
        }
    }

    /// Get the task ID this waker is for
    pub fn task_id(&self) -> TaskId {
        self.task_id
    }

    /// Get number of times this waker has been called
    pub fn wake_count(&self) -> usize {
        self.wake_count.load(Ordering::Relaxed)
    }

    /// Wake the associated task
    pub fn wake_task(&self) {
        self.wake_count.fetch_add(1, Ordering::Relaxed);
        
        // Add task to wake queue for batch processing
        let _ = self.waker_queue.try_push(self.task_id);
    }

    /// Create a standard Waker from this SHNNWaker
    pub fn into_waker(self) -> Waker {
        let boxed = Box::new(self);
        let raw_waker = RawWaker::new(
            Box::into_raw(boxed) as *const (),
            &SHNN_WAKER_VTABLE,
        );
        unsafe { Waker::from_raw(raw_waker) }
    }

    /// Create a standard Waker from SHNNWaker reference
    pub fn as_waker(&self) -> Waker {
        let clone = SHNNWaker {
            task_id: self.task_id,
            waker_queue: Arc::clone(&self.waker_queue),
            wake_count: AtomicUsize::new(0),
        };
        clone.into_waker()
    }
}

impl Clone for SHNNWaker {
    fn clone(&self) -> Self {
        Self {
            task_id: self.task_id,
            waker_queue: Arc::clone(&self.waker_queue),
            wake_count: AtomicUsize::new(0),
        }
    }
}

/// Virtual table for SHNN waker operations
static SHNN_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    shnn_waker_clone,
    shnn_waker_wake,
    shnn_waker_wake_by_ref,
    shnn_waker_drop,
);

/// Clone function for SHNN waker
unsafe fn shnn_waker_clone(data: *const ()) -> RawWaker {
    let waker = &*(data as *const SHNNWaker);
    let cloned = waker.clone();
    let boxed = Box::new(cloned);
    RawWaker::new(
        Box::into_raw(boxed) as *const (),
        &SHNN_WAKER_VTABLE,
    )
}

/// Wake function for SHNN waker (consumes the waker)
unsafe fn shnn_waker_wake(data: *const ()) {
    let waker = Box::from_raw(data as *mut SHNNWaker);
    waker.wake_task();
}

/// Wake by reference function for SHNN waker
unsafe fn shnn_waker_wake_by_ref(data: *const ()) {
    let waker = &*(data as *const SHNNWaker);
    waker.wake_task();
}

/// Drop function for SHNN waker
unsafe fn shnn_waker_drop(data: *const ()) {
    let _ = Box::from_raw(data as *mut SHNNWaker);
}

/// Waker registry for managing task wakers
pub struct WakerRegistry {
    /// Wake notification queue
    wake_queue: Arc<LockFreeQueue<TaskId>>,
    /// Registry statistics
    stats: WakerStats,
}

impl WakerRegistry {
    /// Create new waker registry
    pub fn new() -> Self {
        Self {
            wake_queue: Arc::new(LockFreeQueue::with_capacity(4096)),
            stats: WakerStats::new(),
        }
    }

    /// Create waker for task
    pub fn create_waker(&self, task_id: TaskId) -> Waker {
        self.stats.wakers_created.fetch_add(1, Ordering::Relaxed);
        
        let shnn_waker = SHNNWaker::new(task_id, Arc::clone(&self.wake_queue));
        shnn_waker.into_waker()
    }

    /// Process pending wake notifications
    pub fn process_wake_notifications(&self) -> Vec<TaskId> {
        let mut woken_tasks = Vec::new();
        
        while let Some(task_id) = self.wake_queue.try_pop() {
            woken_tasks.push(task_id);
            self.stats.tasks_woken.fetch_add(1, Ordering::Relaxed);
        }
        
        woken_tasks
    }

    /// Get wake queue length
    pub fn pending_wakes(&self) -> usize {
        self.wake_queue.len()
    }

    /// Get waker statistics
    pub fn stats(&self) -> &WakerStats {
        &self.stats
    }
}

impl Default for WakerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Waker performance statistics
pub struct WakerStats {
    /// Number of wakers created
    pub wakers_created: AtomicUsize,
    /// Number of tasks woken
    pub tasks_woken: AtomicUsize,
    /// Number of spurious wakes
    pub spurious_wakes: AtomicUsize,
    /// Number of batched wake operations
    pub batched_wakes: AtomicUsize,
}

impl WakerStats {
    /// Create new waker statistics
    pub fn new() -> Self {
        Self {
            wakers_created: AtomicUsize::new(0),
            tasks_woken: AtomicUsize::new(0),
            spurious_wakes: AtomicUsize::new(0),
            batched_wakes: AtomicUsize::new(0),
        }
    }

    /// Get wake efficiency (tasks woken per waker created)
    pub fn wake_efficiency(&self) -> f64 {
        let created = self.wakers_created.load(Ordering::Relaxed);
        if created == 0 {
            0.0
        } else {
            self.tasks_woken.load(Ordering::Relaxed) as f64 / created as f64
        }
    }

    /// Get spurious wake rate
    pub fn spurious_wake_rate(&self) -> f64 {
        let total_wakes = self.tasks_woken.load(Ordering::Relaxed);
        if total_wakes == 0 {
            0.0
        } else {
            self.spurious_wakes.load(Ordering::Relaxed) as f64 / total_wakes as f64
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.wakers_created.store(0, Ordering::Relaxed);
        self.tasks_woken.store(0, Ordering::Relaxed);
        self.spurious_wakes.store(0, Ordering::Relaxed);
        self.batched_wakes.store(0, Ordering::Relaxed);
    }
}

impl Default for WakerStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch waker for efficient bulk task notifications
pub struct BatchWaker {
    /// Tasks to wake
    tasks: LockFreeQueue<TaskId>,
    /// Waker registry
    registry: Arc<WakerRegistry>,
    /// Batch size threshold
    batch_threshold: usize,
}

impl BatchWaker {
    /// Create new batch waker
    pub fn new(registry: Arc<WakerRegistry>, batch_threshold: usize) -> Self {
        Self {
            tasks: LockFreeQueue::with_capacity(batch_threshold * 2),
            registry,
            batch_threshold,
        }
    }

    /// Add task to batch
    pub fn add_task(&self, task_id: TaskId) {
        if self.tasks.try_push(task_id).is_ok() {
            // Check if we should flush the batch
            if self.tasks.len() >= self.batch_threshold {
                self.flush();
            }
        }
    }

    /// Flush all pending tasks
    pub fn flush(&self) {
        let mut batch_count = 0;
        
        while let Some(task_id) = self.tasks.try_pop() {
            // Create and wake the task
            let waker = self.registry.create_waker(task_id);
            waker.wake();
            batch_count += 1;
        }
        
        if batch_count > 0 {
            self.registry.stats.batched_wakes.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get pending task count
    pub fn pending_count(&self) -> usize {
        self.tasks.len()
    }
}

/// Utility functions for waker management
pub mod utils {
    use super::*;

    /// Create a no-op waker for testing
    pub fn noop_waker() -> Waker {
        static NOOP_VTABLE: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(core::ptr::null(), &NOOP_VTABLE),
            |_| {},
            |_| {},
            |_| {},
        );

        let raw_waker = RawWaker::new(core::ptr::null(), &NOOP_VTABLE);
        unsafe { Waker::from_raw(raw_waker) }
    }

    /// Check if two wakers refer to the same task
    pub fn wakers_equal(waker1: &Waker, waker2: &Waker) -> bool {
        // This is a simplified implementation
        // In practice, you'd need to compare the underlying task IDs
        waker1.will_wake(waker2)
    }

    /// Count references to a waker
    pub fn waker_ref_count(_waker: &Waker) -> usize {
        // This would require accessing internal reference counting
        // For now, return a placeholder
        1
    }
}

/// Spike-specific waker optimizations
pub struct SpikeWaker {
    /// Base waker
    base: SHNNWaker,
    /// Spike priority flag
    is_spike_priority: bool,
    /// Deadline for spike processing
    deadline: Option<crate::SpikeTime>,
}

impl SpikeWaker {
    /// Create new spike waker with priority
    pub fn new_spike_priority(
        task_id: TaskId,
        waker_queue: Arc<LockFreeQueue<TaskId>>,
        deadline: Option<crate::SpikeTime>,
    ) -> Self {
        Self {
            base: SHNNWaker::new(task_id, waker_queue),
            is_spike_priority: true,
            deadline,
        }
    }

    /// Create new regular spike waker
    pub fn new_regular(
        task_id: TaskId,
        waker_queue: Arc<LockFreeQueue<TaskId>>,
    ) -> Self {
        Self {
            base: SHNNWaker::new(task_id, waker_queue),
            is_spike_priority: false,
            deadline: None,
        }
    }

    /// Check if this is a spike priority waker
    pub fn is_spike_priority(&self) -> bool {
        self.is_spike_priority
    }

    /// Get deadline if set
    pub fn deadline(&self) -> Option<crate::SpikeTime> {
        self.deadline
    }

    /// Wake with spike priority
    pub fn wake_spike(&self) {
        if self.is_spike_priority {
            // For spike priority, we might use a different queue or mechanism
            self.base.wake_task();
        } else {
            self.base.wake_task();
        }
    }

    /// Convert to standard waker
    pub fn into_waker(self) -> Waker {
        self.base.into_waker()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shnn_waker() {
        let registry = WakerRegistry::new();
        let task_id = TaskId::new();
        
        let waker = registry.create_waker(task_id);
        assert!(waker.will_wake(&waker));
        
        waker.wake_by_ref();
        let woken_tasks = registry.process_wake_notifications();
        assert_eq!(woken_tasks.len(), 1);
        assert_eq!(woken_tasks[0], task_id);
    }

    #[test]
    fn test_waker_registry() {
        let registry = WakerRegistry::new();
        assert_eq!(registry.pending_wakes(), 0);
        
        let task_id = TaskId::new();
        let waker = registry.create_waker(task_id);
        waker.wake();
        
        assert!(registry.pending_wakes() > 0);
        let woken = registry.process_wake_notifications();
        assert_eq!(woken.len(), 1);
        assert_eq!(registry.pending_wakes(), 0);
    }

    #[test]
    fn test_batch_waker() {
        let registry = Arc::new(WakerRegistry::new());
        let batch_waker = BatchWaker::new(Arc::clone(&registry), 3);
        
        let task1 = TaskId::new();
        let task2 = TaskId::new();
        let task3 = TaskId::new();
        
        batch_waker.add_task(task1);
        batch_waker.add_task(task2);
        assert_eq!(batch_waker.pending_count(), 2);
        
        batch_waker.add_task(task3); // Should trigger flush
        assert_eq!(batch_waker.pending_count(), 0);
    }

    #[test]
    fn test_spike_waker() {
        let registry = WakerRegistry::new();
        let task_id = TaskId::new();
        let deadline = Some(crate::SpikeTime::from_millis(10));
        
        let spike_waker = SpikeWaker::new_spike_priority(
            task_id,
            Arc::clone(&registry.wake_queue),
            deadline,
        );
        
        assert!(spike_waker.is_spike_priority());
        assert_eq!(spike_waker.deadline(), deadline);
    }

    #[test]
    fn test_waker_stats() {
        let stats = WakerStats::new();
        
        assert_eq!(stats.wake_efficiency(), 0.0);
        assert_eq!(stats.spurious_wake_rate(), 0.0);
        
        stats.wakers_created.fetch_add(2, Ordering::Relaxed);
        stats.tasks_woken.fetch_add(3, Ordering::Relaxed);
        
        assert_eq!(stats.wake_efficiency(), 1.5);
    }

    #[test]
    fn test_noop_waker() {
        let waker = utils::noop_waker();
        
        // Should not panic
        waker.wake_by_ref();
        waker.wake();
    }
}