//! Task management for the SHNN async runtime
//! 
//! Provides lightweight task abstractions optimized for neuromorphic workloads.

use core::{
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    ptr::NonNull,
};
use alloc::{boxed::Box, sync::Arc};

use crate::{SpikeTime, waker::SHNNWaker};

/// Unique task identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(u64);

impl TaskId {
    /// Generate new unique task ID
    pub fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

/// Task priority levels for neuromorphic scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Real-time spike processing (highest priority)
    Spike = 0,
    /// High-priority neural computation
    Neural = 1,
    /// Normal async tasks
    Normal = 2,
    /// Background/maintenance tasks
    Background = 3,
}

impl TaskPriority {
    /// Get priority as numeric value for scheduling
    pub fn as_usize(&self) -> usize {
        *self as usize
    }
}

/// Task state tracking
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskState {
    /// Task is ready to run
    Ready,
    /// Task is currently running
    Running,
    /// Task is waiting (blocked)
    Waiting,
    /// Task completed successfully
    Completed,
    /// Task was cancelled
    Cancelled,
}

/// Generic task for the async runtime
pub struct Task {
    id: TaskId,
    priority: TaskPriority,
    future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
    state: TaskState,
    waker: Option<Waker>,
    spawn_time: SpikeTime,
    execution_count: usize,
}

impl Task {
    /// Create new task with specified priority
    pub fn new<F>(future: F, priority: TaskPriority) -> Self
    where
        F: Future<Output = ()> + Send + 'static,
    {
        Self {
            id: TaskId::new(),
            priority,
            future: Box::pin(future),
            state: TaskState::Ready,
            waker: None,
            spawn_time: SpikeTime::from_nanos(0), // TODO: Use actual time
            execution_count: 0,
        }
    }

    /// Get task ID
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Get task priority
    pub fn priority(&self) -> TaskPriority {
        self.priority
    }

    /// Get current task state
    pub fn state(&self) -> TaskState {
        self.state
    }

    /// Set task state
    pub fn set_state(&mut self, state: TaskState) {
        self.state = state;
    }

    /// Poll the task future
    pub fn poll(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        self.execution_count += 1;
        self.state = TaskState::Running;
        
        let poll_result = self.future.as_mut().poll(cx);
        
        match poll_result {
            Poll::Ready(()) => {
                self.state = TaskState::Completed;
            }
            Poll::Pending => {
                self.state = TaskState::Waiting;
                self.waker = Some(cx.waker().clone());
            }
        }
        
        poll_result
    }

    /// Wake the task if it has a waker
    pub fn wake(&mut self) {
        if let Some(waker) = self.waker.take() {
            self.state = TaskState::Ready;
            waker.wake();
        }
    }

    /// Get spawn time
    pub fn spawn_time(&self) -> SpikeTime {
        self.spawn_time
    }

    /// Get execution count
    pub fn execution_count(&self) -> usize {
        self.execution_count
    }
}

/// High-priority spike processing task
pub struct SpikeTask<T> {
    id: TaskId,
    future: Pin<Box<dyn Future<Output = T> + Send + 'static>>,
    state: TaskState,
    waker: Option<Waker>,
    result: Option<T>,
    deadline: Option<SpikeTime>,
    execution_count: usize,
}

impl<T> SpikeTask<T>
where
    T: Send + 'static,
{
    /// Create new spike processing task
    pub fn new<F>(future: F, priority: TaskPriority) -> Self
    where
        F: Future<Output = T> + Send + 'static,
    {
        Self {
            id: TaskId::new(),
            future: Box::pin(future),
            state: TaskState::Ready,
            waker: None,
            result: None,
            deadline: None,
            execution_count: 0,
        }
    }

    /// Set deadline for task completion
    pub fn with_deadline(mut self, deadline: SpikeTime) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Get task ID
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Get current state
    pub fn state(&self) -> TaskState {
        self.state
    }

    /// Check if task has deadline
    pub fn has_deadline(&self) -> bool {
        self.deadline.is_some()
    }

    /// Get deadline if set
    pub fn deadline(&self) -> Option<SpikeTime> {
        self.deadline
    }

    /// Poll the spike task
    pub fn poll(&mut self, cx: &mut Context<'_>) -> Poll<T> {
        if let Some(result) = self.result.take() {
            return Poll::Ready(result);
        }

        self.execution_count += 1;
        self.state = TaskState::Running;

        let poll_result = self.future.as_mut().poll(cx);

        match poll_result {
            Poll::Ready(result) => {
                self.state = TaskState::Completed;
                self.result = Some(result);
                Poll::Ready(self.result.take().unwrap())
            }
            Poll::Pending => {
                self.state = TaskState::Waiting;
                self.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }

    /// Wake the task
    pub fn wake(&mut self) {
        if let Some(waker) = self.waker.take() {
            self.state = TaskState::Ready;
            waker.wake();
        }
    }

    /// Get execution count
    pub fn execution_count(&self) -> usize {
        self.execution_count
    }
}

/// Handle for a regular task
pub struct TaskHandle<T> {
    id: TaskId,
    result: Arc<AtomicResult<T>>,
}

impl<T> TaskHandle<T>
where
    T: Send + 'static,
{
    /// Create new task handle
    pub fn new(id: TaskId) -> Self {
        Self {
            id,
            result: Arc::new(AtomicResult::new()),
        }
    }

    /// Get task ID
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Try to get result without blocking
    pub fn try_get_result(&self) -> Option<T> {
        self.result.try_take()
    }

    /// Set the result (called by runtime)
    pub fn set_result(&self, result: T) {
        self.result.set(result);
    }
}

/// Handle for a spike processing task
pub struct SpikeTaskHandle<T> {
    id: TaskId,
    result: Arc<AtomicResult<Result<T, SpikeTaskError>>>,
}

impl<T> SpikeTaskHandle<T>
where
    T: Send + 'static,
{
    /// Create new spike task handle
    pub fn new(id: TaskId) -> Self {
        Self {
            id,
            result: Arc::new(AtomicResult::new()),
        }
    }

    /// Get task ID
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Try to get result without blocking
    pub fn try_get_result(&self) -> Option<Result<T, SpikeTaskError>> {
        self.result.try_take()
    }

    /// Set the result (called by runtime)
    pub fn set_result(&self, result: Result<T, SpikeTaskError>) {
        self.result.set(result);
    }

    /// Await task completion
    pub async fn await_result(self) -> Result<T, SpikeTaskError> {
        // Simple polling loop - in a real implementation this would use proper async waiting
        loop {
            if let Some(result) = self.try_get_result() {
                return result;
            }
            
            // Yield to allow other tasks to run
            yield_now().await;
        }
    }
}

impl<T> core::future::Future for SpikeTaskHandle<T>
where
    T: Send + 'static,
{
    type Output = Result<T, SpikeTaskError>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(result) = self.try_get_result() {
            Poll::Ready(result)
        } else {
            Poll::Pending
        }
    }
}

/// Error types for spike task execution
#[derive(Debug, Clone, PartialEq)]
pub enum SpikeTaskError {
    /// Task execution timed out
    Timeout,
    /// Task was cancelled
    Cancelled,
    /// Task panicked during execution
    Panicked,
    /// Deadline was missed
    DeadlineMissed,
}

impl core::fmt::Display for SpikeTaskError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SpikeTaskError::Timeout => write!(f, "Task execution timed out"),
            SpikeTaskError::Cancelled => write!(f, "Task was cancelled"),
            SpikeTaskError::Panicked => write!(f, "Task panicked during execution"),
            SpikeTaskError::DeadlineMissed => write!(f, "Task deadline was missed"),
        }
    }
}

/// Thread-safe result container for task handles
struct AtomicResult<T> {
    value: core::sync::atomic::AtomicPtr<T>,
}

impl<T> AtomicResult<T> {
    fn new() -> Self {
        Self {
            value: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
        }
    }

    fn set(&self, result: T) {
        let boxed = Box::into_raw(Box::new(result));
        let old = self.value.swap(boxed, Ordering::Release);
        
        // Clean up old value if it exists
        if !old.is_null() {
            unsafe {
                let _ = Box::from_raw(old);
            }
        }
    }

    fn try_take(&self) -> Option<T> {
        let ptr = self.value.swap(core::ptr::null_mut(), Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            unsafe {
                Some(*Box::from_raw(ptr))
            }
        }
    }
}

impl<T> Drop for AtomicResult<T> {
    fn drop(&mut self) {
        let ptr = self.value.load(Ordering::Acquire);
        if !ptr.is_null() {
            unsafe {
                let _ = Box::from_raw(ptr);
            }
        }
    }
}

unsafe impl<T: Send> Send for AtomicResult<T> {}
unsafe impl<T: Send> Sync for AtomicResult<T> {}

/// Yield control to allow other tasks to run
pub async fn yield_now() {
    struct YieldFuture {
        yielded: bool,
    }

    impl Future for YieldFuture {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.yielded {
                Poll::Ready(())
            } else {
                self.yielded = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    YieldFuture { yielded: false }.await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_id_generation() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        assert_ne!(id1, id2);
        assert!(id2.as_u64() > id1.as_u64());
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Spike < TaskPriority::Neural);
        assert!(TaskPriority::Neural < TaskPriority::Normal);
        assert!(TaskPriority::Normal < TaskPriority::Background);
    }

    #[test]
    fn test_atomic_result() {
        let result = AtomicResult::new();
        assert!(result.try_take().is_none());
        
        result.set(42);
        assert_eq!(result.try_take(), Some(42));
        assert!(result.try_take().is_none());
    }

    #[test]
    fn test_task_handle() {
        let handle: TaskHandle<i32> = TaskHandle::new(TaskId::new());
        assert!(handle.try_get_result().is_none());
        
        handle.set_result(100);
        assert_eq!(handle.try_get_result(), Some(100));
    }

    #[cfg(feature = "std")]
    #[tokio::test]
    async fn test_yield_now() {
        yield_now().await;
        // Should complete without hanging
    }
}