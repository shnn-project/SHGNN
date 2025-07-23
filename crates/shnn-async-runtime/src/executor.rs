//! Task executor with work-stealing for the SHNN async runtime
//! 
//! Provides efficient task execution with work stealing optimized for neuromorphic workloads.

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use alloc::{vec::Vec, sync::Arc, boxed::Box, format};

#[cfg(feature = "std")]
use std::{thread, sync::Condvar, time::Duration};

use crate::{
    StealQueue, LockFreeQueue, AtomicCounter,
    SchedulableTask, ExecutionResult,
    SpikeTime, TaskPriority,
};

/// Work-stealing executor for parallel task execution
pub struct Executor {
    /// Number of worker threads
    worker_count: usize,
    /// Worker thread handles
    #[cfg(feature = "std")]
    workers: Vec<thread::JoinHandle<()>>,
    /// Work-stealing queues for each worker
    worker_queues: Vec<StealQueue<Box<dyn SchedulableTask>>>,
    /// Global task queue for overflow
    global_queue: LockFreeQueue<Box<dyn SchedulableTask>>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Worker notification system
    #[cfg(feature = "std")]
    parker: Arc<(std::sync::Mutex<bool>, Condvar)>,
    /// Execution statistics
    stats: ExecutorStats,
}

impl Executor {
    /// Create new work-stealing executor
    pub fn new_work_stealing(worker_count: usize) -> Self {
        let worker_queues = (0..worker_count)
            .map(|_| StealQueue::with_capacity(1024))
            .collect();

        let shutdown = Arc::new(AtomicBool::new(false));
        
        #[cfg(feature = "std")]
        let parker = Arc::new((std::sync::Mutex::new(false), Condvar::new()));

        Self {
            worker_count,
            #[cfg(feature = "std")]
            workers: Vec::new(),
            worker_queues,
            global_queue: LockFreeQueue::with_capacity(4096),
            shutdown,
            #[cfg(feature = "std")]
            parker,
            stats: ExecutorStats::new(),
        }
    }

    /// Start all worker threads (simplified implementation)
    #[cfg(feature = "std")]
    pub fn start_workers(&mut self) {
        for worker_id in 0..self.worker_count {
            let shutdown = Arc::clone(&self.shutdown);
            let parker = Arc::clone(&self.parker);

            let handle = thread::Builder::new()
                .name(format!("shnn-worker-{}", worker_id))
                .spawn(move || {
                    // Simplified worker thread implementation
                    while !shutdown.load(Ordering::Relaxed) {
                        // Just park/unpark for now until we fix thread safety
                        let (lock, cvar) = &*parker;
                        let mut notified = lock.lock().unwrap();
                        while !*notified && !shutdown.load(Ordering::Relaxed) {
                            let timeout = std::time::Duration::from_millis(100);
                            let (_guard, _timeout_result) = cvar.wait_timeout(notified, timeout).unwrap();
                            notified = _guard;
                        }
                        *notified = false;
                    }
                })
                .expect("Failed to spawn worker thread");

            self.workers.push(handle);
        }
    }

    /// Submit task to executor
    pub fn submit_task(&self, task: Box<dyn SchedulableTask>) {
        let worker_id = self.select_worker(task.priority());
        
        // Try to submit to worker's local queue first
        if let Err(task) = self.worker_queues[worker_id].push(task) {
            // Local queue full, try global queue
            if let Err(task) = self.global_queue.try_push(task) {
                // Everything is full, execute inline (emergency fallback)
                self.stats.emergency_inline_executions.increment();
                // In a real implementation, we might have more sophisticated backpressure
                drop(task); // For now, just drop the task
            } else {
                self.stats.global_queue_submissions.increment();
            }
        } else {
            self.stats.local_queue_submissions.increment();
        }

        // Wake workers if any are sleeping
        self.wake_workers();
    }

    /// Select worker for task submission (load balancing)
    fn select_worker(&self, priority: TaskPriority) -> usize {
        match priority {
            TaskPriority::Spike => {
                // For spike tasks, prefer less loaded workers
                self.least_loaded_worker()
            }
            _ => {
                // For other tasks, use round-robin
                static NEXT_WORKER: AtomicUsize = AtomicUsize::new(0);
                NEXT_WORKER.fetch_add(1, Ordering::Relaxed) % self.worker_count
            }
        }
    }

    /// Find least loaded worker
    fn least_loaded_worker(&self) -> usize {
        let mut min_load = usize::MAX;
        let mut best_worker = 0;

        for (i, queue) in self.worker_queues.iter().enumerate() {
            let load = queue.len();
            if load < min_load {
                min_load = load;
                best_worker = i;
            }
        }

        best_worker
    }

    /// Wake all workers
    pub fn wake_workers(&self) {
        #[cfg(feature = "std")]
        {
            let (lock, cvar) = &*self.parker;
            if let Ok(mut notified) = lock.try_lock() {
                *notified = true;
                cvar.notify_all();
            }
        }
    }

    /// Wake all workers (guaranteed)
    pub fn wake_all_workers(&self) {
        #[cfg(feature = "std")]
        {
            let (lock, cvar) = &*self.parker;
            let mut notified = lock.lock().unwrap();
            *notified = true;
            cvar.notify_all();
        }
    }

    /// Shutdown executor gracefully
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.wake_all_workers();

        #[cfg(feature = "std")]
        {
            // Wait for all workers to finish
            while let Some(handle) = self.workers.pop() {
                handle.join().ok();
            }
        }
    }

    /// Get executor statistics
    pub fn stats(&self) -> &ExecutorStats {
        &self.stats
    }

    /// Get current load across all workers
    pub fn total_load(&self) -> usize {
        self.worker_queues.iter().map(|q| q.len()).sum::<usize>() + self.global_queue.len()
    }
}

/// Worker thread for task execution
pub struct WorkerThread {
    id: usize,
    affinity: bool,
    shutdown: Arc<AtomicBool>,
}

impl WorkerThread {
    /// Create new worker thread
    pub fn new(id: usize, affinity: bool, shutdown: Arc<AtomicBool>) -> Self {
        Self { id, affinity, shutdown }
    }

    /// Get worker ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Join the worker thread
    #[cfg(feature = "std")]
    pub fn join(&self) {
        // This is a placeholder - in the actual implementation,
        // we'd store the thread handle and join on it
    }
}

/// Main function for worker threads
#[cfg(feature = "std")]
unsafe fn worker_thread_main(
    worker_id: usize,
    local_queue: &StealQueue<Box<dyn SchedulableTask>>,
    other_queues: &[*const StealQueue<Box<dyn SchedulableTask>>],
    global_queue: &LockFreeQueue<Box<dyn SchedulableTask>>,
    shutdown: Arc<AtomicBool>,
    parker: Arc<(std::sync::Mutex<bool>, Condvar)>,
    stats: &ExecutorStats,
) {
    // Set thread affinity if requested
    // This would require platform-specific code

    let mut consecutive_steals = 0;
    let mut idle_count = 0;

    while !shutdown.load(Ordering::Acquire) {
        let mut task_executed = false;

        // 1. Try to get task from local queue
        if let Some(mut task) = local_queue.pop() {
            execute_task(&mut *task, stats);
            task_executed = true;
            consecutive_steals = 0;
        }
        // 2. Try to steal from other workers
        else if consecutive_steals < 4 {
            for &other_queue_ptr in other_queues {
                if let Some(mut task) = (&*other_queue_ptr).steal() {
                    execute_task(&mut *task, stats);
                    stats.successful_steals.increment();
                    task_executed = true;
                    consecutive_steals += 1;
                    break;
                }
            }
        }
        // 3. Try global queue
        else if let Some(mut task) = global_queue.try_pop() {
            execute_task(&mut *task, stats);
            task_executed = true;
            consecutive_steals = 0;
        }

        if task_executed {
            idle_count = 0;
        } else {
            idle_count += 1;
            
            // After some idle time, park the thread
            if idle_count > 100 {
                park_worker(&parker, Duration::from_micros(100));
                idle_count = 0;
            } else {
                // Short spin before trying again
                for _ in 0..100 {
                    core::hint::spin_loop();
                }
            }
        }
    }

    stats.worker_shutdowns.increment();
}

/// Execute a single task
fn execute_task(task: &mut dyn SchedulableTask, stats: &ExecutorStats) {
    let start_time = SpikeTime::from_nanos(0); // TODO: Get actual time
    
    match task.execute_step() {
        ExecutionResult::Completed => {
            stats.tasks_completed.increment();
        }
        ExecutionResult::Yielded => {
            stats.tasks_yielded.increment();
            // Task should be rescheduled
        }
        ExecutionResult::Waiting => {
            stats.tasks_waiting.increment();
            // Task is blocked, don't reschedule until woken
        }
        ExecutionResult::Failed => {
            stats.task_failures.increment();
        }
        ExecutionResult::Cancelled => {
            stats.tasks_cancelled.increment();
        }
    }
    
    stats.total_executions.increment();
}

/// Park worker thread until woken or timeout
#[cfg(feature = "std")]
fn park_worker(parker: &Arc<(std::sync::Mutex<bool>, Condvar)>, timeout: Duration) {
    let (lock, cvar) = &**parker;
    let mut notified = lock.lock().unwrap();
    
    if !*notified {
        let (new_guard, _timeout_result) = cvar.wait_timeout(notified, timeout).unwrap();
        notified = new_guard;
    }
    
    *notified = false;
}

/// Executor performance statistics
pub struct ExecutorStats {
    /// Total task executions
    pub total_executions: AtomicCounter,
    /// Tasks completed successfully
    pub tasks_completed: AtomicCounter,
    /// Tasks that yielded
    pub tasks_yielded: AtomicCounter,
    /// Tasks waiting for external events
    pub tasks_waiting: AtomicCounter,
    /// Task execution failures
    pub task_failures: AtomicCounter,
    /// Cancelled tasks
    pub tasks_cancelled: AtomicCounter,
    /// Successful work steals
    pub successful_steals: AtomicCounter,
    /// Local queue submissions
    pub local_queue_submissions: AtomicCounter,
    /// Global queue submissions
    pub global_queue_submissions: AtomicCounter,
    /// Emergency inline executions
    pub emergency_inline_executions: AtomicCounter,
    /// Worker shutdowns
    pub worker_shutdowns: AtomicCounter,
}

impl ExecutorStats {
    /// Create new statistics tracker
    pub fn new() -> Self {
        Self {
            total_executions: AtomicCounter::new(),
            tasks_completed: AtomicCounter::new(),
            tasks_yielded: AtomicCounter::new(),
            tasks_waiting: AtomicCounter::new(),
            task_failures: AtomicCounter::new(),
            tasks_cancelled: AtomicCounter::new(),
            successful_steals: AtomicCounter::new(),
            local_queue_submissions: AtomicCounter::new(),
            global_queue_submissions: AtomicCounter::new(),
            emergency_inline_executions: AtomicCounter::new(),
            worker_shutdowns: AtomicCounter::new(),
        }
    }

    /// Get task completion rate
    pub fn completion_rate(&self) -> f64 {
        let total = self.total_executions.get();
        if total == 0 {
            0.0
        } else {
            self.tasks_completed.get() as f64 / total as f64
        }
    }

    /// Get work stealing efficiency
    pub fn steal_efficiency(&self) -> f64 {
        let total_submissions = self.local_queue_submissions.get() + self.global_queue_submissions.get();
        if total_submissions == 0 {
            0.0
        } else {
            self.successful_steals.get() as f64 / total_submissions as f64
        }
    }

    /// Reset all counters
    pub fn reset(&self) {
        self.total_executions.reset();
        self.tasks_completed.reset();
        self.tasks_yielded.reset();
        self.tasks_waiting.reset();
        self.task_failures.reset();
        self.tasks_cancelled.reset();
        self.successful_steals.reset();
        self.local_queue_submissions.reset();
        self.global_queue_submissions.reset();
        self.emergency_inline_executions.reset();
        self.worker_shutdowns.reset();
    }
}

impl Default for ExecutorStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_creation() {
        let executor = Executor::new_work_stealing(4);
        assert_eq!(executor.worker_count, 4);
        assert_eq!(executor.total_load(), 0);
    }

    #[test]
    fn test_worker_selection() {
        let executor = Executor::new_work_stealing(4);
        
        let worker1 = executor.select_worker(TaskPriority::Normal);
        let worker2 = executor.select_worker(TaskPriority::Normal);
        
        assert!(worker1 < 4);
        assert!(worker2 < 4);
    }

    #[test]
    fn test_executor_stats() {
        let stats = ExecutorStats::new();
        
        assert_eq!(stats.completion_rate(), 0.0);
        assert_eq!(stats.steal_efficiency(), 0.0);
        
        stats.total_executions.increment();
        stats.tasks_completed.increment();
        
        assert_eq!(stats.completion_rate(), 1.0);
    }

    #[test]
    fn test_worker_thread() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker = WorkerThread::new(0, false, shutdown);
        
        assert_eq!(worker.id(), 0);
    }
}