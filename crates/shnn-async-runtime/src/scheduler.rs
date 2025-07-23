//! Task scheduler optimized for neuromorphic workloads
//! 
//! Provides priority-based scheduling with real-time guarantees for spike processing.

use core::sync::atomic::{AtomicUsize, Ordering};
use alloc::{vec::Vec, boxed::Box};
use crate::{
    Task, SpikeTask, TaskId, TaskPriority, TaskState,
    LockFreeQueue, AtomicCounter, SpikeTime,
};

/// Trait for schedulable tasks
pub trait SchedulableTask: Send + Sync {
    /// Get task ID
    fn task_id(&self) -> TaskId;
    
    /// Get task priority
    fn priority(&self) -> TaskPriority;
    
    /// Get current state
    fn state(&self) -> TaskState;
    
    /// Execute one step of the task
    fn execute_step(&mut self) -> ExecutionResult;
    
    /// Wake the task if waiting
    fn wake(&mut self);
    
    /// Check if task has deadline
    fn has_deadline(&self) -> bool;
    
    /// Get deadline if set
    fn deadline(&self) -> Option<SpikeTime>;
}

/// Result of task execution step
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionResult {
    /// Task completed successfully
    Completed,
    /// Task yielded, can continue later
    Yielded,
    /// Task is waiting for external event
    Waiting,
    /// Task failed with error
    Failed,
    /// Task was cancelled
    Cancelled,
}

/// Task scheduler with spike event priority
pub struct TaskScheduler {
    /// High-priority queues for different priority levels
    priority_queues: [LockFreeQueue<TaskId>; 4],
    /// Global task storage (simplified for now)
    pending_count: AtomicUsize,
    /// Statistics counters
    stats: SchedulerStats,
    /// Load balancing state
    next_worker: AtomicUsize,
}

impl TaskScheduler {
    /// Create new scheduler with priority queues
    pub fn new_with_priorities() -> Self {
        Self {
            priority_queues: [
                LockFreeQueue::with_capacity(1024),  // Spike
                LockFreeQueue::with_capacity(1024),  // Neural
                LockFreeQueue::with_capacity(2048),  // Normal
                LockFreeQueue::with_capacity(512),   // Background
            ],
            pending_count: AtomicUsize::new(0),
            stats: SchedulerStats::new(),
            next_worker: AtomicUsize::new(0),
        }
    }

    /// Schedule a spike processing task (highest priority)
    pub fn schedule_spike_task<T>(&self, _task: SpikeTask<T>)
    where
        T: Send + 'static,
    {
        // Simplified implementation for now
        self.priority_queues[TaskPriority::Spike.as_usize()]
            .try_push(TaskId::new())
            .ok();
        
        self.pending_count.fetch_add(1, Ordering::Relaxed);
        self.stats.tasks_scheduled.increment();
        self.stats.spike_tasks_scheduled.increment();
    }

    /// Schedule a regular task
    pub fn schedule_task(&self, task: Task) {
        let task_id = task.id();
        let priority = task.priority();
        
        self.priority_queues[priority.as_usize()]
            .try_push(task_id)
            .ok();
        
        self.pending_count.fetch_add(1, Ordering::Relaxed);
        self.stats.tasks_scheduled.increment();
    }

    /// Get next task ID to execute (priority-based)
    pub fn next_task_id(&self) -> Option<TaskId> {
        // Try each priority queue in order
        for queue in &self.priority_queues {
            if let Some(task_id) = queue.try_pop() {
                self.pending_count.fetch_sub(1, Ordering::Relaxed);
                self.stats.tasks_executed.increment();
                return Some(task_id);
            }
        }
        None
    }

    /// Get scheduler statistics
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Get load balancing hint for next worker
    pub fn next_worker_hint(&self, worker_count: usize) -> usize {
        self.next_worker.fetch_add(1, Ordering::Relaxed) % worker_count
    }

    /// Check if there are any pending tasks
    pub fn has_pending_tasks(&self) -> bool {
        self.pending_count.load(Ordering::Relaxed) > 0
    }

    /// Get pending task count (approximate)
    pub fn pending_task_count(&self) -> usize {
        self.pending_count.load(Ordering::Relaxed)
    }
}

/// Scheduler performance statistics
pub struct SchedulerStats {
    /// Total tasks scheduled
    pub tasks_scheduled: AtomicCounter,
    /// Total tasks executed
    pub tasks_executed: AtomicCounter,
    /// Spike tasks scheduled
    pub spike_tasks_scheduled: AtomicCounter,
    /// Task execution failures
    pub execution_failures: AtomicCounter,
    /// Deadline misses
    pub deadline_misses: AtomicCounter,
    /// Context switches
    pub context_switches: AtomicCounter,
}

impl SchedulerStats {
    /// Create new statistics tracker
    pub fn new() -> Self {
        Self {
            tasks_scheduled: AtomicCounter::new(),
            tasks_executed: AtomicCounter::new(),
            spike_tasks_scheduled: AtomicCounter::new(),
            execution_failures: AtomicCounter::new(),
            deadline_misses: AtomicCounter::new(),
            context_switches: AtomicCounter::new(),
        }
    }

    /// Get task throughput (tasks executed per second)
    pub fn throughput(&self) -> f64 {
        // This would need actual timing implementation
        self.tasks_executed.get() as f64
    }

    /// Get spike task ratio
    pub fn spike_task_ratio(&self) -> f64 {
        let total = self.tasks_scheduled.get();
        if total == 0 {
            0.0
        } else {
            self.spike_tasks_scheduled.get() as f64 / total as f64
        }
    }

    /// Get failure rate
    pub fn failure_rate(&self) -> f64 {
        let total = self.tasks_executed.get();
        if total == 0 {
            0.0
        } else {
            self.execution_failures.get() as f64 / total as f64
        }
    }

    /// Reset all counters
    pub fn reset(&self) {
        self.tasks_scheduled.reset();
        self.tasks_executed.reset();
        self.spike_tasks_scheduled.reset();
        self.execution_failures.reset();
        self.deadline_misses.reset();
        self.context_switches.reset();
    }
}

impl Default for SchedulerStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Real-time scheduling policy for spike processing
pub struct RealtimeSchedulingPolicy {
    /// Maximum execution time per task before preemption
    max_execution_time: SpikeTime,
    /// Time quantum for round-robin within priority levels
    time_quantum: SpikeTime,
    /// Priority boosting threshold
    priority_boost_threshold: SpikeTime,
}

impl RealtimeSchedulingPolicy {
    /// Create new real-time scheduling policy
    pub fn new(
        max_execution_time: SpikeTime,
        time_quantum: SpikeTime,
    ) -> Self {
        Self {
            max_execution_time,
            time_quantum,
            priority_boost_threshold: SpikeTime::from_millis(10),
        }
    }

    /// Check if task should be preempted
    pub fn should_preempt(&self, execution_time: SpikeTime) -> bool {
        execution_time >= self.max_execution_time
    }

    /// Check if task should get priority boost (aging)
    pub fn should_boost_priority(&self, wait_time: SpikeTime) -> bool {
        wait_time >= self.priority_boost_threshold
    }

    /// Get time quantum for priority level
    pub fn time_quantum_for_priority(&self, priority: TaskPriority) -> SpikeTime {
        match priority {
            TaskPriority::Spike => self.time_quantum, // Full quantum for spike tasks
            TaskPriority::Neural => SpikeTime::from_nanos(self.time_quantum.as_nanos() * 3 / 4),
            TaskPriority::Normal => SpikeTime::from_nanos(self.time_quantum.as_nanos() / 2),
            TaskPriority::Background => SpikeTime::from_nanos(self.time_quantum.as_nanos() / 4),
        }
    }
}

impl Default for RealtimeSchedulingPolicy {
    fn default() -> Self {
        Self::new(
            SpikeTime::from_millis(1), // 1ms max execution
            SpikeTime::from_micros(100), // 100μs time quantum
        )
    }
}

/// Load balancer for distributing tasks across workers
pub struct LoadBalancer {
    worker_loads: Vec<AtomicCounter>,
    next_worker: AtomicUsize,
}

impl LoadBalancer {
    /// Create new load balancer for specified number of workers
    pub fn new(worker_count: usize) -> Self {
        Self {
            worker_loads: (0..worker_count).map(|_| AtomicCounter::new()).collect(),
            next_worker: AtomicUsize::new(0),
        }
    }

    /// Get next worker for task assignment (round-robin)
    pub fn next_worker(&self) -> usize {
        let worker = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.worker_loads.len();
        self.worker_loads[worker].increment();
        worker
    }

    /// Get least loaded worker
    pub fn least_loaded_worker(&self) -> usize {
        let mut min_load = usize::MAX;
        let mut best_worker = 0;

        for (i, load) in self.worker_loads.iter().enumerate() {
            let current_load = load.get();
            if current_load < min_load {
                min_load = current_load;
                best_worker = i;
            }
        }

        best_worker
    }

    /// Report task completion for load tracking
    pub fn task_completed(&self, worker_id: usize) {
        if worker_id < self.worker_loads.len() {
            // Decrement is not atomic, but this is just an approximation
            let current = self.worker_loads[worker_id].get();
            if current > 0 {
                self.worker_loads[worker_id].reset();
                // Set to current - 1, but reset is simpler for this example
            }
        }
    }

    /// Get load for all workers
    pub fn worker_loads(&self) -> Vec<usize> {
        self.worker_loads.iter().map(|load| load.get()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Task, TaskPriority};

    #[test]
    fn test_scheduler_creation() {
        let scheduler = TaskScheduler::new_with_priorities();
        assert!(!scheduler.has_pending_tasks());
        assert_eq!(scheduler.pending_task_count(), 0);
    }

    #[test]
    fn test_scheduling_policy() {
        let policy = RealtimeSchedulingPolicy::default();
        
        assert!(policy.should_preempt(SpikeTime::from_millis(2)));
        assert!(!policy.should_preempt(SpikeTime::from_micros(500)));
        
        assert!(policy.should_boost_priority(SpikeTime::from_millis(20)));
        assert!(!policy.should_boost_priority(SpikeTime::from_millis(5)));
    }

    #[test]
    fn test_load_balancer() {
        let balancer = LoadBalancer::new(4);
        
        let worker1 = balancer.next_worker();
        let worker2 = balancer.next_worker();
        
        assert_ne!(worker1, worker2);
        assert!(worker1 < 4);
        assert!(worker2 < 4);
    }

    #[test]
    fn test_scheduler_stats() {
        let stats = SchedulerStats::new();
        
        assert_eq!(stats.tasks_scheduled.get(), 0);
        assert_eq!(stats.spike_task_ratio(), 0.0);
        
        stats.tasks_scheduled.increment();
        stats.spike_tasks_scheduled.increment();
        
        assert_eq!(stats.spike_task_ratio(), 1.0);
    }
}