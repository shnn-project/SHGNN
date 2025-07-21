//! Task scheduling for async neural networks
//!
//! This module provides intelligent task scheduling for optimal resource
//! utilization in neuromorphic processing.

use crate::error::{AsyncError, AsyncResult};
use shnn_core::time::{Time, Duration};

use std::{
    collections::{BinaryHeap, HashMap},
    cmp::Ordering,
    sync::{Arc, RwLock},
    time::Instant,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TaskPriority {
    /// Low priority background tasks
    Low = 0,
    /// Normal priority tasks
    Normal = 1,
    /// High priority tasks
    High = 2,
    /// Critical real-time tasks
    Critical = 3,
}

impl Default for TaskPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Scheduling policies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SchedulingPolicy {
    /// First In, First Out
    FIFO,
    /// Priority-based scheduling
    Priority,
    /// Round-robin scheduling
    RoundRobin,
    /// Shortest Job First
    SJF,
    /// Real-time scheduling
    Realtime,
}

impl Default for SchedulingPolicy {
    fn default() -> Self {
        Self::Priority
    }
}

/// Scheduled task
#[derive(Debug, Clone)]
pub struct ScheduledTask {
    /// Unique task ID
    pub id: u64,
    /// Task name
    pub name: String,
    /// Task priority
    pub priority: TaskPriority,
    /// Estimated execution time
    pub estimated_duration: Duration,
    /// Deadline (for real-time tasks)
    pub deadline: Option<Time>,
    /// Creation time
    pub created_at: Instant,
    /// Scheduled execution time
    pub scheduled_at: Option<Instant>,
    /// Task state
    pub state: TaskState,
}

/// Task state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TaskState {
    /// Task is waiting to be scheduled
    Pending,
    /// Task is scheduled but not running
    Scheduled,
    /// Task is currently running
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
    /// Task was cancelled
    Cancelled,
}

impl ScheduledTask {
    /// Create a new scheduled task
    pub fn new(
        id: u64,
        name: String,
        priority: TaskPriority,
        estimated_duration: Duration,
    ) -> Self {
        Self {
            id,
            name,
            priority,
            estimated_duration,
            deadline: None,
            created_at: Instant::now(),
            scheduled_at: None,
            state: TaskState::Pending,
        }
    }
    
    /// Create a real-time task with deadline
    pub fn realtime(
        id: u64,
        name: String,
        estimated_duration: Duration,
        deadline: Time,
    ) -> Self {
        Self {
            id,
            name,
            priority: TaskPriority::Critical,
            estimated_duration,
            deadline: Some(deadline),
            created_at: Instant::now(),
            scheduled_at: None,
            state: TaskState::Pending,
        }
    }
    
    /// Check if task has missed its deadline
    pub fn is_deadline_missed(&self, current_time: Time) -> bool {
        if let Some(deadline) = self.deadline {
            current_time > deadline
        } else {
            false
        }
    }
    
    /// Get task age
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
    
    /// Mark task as scheduled
    pub fn schedule(&mut self) {
        self.state = TaskState::Scheduled;
        self.scheduled_at = Some(Instant::now());
    }
    
    /// Mark task as running
    pub fn start(&mut self) {
        self.state = TaskState::Running;
    }
    
    /// Mark task as completed
    pub fn complete(&mut self) {
        self.state = TaskState::Completed;
    }
    
    /// Mark task as failed
    pub fn fail(&mut self) {
        self.state = TaskState::Failed;
    }
    
    /// Mark task as cancelled
    pub fn cancel(&mut self) {
        self.state = TaskState::Cancelled;
    }
}

impl PartialEq for ScheduledTask {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ScheduledTask {}

impl PartialOrd for ScheduledTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority tasks come first
        // If priorities are equal, older tasks come first
        match other.priority.cmp(&self.priority) {
            Ordering::Equal => self.created_at.cmp(&other.created_at),
            other => other,
        }
    }
}

/// Task scheduler
pub struct TaskScheduler {
    /// Scheduling policy
    policy: SchedulingPolicy,
    /// Task queue
    queue: Arc<RwLock<BinaryHeap<ScheduledTask>>>,
    /// Running tasks
    running_tasks: Arc<RwLock<HashMap<u64, ScheduledTask>>>,
    /// Completed tasks
    completed_tasks: Arc<RwLock<Vec<ScheduledTask>>>,
    /// Task ID counter
    next_task_id: Arc<std::sync::atomic::AtomicU64>,
    /// Scheduler statistics
    stats: Arc<RwLock<SchedulerStats>>,
}

impl TaskScheduler {
    /// Create a new task scheduler
    pub fn new() -> Self {
        Self::with_policy(SchedulingPolicy::default())
    }
    
    /// Create with specific scheduling policy
    pub fn with_policy(policy: SchedulingPolicy) -> Self {
        Self {
            policy,
            queue: Arc::new(RwLock::new(BinaryHeap::new())),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
            completed_tasks: Arc::new(RwLock::new(Vec::new())),
            next_task_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            stats: Arc::new(RwLock::new(SchedulerStats::default())),
        }
    }
    
    /// Submit a new task
    pub async fn submit_task(
        &self,
        name: String,
        priority: TaskPriority,
        estimated_duration: Duration,
    ) -> AsyncResult<u64> {
        let task_id = self.next_task_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let task = ScheduledTask::new(task_id, name.clone(), priority, estimated_duration);
        
        {
            let mut queue = self.queue.write().unwrap();
            queue.push(task);
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.tasks_submitted += 1;
            stats.pending_tasks += 1;
        }
        
        #[cfg(feature = "tracing")]
        debug!(task_id, name, ?priority, "Task submitted");
        
        Ok(task_id)
    }
    
    /// Submit a real-time task with deadline
    pub async fn submit_realtime_task(
        &self,
        name: String,
        estimated_duration: Duration,
        deadline: Time,
    ) -> AsyncResult<u64> {
        let task_id = self.next_task_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let task = ScheduledTask::realtime(task_id, name.clone(), estimated_duration, deadline);
        
        {
            let mut queue = self.queue.write().unwrap();
            queue.push(task);
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.tasks_submitted += 1;
            stats.pending_tasks += 1;
            stats.realtime_tasks += 1;
        }
        
        #[cfg(feature = "tracing")]
        debug!(task_id, name, ?deadline, "Real-time task submitted");
        
        Ok(task_id)
    }
    
    /// Get next task to execute
    pub async fn get_next_task(&self) -> Option<ScheduledTask> {
        let mut task = {
            let mut queue = self.queue.write().unwrap();
            queue.pop()
        };
        
        if let Some(ref mut t) = task {
            t.schedule();
            
            // Move to running tasks
            {
                let mut running = self.running_tasks.write().unwrap();
                running.insert(t.id, t.clone());
            }
            
            // Update statistics
            {
                let mut stats = self.stats.write().unwrap();
                stats.pending_tasks = stats.pending_tasks.saturating_sub(1);
                stats.running_tasks += 1;
            }
            
            #[cfg(feature = "tracing")]
            debug!(task_id = t.id, name = %t.name, "Task scheduled for execution");
        }
        
        task
    }
    
    /// Mark task as started
    pub async fn start_task(&self, task_id: u64) -> AsyncResult<()> {
        let mut running = self.running_tasks.write().unwrap();
        
        if let Some(task) = running.get_mut(&task_id) {
            task.start();
            
            #[cfg(feature = "tracing")]
            debug!(task_id, "Task started");
            
            Ok(())
        } else {
            Err(AsyncError::scheduling_with_task(task_id, "Task not found in running tasks"))
        }
    }
    
    /// Mark task as completed
    pub async fn complete_task(&self, task_id: u64) -> AsyncResult<()> {
        let task = {
            let mut running = self.running_tasks.write().unwrap();
            running.remove(&task_id)
        };
        
        if let Some(mut task) = task {
            task.complete();
            
            // Move to completed tasks
            {
                let mut completed = self.completed_tasks.write().unwrap();
                completed.push(task);
            }
            
            // Update statistics
            {
                let mut stats = self.stats.write().unwrap();
                stats.running_tasks = stats.running_tasks.saturating_sub(1);
                stats.completed_tasks += 1;
            }
            
            #[cfg(feature = "tracing")]
            debug!(task_id, "Task completed");
            
            Ok(())
        } else {
            Err(AsyncError::scheduling_with_task(task_id, "Task not found"))
        }
    }
    
    /// Mark task as failed
    pub async fn fail_task(&self, task_id: u64, reason: &str) -> AsyncResult<()> {
        let task = {
            let mut running = self.running_tasks.write().unwrap();
            running.remove(&task_id)
        };
        
        if let Some(mut task) = task {
            task.fail();
            
            // Move to completed tasks (even failed ones)
            {
                let mut completed = self.completed_tasks.write().unwrap();
                completed.push(task);
            }
            
            // Update statistics
            {
                let mut stats = self.stats.write().unwrap();
                stats.running_tasks = stats.running_tasks.saturating_sub(1);
                stats.failed_tasks += 1;
            }
            
            #[cfg(feature = "tracing")]
            warn!(task_id, reason, "Task failed");
            
            Ok(())
        } else {
            Err(AsyncError::scheduling_with_task(task_id, "Task not found"))
        }
    }
    
    /// Cancel a pending task
    pub async fn cancel_task(&self, task_id: u64) -> AsyncResult<()> {
        // Try to remove from queue first
        {
            let mut queue = self.queue.write().unwrap();
            let tasks: Vec<_> = queue.drain().collect();
            
            let mut found = false;
            for mut task in tasks {
                if task.id == task_id {
                    task.cancel();
                    found = true;
                    
                    // Move to completed tasks
                    let mut completed = self.completed_tasks.write().unwrap();
                    completed.push(task);
                } else {
                    queue.push(task);
                }
            }
            
            if found {
                let mut stats = self.stats.write().unwrap();
                stats.pending_tasks = stats.pending_tasks.saturating_sub(1);
                stats.cancelled_tasks += 1;
                
                #[cfg(feature = "tracing")]
                debug!(task_id, "Pending task cancelled");
                
                return Ok(());
            }
        }
        
        // Try to cancel running task
        let task = {
            let mut running = self.running_tasks.write().unwrap();
            running.remove(&task_id)
        };
        
        if let Some(mut task) = task {
            task.cancel();
            
            let mut completed = self.completed_tasks.write().unwrap();
            completed.push(task);
            
            let mut stats = self.stats.write().unwrap();
            stats.running_tasks = stats.running_tasks.saturating_sub(1);
            stats.cancelled_tasks += 1;
            
            #[cfg(feature = "tracing")]
            debug!(task_id, "Running task cancelled");
            
            Ok(())
        } else {
            Err(AsyncError::scheduling_with_task(task_id, "Task not found"))
        }
    }
    
    /// Get scheduler statistics
    pub fn get_stats(&self) -> SchedulerStats {
        self.stats.read().unwrap().clone()
    }
    
    /// Get pending task count
    pub fn pending_count(&self) -> usize {
        self.queue.read().unwrap().len()
    }
    
    /// Get running task count
    pub fn running_count(&self) -> usize {
        self.running_tasks.read().unwrap().len()
    }
    
    /// Clear completed task history
    pub fn clear_completed(&self) {
        let mut completed = self.completed_tasks.write().unwrap();
        completed.clear();
    }
    
    /// Check for missed deadlines
    pub async fn check_deadlines(&self, current_time: Time) -> Vec<u64> {
        let mut missed_tasks = Vec::new();
        
        // Check running tasks
        {
            let running = self.running_tasks.read().unwrap();
            for task in running.values() {
                if task.is_deadline_missed(current_time) {
                    missed_tasks.push(task.id);
                }
            }
        }
        
        // Check pending tasks
        {
            let queue = self.queue.read().unwrap();
            for task in queue.iter() {
                if task.is_deadline_missed(current_time) {
                    missed_tasks.push(task.id);
                }
            }
        }
        
        if !missed_tasks.is_empty() {
            #[cfg(feature = "tracing")]
            warn!(missed_count = missed_tasks.len(), "Deadline violations detected");
        }
        
        missed_tasks
    }
}

impl Default for TaskScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Scheduler statistics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SchedulerStats {
    /// Total tasks submitted
    pub tasks_submitted: u64,
    /// Currently pending tasks
    pub pending_tasks: u64,
    /// Currently running tasks
    pub running_tasks: u64,
    /// Completed tasks
    pub completed_tasks: u64,
    /// Failed tasks
    pub failed_tasks: u64,
    /// Cancelled tasks
    pub cancelled_tasks: u64,
    /// Real-time tasks submitted
    pub realtime_tasks: u64,
}

impl SchedulerStats {
    /// Get total tasks processed
    pub fn total_processed(&self) -> u64 {
        self.completed_tasks + self.failed_tasks + self.cancelled_tasks
    }
    
    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.total_processed();
        if total == 0 {
            0.0
        } else {
            self.completed_tasks as f64 / total as f64
        }
    }
}

impl std::fmt::Display for SchedulerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Scheduler: {} pending, {} running, {} completed ({:.1}% success)",
            self.pending_tasks,
            self.running_tasks,
            self.completed_tasks,
            self.success_rate() * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_task_scheduler() {
        let scheduler = TaskScheduler::new();
        
        let task_id = scheduler.submit_task(
            "test_task".to_string(),
            TaskPriority::Normal,
            Duration::from_millis(100),
        ).await.unwrap();
        
        assert_eq!(task_id, 1);
        assert_eq!(scheduler.pending_count(), 1);
        
        let task = scheduler.get_next_task().await.unwrap();
        assert_eq!(task.id, task_id);
        assert_eq!(scheduler.running_count(), 1);
        
        scheduler.complete_task(task_id).await.unwrap();
        assert_eq!(scheduler.running_count(), 0);
        
        let stats = scheduler.get_stats();
        assert_eq!(stats.completed_tasks, 1);
    }
    
    #[tokio::test]
    async fn test_task_priority() {
        let scheduler = TaskScheduler::new();
        
        // Submit tasks in reverse priority order
        let low_id = scheduler.submit_task(
            "low".to_string(),
            TaskPriority::Low,
            Duration::from_millis(100),
        ).await.unwrap();
        
        let high_id = scheduler.submit_task(
            "high".to_string(),
            TaskPriority::High,
            Duration::from_millis(100),
        ).await.unwrap();
        
        // High priority task should come first
        let task = scheduler.get_next_task().await.unwrap();
        assert_eq!(task.id, high_id);
        assert_eq!(task.priority, TaskPriority::High);
    }
    
    #[tokio::test]
    async fn test_task_cancellation() {
        let scheduler = TaskScheduler::new();
        
        let task_id = scheduler.submit_task(
            "test_task".to_string(),
            TaskPriority::Normal,
            Duration::from_millis(100),
        ).await.unwrap();
        
        assert!(scheduler.cancel_task(task_id).await.is_ok());
        assert_eq!(scheduler.pending_count(), 0);
        
        let stats = scheduler.get_stats();
        assert_eq!(stats.cancelled_tasks, 1);
    }
    
    #[test]
    fn test_task_ordering() {
        let task1 = ScheduledTask::new(1, "task1".to_string(), TaskPriority::Low, Duration::from_millis(100));
        let task2 = ScheduledTask::new(2, "task2".to_string(), TaskPriority::High, Duration::from_millis(100));
        
        // Higher priority should be "less than" for max-heap behavior
        assert!(task2 < task1);
    }
}