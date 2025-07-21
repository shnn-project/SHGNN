//! Async runtime management for SHNN
//!
//! This module provides unified runtime management across different async
//! implementations (Tokio, async-std, etc.).

use crate::error::{AsyncError, AsyncResult};
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Async runtime configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RuntimeConfig {
    /// Number of worker threads
    pub worker_threads: Option<usize>,
    /// Enable I/O threads
    pub enable_io: bool,
    /// Enable time drivers
    pub enable_time: bool,
    /// Thread stack size
    pub thread_stack_size: Option<usize>,
    /// Thread name prefix
    pub thread_name: Option<String>,
    /// Maximum blocking threads
    pub max_blocking_threads: Option<usize>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            worker_threads: None, // Use system default
            enable_io: true,
            enable_time: true,
            thread_stack_size: None,
            thread_name: Some("shnn-async".to_string()),
            max_blocking_threads: None,
        }
    }
}

/// Unified async runtime trait
pub trait AsyncRuntime: Send + Sync + 'static {
    /// Spawn a future on the runtime
    fn spawn<F>(&self, future: F) -> AsyncResult<TaskHandle>
    where
        F: futures::Future<Output = ()> + Send + 'static;
    
    /// Spawn a blocking task
    fn spawn_blocking<F, R>(&self, f: F) -> AsyncResult<TaskHandle>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static;
    
    /// Sleep for a duration
    fn sleep(&self, duration: Duration) -> BoxFuture<'_, ()>;
    
    /// Get current time
    fn now(&self) -> std::time::Instant;
    
    /// Shutdown the runtime
    fn shutdown(&self) -> BoxFuture<'_, AsyncResult<()>>;
    
    /// Get runtime statistics
    fn stats(&self) -> RuntimeStats;
}

/// Type alias for boxed futures
pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn futures::Future<Output = T> + Send + 'a>>;

/// Task handle for spawned tasks
#[derive(Debug)]
pub struct TaskHandle {
    /// Task ID
    pub id: u64,
    /// Task name
    pub name: Option<String>,
    /// Whether task is completed
    completed: Arc<std::sync::atomic::AtomicBool>,
}

impl TaskHandle {
    /// Create a new task handle
    pub fn new(id: u64) -> Self {
        Self {
            id,
            name: None,
            completed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }
    
    /// Create with name
    pub fn with_name(id: u64, name: String) -> Self {
        Self {
            id,
            name: Some(name),
            completed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }
    
    /// Check if task is completed
    pub fn is_completed(&self) -> bool {
        self.completed.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// Mark task as completed
    pub(crate) fn mark_completed(&self) {
        self.completed.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Abort the task (if supported by runtime)
    pub fn abort(&self) -> AsyncResult<()> {
        // Implementation would depend on the specific runtime
        Ok(())
    }
}

/// Runtime statistics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RuntimeStats {
    /// Number of active tasks
    pub active_tasks: u64,
    /// Number of completed tasks
    pub completed_tasks: u64,
    /// Number of worker threads
    pub worker_threads: usize,
    /// Number of blocking threads
    pub blocking_threads: usize,
    /// Runtime uptime in seconds
    pub uptime_secs: u64,
}

impl RuntimeStats {
    /// Get total tasks
    pub fn total_tasks(&self) -> u64 {
        self.active_tasks + self.completed_tasks
    }
    
    /// Get task completion rate
    pub fn completion_rate(&self) -> f64 {
        let total = self.total_tasks();
        if total == 0 {
            0.0
        } else {
            self.completed_tasks as f64 / total as f64
        }
    }
}

/// Tokio runtime implementation
#[cfg(feature = "tokio-runtime")]
pub struct TokioRuntime {
    handle: tokio::runtime::Handle,
    start_time: std::time::Instant,
    task_counter: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "tokio-runtime")]
impl TokioRuntime {
    /// Create a new Tokio runtime
    pub fn new(config: RuntimeConfig) -> AsyncResult<Self> {
        let mut builder = tokio::runtime::Builder::new_multi_thread();
        
        if let Some(threads) = config.worker_threads {
            builder.worker_threads(threads);
        }
        
        if let Some(stack_size) = config.thread_stack_size {
            builder.thread_stack_size(stack_size);
        }
        
        if let Some(name) = config.thread_name {
            builder.thread_name(name);
        }
        
        if let Some(max_blocking) = config.max_blocking_threads {
            builder.max_blocking_threads(max_blocking);
        }
        
        builder.enable_all();
        
        let runtime = builder.build()
            .map_err(|e| AsyncError::runtime("tokio", &e.to_string()))?;
        
        let handle = runtime.handle().clone();
        
        // Spawn the runtime in a background thread
        std::thread::spawn(move || {
            runtime.block_on(async {
                // Keep runtime alive
                loop {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            })
        });
        
        Ok(Self {
            handle,
            start_time: std::time::Instant::now(),
            task_counter: std::sync::atomic::AtomicU64::new(0),
        })
    }
    
    /// Get the Tokio handle
    pub fn handle(&self) -> &tokio::runtime::Handle {
        &self.handle
    }
}

#[cfg(feature = "tokio-runtime")]
impl AsyncRuntime for TokioRuntime {
    fn spawn<F>(&self, future: F) -> AsyncResult<TaskHandle>
    where
        F: futures::Future<Output = ()> + Send + 'static,
    {
        let task_id = self.task_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let handle = TaskHandle::new(task_id);
        let completed = handle.completed.clone();
        
        self.handle.spawn(async move {
            future.await;
            completed.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        
        Ok(handle)
    }
    
    fn spawn_blocking<F, R>(&self, f: F) -> AsyncResult<TaskHandle>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let task_id = self.task_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let handle = TaskHandle::new(task_id);
        let completed = handle.completed.clone();
        
        self.handle.spawn_blocking(move || {
            let _result = f();
            completed.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        
        Ok(handle)
    }
    
    fn sleep(&self, duration: Duration) -> BoxFuture<'_, ()> {
        Box::pin(tokio::time::sleep(duration))
    }
    
    fn now(&self) -> std::time::Instant {
        tokio::time::Instant::now().into_std()
    }
    
    fn shutdown(&self) -> BoxFuture<'_, AsyncResult<()>> {
        Box::pin(async {
            // Tokio runtime shutdown is handled when the handle is dropped
            Ok(())
        })
    }
    
    fn stats(&self) -> RuntimeStats {
        RuntimeStats {
            active_tasks: 0, // Would need access to internal tokio stats
            completed_tasks: self.task_counter.load(std::sync::atomic::Ordering::Relaxed),
            worker_threads: 0, // Would need runtime introspection
            blocking_threads: 0,
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }
}

/// async-std runtime implementation
#[cfg(feature = "async-std-runtime")]
pub struct AsyncStdRuntime {
    start_time: std::time::Instant,
    task_counter: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "async-std-runtime")]
impl AsyncStdRuntime {
    /// Create a new async-std runtime
    pub fn new(_config: RuntimeConfig) -> AsyncResult<Self> {
        Ok(Self {
            start_time: std::time::Instant::now(),
            task_counter: std::sync::atomic::AtomicU64::new(0),
        })
    }
}

#[cfg(feature = "async-std-runtime")]
impl AsyncRuntime for AsyncStdRuntime {
    fn spawn<F>(&self, future: F) -> AsyncResult<TaskHandle>
    where
        F: futures::Future<Output = ()> + Send + 'static,
    {
        let task_id = self.task_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let handle = TaskHandle::new(task_id);
        let completed = handle.completed.clone();
        
        async_std::task::spawn(async move {
            future.await;
            completed.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        
        Ok(handle)
    }
    
    fn spawn_blocking<F, R>(&self, f: F) -> AsyncResult<TaskHandle>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let task_id = self.task_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let handle = TaskHandle::new(task_id);
        let completed = handle.completed.clone();
        
        async_std::task::spawn_blocking(move || {
            let _result = f();
            completed.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        
        Ok(handle)
    }
    
    fn sleep(&self, duration: Duration) -> BoxFuture<'_, ()> {
        Box::pin(async_std::task::sleep(duration))
    }
    
    fn now(&self) -> std::time::Instant {
        std::time::Instant::now()
    }
    
    fn shutdown(&self) -> BoxFuture<'_, AsyncResult<()>> {
        Box::pin(async {
            // async-std doesn't need explicit shutdown
            Ok(())
        })
    }
    
    fn stats(&self) -> RuntimeStats {
        RuntimeStats {
            active_tasks: 0,
            completed_tasks: self.task_counter.load(std::sync::atomic::Ordering::Relaxed),
            worker_threads: 0,
            blocking_threads: 0,
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }
}

/// Runtime manager that selects appropriate runtime based on features
pub struct RuntimeManager {
    runtime: Box<dyn AsyncRuntime>,
    config: RuntimeConfig,
}

impl RuntimeManager {
    /// Create a new runtime manager with default runtime
    pub fn new() -> AsyncResult<Self> {
        Self::with_config(RuntimeConfig::default())
    }
    
    /// Create with specific configuration
    pub fn with_config(config: RuntimeConfig) -> AsyncResult<Self> {
        let runtime = Self::create_runtime(config.clone())?;
        Ok(Self { runtime, config })
    }
    
    /// Create runtime based on available features
    fn create_runtime(config: RuntimeConfig) -> AsyncResult<Box<dyn AsyncRuntime>> {
        #[cfg(feature = "tokio-runtime")]
        {
            let runtime = TokioRuntime::new(config)?;
            return Ok(Box::new(runtime));
        }
        
        #[cfg(all(feature = "async-std-runtime", not(feature = "tokio-runtime")))]
        {
            let runtime = AsyncStdRuntime::new(config)?;
            return Ok(Box::new(runtime));
        }
        
        #[cfg(not(any(feature = "tokio-runtime", feature = "async-std-runtime")))]
        {
            Err(AsyncError::runtime("none", "No async runtime feature enabled"))
        }
    }
    
    /// Get runtime reference
    pub fn runtime(&self) -> &dyn AsyncRuntime {
        self.runtime.as_ref()
    }
    
    /// Get configuration
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }
    
    /// Spawn a future
    pub fn spawn<F>(&self, future: F) -> AsyncResult<TaskHandle>
    where
        F: futures::Future<Output = ()> + Send + 'static,
    {
        self.runtime.spawn(future)
    }
    
    /// Spawn a blocking task
    pub fn spawn_blocking<F, R>(&self, f: F) -> AsyncResult<TaskHandle>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.runtime.spawn_blocking(f)
    }
    
    /// Sleep for a duration
    pub async fn sleep(&self, duration: Duration) {
        self.runtime.sleep(duration).await
    }
    
    /// Get current time
    pub fn now(&self) -> std::time::Instant {
        self.runtime.now()
    }
    
    /// Get runtime statistics
    pub fn stats(&self) -> RuntimeStats {
        self.runtime.stats()
    }
    
    /// Shutdown the runtime
    pub async fn shutdown(self) -> AsyncResult<()> {
        self.runtime.shutdown().await
    }
}

impl Default for RuntimeManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default runtime")
    }
}

/// Global runtime instance (optional convenience)
static GLOBAL_RUNTIME: std::sync::OnceLock<RuntimeManager> = std::sync::OnceLock::new();

/// Initialize global runtime
pub fn init_global_runtime(config: RuntimeConfig) -> AsyncResult<()> {
    let manager = RuntimeManager::with_config(config)?;
    GLOBAL_RUNTIME.set(manager)
        .map_err(|_| AsyncError::runtime("global", "Runtime already initialized"))?;
    Ok(())
}

/// Get global runtime reference
pub fn global_runtime() -> AsyncResult<&'static RuntimeManager> {
    GLOBAL_RUNTIME.get()
        .ok_or_else(|| AsyncError::runtime("global", "Runtime not initialized"))
}

/// Spawn on global runtime
pub fn spawn<F>(future: F) -> AsyncResult<TaskHandle>
where
    F: futures::Future<Output = ()> + Send + 'static,
{
    global_runtime()?.spawn(future)
}

/// Spawn blocking on global runtime
pub fn spawn_blocking<F, R>(f: F) -> AsyncResult<TaskHandle>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    global_runtime()?.spawn_blocking(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_runtime_config_default() {
        let config = RuntimeConfig::default();
        assert!(config.enable_io);
        assert!(config.enable_time);
        assert_eq!(config.thread_name, Some("shnn-async".to_string()));
    }
    
    #[test]
    fn test_task_handle() {
        let handle = TaskHandle::new(42);
        assert_eq!(handle.id, 42);
        assert!(!handle.is_completed());
        
        handle.mark_completed();
        assert!(handle.is_completed());
    }
    
    #[test]
    fn test_runtime_stats() {
        let mut stats = RuntimeStats::default();
        stats.active_tasks = 5;
        stats.completed_tasks = 10;
        
        assert_eq!(stats.total_tasks(), 15);
        assert_eq!(stats.completion_rate(), 10.0 / 15.0);
    }
    
    #[cfg(feature = "tokio-runtime")]
    #[tokio::test]
    async fn test_tokio_runtime() {
        let runtime = TokioRuntime::new(RuntimeConfig::default()).unwrap();
        
        let handle = runtime.spawn(async {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }).unwrap();
        
        // Wait a bit for task to complete
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(handle.is_completed());
    }
    
    #[test]
    fn test_runtime_manager_creation() {
        let config = RuntimeConfig::default();
        let manager = RuntimeManager::with_config(config);
        assert!(manager.is_ok());
    }
}