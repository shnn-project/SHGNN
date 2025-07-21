//! Error types for SHNN-Async operations
//!
//! This module provides comprehensive error handling for asynchronous
//! neuromorphic computing operations.

use shnn_core::error::SHNNError;
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Result type alias for async operations
pub type AsyncResult<T> = Result<T, AsyncError>;

/// Main error type for async operations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AsyncError {
    /// Core SHNN error
    Core(SHNNError),
    
    /// Runtime-related errors
    Runtime {
        /// Runtime type (tokio, async-std, etc.)
        runtime: String,
        /// Error description
        reason: String,
    },
    
    /// Network management errors
    Network {
        /// Network ID if applicable
        network_id: Option<u64>,
        /// Error description
        reason: String,
    },
    
    /// Channel communication errors
    Channel {
        /// Channel name or type
        channel: String,
        /// Error description
        reason: String,
    },
    
    /// Streaming errors
    Streaming {
        /// Stream ID if applicable
        stream_id: Option<u64>,
        /// Error description
        reason: String,
    },
    
    /// Distributed processing errors
    Distributed {
        /// Node ID if applicable
        node_id: Option<String>,
        /// Error description
        reason: String,
    },
    
    /// Load balancing errors
    LoadBalancing {
        /// Balancer configuration
        strategy: String,
        /// Error description
        reason: String,
    },
    
    /// Scheduling errors
    Scheduling {
        /// Task ID if applicable
        task_id: Option<u64>,
        /// Error description
        reason: String,
    },
    
    /// Monitoring errors
    Monitoring {
        /// Monitor type
        monitor_type: String,
        /// Error description
        reason: String,
    },
    
    /// Timeout errors
    Timeout {
        /// Operation that timed out
        operation: String,
        /// Timeout duration in milliseconds
        timeout_ms: u64,
    },
    
    /// Resource exhaustion errors
    ResourceExhausted {
        /// Resource type
        resource: String,
        /// Current usage
        current: u64,
        /// Maximum allowed
        limit: u64,
    },
    
    /// Configuration errors
    Configuration {
        /// Configuration field
        field: String,
        /// Error description
        reason: String,
    },
    
    /// Serialization/deserialization errors
    #[cfg(feature = "serde")]
    Serialization {
        /// Data type being serialized
        data_type: String,
        /// Error description
        reason: String,
    },
    
    /// I/O errors
    Io {
        /// Operation that failed
        operation: String,
        /// Error description
        reason: String,
    },
    
    /// Concurrency errors
    Concurrency {
        /// Type of concurrency issue
        issue_type: ConcurrencyIssue,
        /// Error description
        reason: String,
    },
    
    /// Generic async error
    Generic {
        /// Error message
        message: String,
    },
}

/// Types of concurrency issues
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ConcurrencyIssue {
    /// Deadlock detected
    Deadlock,
    /// Race condition
    RaceCondition,
    /// Lock contention
    LockContention,
    /// Channel closed unexpectedly
    ChannelClosed,
    /// Task cancellation
    TaskCanceled,
    /// Backpressure
    Backpressure,
}

impl fmt::Display for AsyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(err) => write!(f, "Core error: {}", err),
            Self::Runtime { runtime, reason } => {
                write!(f, "Runtime error ({}): {}", runtime, reason)
            }
            Self::Network { network_id, reason } => {
                if let Some(id) = network_id {
                    write!(f, "Network error (ID {}): {}", id, reason)
                } else {
                    write!(f, "Network error: {}", reason)
                }
            }
            Self::Channel { channel, reason } => {
                write!(f, "Channel error ({}): {}", channel, reason)
            }
            Self::Streaming { stream_id, reason } => {
                if let Some(id) = stream_id {
                    write!(f, "Streaming error (ID {}): {}", id, reason)
                } else {
                    write!(f, "Streaming error: {}", reason)
                }
            }
            Self::Distributed { node_id, reason } => {
                if let Some(id) = node_id {
                    write!(f, "Distributed error (node {}): {}", id, reason)
                } else {
                    write!(f, "Distributed error: {}", reason)
                }
            }
            Self::LoadBalancing { strategy, reason } => {
                write!(f, "Load balancing error ({}): {}", strategy, reason)
            }
            Self::Scheduling { task_id, reason } => {
                if let Some(id) = task_id {
                    write!(f, "Scheduling error (task {}): {}", id, reason)
                } else {
                    write!(f, "Scheduling error: {}", reason)
                }
            }
            Self::Monitoring { monitor_type, reason } => {
                write!(f, "Monitoring error ({}): {}", monitor_type, reason)
            }
            Self::Timeout { operation, timeout_ms } => {
                write!(f, "Timeout error: {} timed out after {}ms", operation, timeout_ms)
            }
            Self::ResourceExhausted { resource, current, limit } => {
                write!(f, "Resource exhausted: {} usage {}/{}", resource, current, limit)
            }
            Self::Configuration { field, reason } => {
                write!(f, "Configuration error in '{}': {}", field, reason)
            }
            #[cfg(feature = "serde")]
            Self::Serialization { data_type, reason } => {
                write!(f, "Serialization error for {}: {}", data_type, reason)
            }
            Self::Io { operation, reason } => {
                write!(f, "I/O error during {}: {}", operation, reason)
            }
            Self::Concurrency { issue_type, reason } => {
                write!(f, "Concurrency error ({:?}): {}", issue_type, reason)
            }
            Self::Generic { message } => {
                write!(f, "Async error: {}", message)
            }
        }
    }
}

impl std::error::Error for AsyncError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Core(err) => Some(err),
            _ => None,
        }
    }
}

// Conversions from other error types
impl From<SHNNError> for AsyncError {
    fn from(err: SHNNError) -> Self {
        Self::Core(err)
    }
}

impl From<std::io::Error> for AsyncError {
    fn from(err: std::io::Error) -> Self {
        Self::Io {
            operation: "unknown".to_string(),
            reason: err.to_string(),
        }
    }
}

#[cfg(feature = "tokio-runtime")]
impl From<tokio::time::error::Elapsed> for AsyncError {
    fn from(_: tokio::time::error::Elapsed) -> Self {
        Self::Timeout {
            operation: "tokio operation".to_string(),
            timeout_ms: 0, // Unknown timeout duration
        }
    }
}

#[cfg(feature = "serde")]
impl From<bincode::Error> for AsyncError {
    fn from(err: bincode::Error) -> Self {
        Self::Serialization {
            data_type: "bincode".to_string(),
            reason: err.to_string(),
        }
    }
}

#[cfg(feature = "serde")]
impl From<serde_json::Error> for AsyncError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization {
            data_type: "json".to_string(),
            reason: err.to_string(),
        }
    }
}

// Convenience constructors
impl AsyncError {
    /// Create a runtime error
    pub fn runtime(runtime: &str, reason: &str) -> Self {
        Self::Runtime {
            runtime: runtime.to_string(),
            reason: reason.to_string(),
        }
    }
    
    /// Create a network error
    pub fn network(reason: &str) -> Self {
        Self::Network {
            network_id: None,
            reason: reason.to_string(),
        }
    }
    
    /// Create a network error with ID
    pub fn network_with_id(network_id: u64, reason: &str) -> Self {
        Self::Network {
            network_id: Some(network_id),
            reason: reason.to_string(),
        }
    }
    
    /// Create a channel error
    pub fn channel(channel: &str, reason: &str) -> Self {
        Self::Channel {
            channel: channel.to_string(),
            reason: reason.to_string(),
        }
    }
    
    /// Create a streaming error
    pub fn streaming(reason: &str) -> Self {
        Self::Streaming {
            stream_id: None,
            reason: reason.to_string(),
        }
    }
    
    /// Create a streaming error with ID
    pub fn streaming_with_id(stream_id: u64, reason: &str) -> Self {
        Self::Streaming {
            stream_id: Some(stream_id),
            reason: reason.to_string(),
        }
    }
    
    /// Create a distributed error
    pub fn distributed(reason: &str) -> Self {
        Self::Distributed {
            node_id: None,
            reason: reason.to_string(),
        }
    }
    
    /// Create a distributed error with node ID
    pub fn distributed_with_node(node_id: &str, reason: &str) -> Self {
        Self::Distributed {
            node_id: Some(node_id.to_string()),
            reason: reason.to_string(),
        }
    }
    
    /// Create a load balancing error
    pub fn load_balancing(strategy: &str, reason: &str) -> Self {
        Self::LoadBalancing {
            strategy: strategy.to_string(),
            reason: reason.to_string(),
        }
    }
    
    /// Create a scheduling error
    pub fn scheduling(reason: &str) -> Self {
        Self::Scheduling {
            task_id: None,
            reason: reason.to_string(),
        }
    }
    
    /// Create a scheduling error with task ID
    pub fn scheduling_with_task(task_id: u64, reason: &str) -> Self {
        Self::Scheduling {
            task_id: Some(task_id),
            reason: reason.to_string(),
        }
    }
    
    /// Create a monitoring error
    pub fn monitoring(monitor_type: &str, reason: &str) -> Self {
        Self::Monitoring {
            monitor_type: monitor_type.to_string(),
            reason: reason.to_string(),
        }
    }
    
    /// Create a timeout error
    pub fn timeout(operation: &str, timeout_ms: u64) -> Self {
        Self::Timeout {
            operation: operation.to_string(),
            timeout_ms,
        }
    }
    
    /// Create a resource exhausted error
    pub fn resource_exhausted(resource: &str, current: u64, limit: u64) -> Self {
        Self::ResourceExhausted {
            resource: resource.to_string(),
            current,
            limit,
        }
    }
    
    /// Create a configuration error
    pub fn configuration(field: &str, reason: &str) -> Self {
        Self::Configuration {
            field: field.to_string(),
            reason: reason.to_string(),
        }
    }
    
    /// Create an I/O error
    pub fn io(operation: &str, reason: &str) -> Self {
        Self::Io {
            operation: operation.to_string(),
            reason: reason.to_string(),
        }
    }
    
    /// Create a concurrency error
    pub fn concurrency(issue_type: ConcurrencyIssue, reason: &str) -> Self {
        Self::Concurrency {
            issue_type,
            reason: reason.to_string(),
        }
    }
    
    /// Create a generic error
    pub fn generic(message: &str) -> Self {
        Self::Generic {
            message: message.to_string(),
        }
    }
}

/// Error context trait for adding context to errors
pub trait ErrorContext<T> {
    /// Add context to an error
    fn with_context<F>(self, f: F) -> AsyncResult<T>
    where
        F: FnOnce() -> String;
    
    /// Add static context to an error
    fn context(self, msg: &'static str) -> AsyncResult<T>;
}

impl<T> ErrorContext<T> for AsyncResult<T> {
    fn with_context<F>(self, f: F) -> AsyncResult<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|err| {
            let context = f();
            match err {
                AsyncError::Generic { message } => AsyncError::Generic {
                    message: format!("{}: {}", context, message),
                },
                other => AsyncError::Generic {
                    message: format!("{}: {}", context, other),
                },
            }
        })
    }
    
    fn context(self, msg: &'static str) -> AsyncResult<T> {
        self.with_context(|| msg.to_string())
    }
}

impl<T, E> ErrorContext<T> for Result<T, E>
where
    E: Into<AsyncError>,
{
    fn with_context<F>(self, f: F) -> AsyncResult<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|err| {
            let context = f();
            let async_err = err.into();
            AsyncError::Generic {
                message: format!("{}: {}", context, async_err),
            }
        })
    }
    
    fn context(self, msg: &'static str) -> AsyncResult<T> {
        self.with_context(|| msg.to_string())
    }
}

/// Macro for creating async errors with formatted messages
#[macro_export]
macro_rules! async_error {
    ($kind:ident, $($arg:expr),+ $(,)?) => {
        $crate::error::AsyncError::$kind(format!($($arg),+))
    };
}

/// Macro for creating async results
#[macro_export]
macro_rules! async_result {
    ($expr:expr) => {
        Ok($expr)
    };
    (Err $kind:ident, $($arg:expr),+ $(,)?) => {
        Err($crate::error::AsyncError::$kind($($arg),+))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let err = AsyncError::network("connection failed");
        assert!(format!("{}", err).contains("Network error"));
        assert!(format!("{}", err).contains("connection failed"));
    }
    
    #[test]
    fn test_error_with_id() {
        let err = AsyncError::network_with_id(42, "timeout");
        assert!(format!("{}", err).contains("ID 42"));
        assert!(format!("{}", err).contains("timeout"));
    }
    
    #[test]
    fn test_timeout_error() {
        let err = AsyncError::timeout("spike processing", 5000);
        let msg = format!("{}", err);
        assert!(msg.contains("spike processing"));
        assert!(msg.contains("5000ms"));
    }
    
    #[test]
    fn test_resource_exhausted() {
        let err = AsyncError::resource_exhausted("memory", 1024, 512);
        let msg = format!("{}", err);
        assert!(msg.contains("memory"));
        assert!(msg.contains("1024/512"));
    }
    
    #[test]
    fn test_concurrency_error() {
        let err = AsyncError::concurrency(ConcurrencyIssue::Deadlock, "mutex contention");
        let msg = format!("{}", err);
        assert!(msg.contains("Deadlock"));
        assert!(msg.contains("mutex contention"));
    }
    
    #[test]
    fn test_error_context() {
        let result: AsyncResult<()> = Err(AsyncError::generic("base error"));
        let with_context = result.context("operation failed");
        
        assert!(with_context.is_err());
        let msg = format!("{}", with_context.unwrap_err());
        assert!(msg.contains("operation failed"));
        assert!(msg.contains("base error"));
    }
    
    #[test]
    fn test_from_core_error() {
        let core_err = SHNNError::generic("core error");
        let async_err: AsyncError = core_err.into();
        
        match async_err {
            AsyncError::Core(_) => (),
            _ => panic!("Expected Core error variant"),
        }
    }
    
    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let async_err: AsyncError = io_err.into();
        
        match async_err {
            AsyncError::Io { .. } => (),
            _ => panic!("Expected Io error variant"),
        }
    }
}