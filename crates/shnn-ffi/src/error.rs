//! Error handling for SHNN FFI operations
//!
//! This module defines error types specific to FFI operations and hardware
//! acceleration, providing clear error reporting for different failure modes.

use thiserror::Error;
use crate::{AcceleratorId, NetworkId, types::AcceleratorType};

/// Result type for FFI operations
pub type FFIResult<T> = Result<T, FFIError>;

/// Comprehensive error types for FFI operations
#[derive(Error, Debug)]
pub enum FFIError {
    /// Hardware accelerator is not available or not supported
    #[error("Unsupported hardware accelerator: {0:?}")]
    UnsupportedHardware(AcceleratorType),
    
    /// Invalid accelerator ID provided
    #[error("Invalid accelerator ID: {0:?}")]
    InvalidAcceleratorId(AcceleratorId),
    
    /// Invalid network ID provided
    #[error("Invalid network ID: {0:?}")]
    InvalidNetworkId(NetworkId),
    
    /// Hardware initialization failed
    #[error("Hardware initialization failed: {0}")]
    HardwareInitializationFailed(String),
    
    /// Memory allocation failed on hardware
    #[error("Hardware memory allocation failed: {0}")]
    MemoryAllocationFailed(String),
    
    /// Network deployment failed
    #[error("Network deployment failed: {0}")]
    NetworkDeploymentFailed(String),
    
    /// Spike processing failed
    #[error("Spike processing failed: {0}")]
    SpikeProcessingFailed(String),
    
    /// Invalid configuration provided
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    
    /// Hardware driver error
    #[error("Hardware driver error: {0}")]
    DriverError(String),
    
    /// Communication timeout with hardware
    #[error("Hardware communication timeout after {timeout_ms}ms")]
    CommunicationTimeout { timeout_ms: u64 },
    
    /// Resource limit exceeded
    #[error("Resource limit exceeded: {resource} (limit: {limit}, requested: {requested})")]
    ResourceLimitExceeded {
        resource: String,
        limit: u64,
        requested: u64,
    },
    
    /// Synchronization error between host and device
    #[error("Synchronization error: {0}")]
    SynchronizationError(String),
    
    /// Version mismatch between components
    #[error("Version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: String, found: String },
    
    /// CUDA-specific errors
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    CudaError(#[from] CudaFFIError),
    
    /// OpenCL-specific errors
    #[cfg(feature = "opencl")]
    #[error("OpenCL error: {0}")]
    OpenCLError(#[from] OpenCLFFIError),
    
    /// FPGA-specific errors
    #[cfg(feature = "fpga")]
    #[error("FPGA error: {0}")]
    FPGAError(#[from] FPGAFFIError),
    
    /// RRAM-specific errors
    #[cfg(feature = "rram")]
    #[error("RRAM error: {0}")]
    RRAMError(#[from] RRAMFFIError),
    
    /// Intel Loihi-specific errors
    #[cfg(feature = "intel-loihi")]
    #[error("Intel Loihi error: {0}")]
    LoihiError(#[from] LoihiFFIError),
    
    /// SpiNNaker-specific errors
    #[cfg(feature = "spiNNaker")]
    #[error("SpiNNaker error: {0}")]
    SpiNNakerError(#[from] SpiNNakerFFIError),
    
    /// Core SHNN library error
    #[error("SHNN core error: {0}")]
    CoreError(#[from] shnn_core::error::SHNNError),
    
    /// IO error (file operations, network communication)
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Serialization/deserialization error
    #[cfg(feature = "distributed")]
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),
    
    /// Generic error for unexpected conditions
    #[error("Unexpected error: {0}")]
    Unexpected(String),
}

impl FFIError {
    /// Create a hardware initialization error
    pub fn hardware_init(message: impl Into<String>) -> Self {
        Self::HardwareInitializationFailed(message.into())
    }
    
    /// Create a memory allocation error
    pub fn memory_allocation(message: impl Into<String>) -> Self {
        Self::MemoryAllocationFailed(message.into())
    }
    
    /// Create a network deployment error
    pub fn network_deployment(message: impl Into<String>) -> Self {
        Self::NetworkDeploymentFailed(message.into())
    }
    
    /// Create a spike processing error
    pub fn spike_processing(message: impl Into<String>) -> Self {
        Self::SpikeProcessingFailed(message.into())
    }
    
    /// Create an invalid configuration error
    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfiguration(message.into())
    }
    
    /// Create a driver error
    pub fn driver_error(message: impl Into<String>) -> Self {
        Self::DriverError(message.into())
    }
    
    /// Create a communication timeout error
    pub fn communication_timeout(timeout_ms: u64) -> Self {
        Self::CommunicationTimeout { timeout_ms }
    }
    
    /// Create a resource limit exceeded error
    pub fn resource_limit_exceeded(
        resource: impl Into<String>,
        limit: u64,
        requested: u64,
    ) -> Self {
        Self::ResourceLimitExceeded {
            resource: resource.into(),
            limit,
            requested,
        }
    }
    
    /// Create a synchronization error
    pub fn synchronization_error(message: impl Into<String>) -> Self {
        Self::SynchronizationError(message.into())
    }
    
    /// Create a version mismatch error
    pub fn version_mismatch(
        expected: impl Into<String>,
        found: impl Into<String>,
    ) -> Self {
        Self::VersionMismatch {
            expected: expected.into(),
            found: found.into(),
        }
    }
    
    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::CommunicationTimeout { .. } => true,
            Self::MemoryAllocationFailed(_) => true,
            Self::SynchronizationError(_) => true,
            Self::DriverError(_) => false,
            Self::UnsupportedHardware(_) => false,
            Self::InvalidAcceleratorId(_) => false,
            Self::InvalidNetworkId(_) => false,
            Self::HardwareInitializationFailed(_) => false,
            Self::VersionMismatch { .. } => false,
            _ => false,
        }
    }
    
    /// Get error category for logging and metrics
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::UnsupportedHardware(_) |
            Self::InvalidAcceleratorId(_) |
            Self::InvalidNetworkId(_) |
            Self::InvalidConfiguration(_) => ErrorCategory::Configuration,
            
            Self::HardwareInitializationFailed(_) |
            Self::DriverError(_) => ErrorCategory::Hardware,
            
            Self::MemoryAllocationFailed(_) |
            Self::ResourceLimitExceeded { .. } => ErrorCategory::Memory,
            
            Self::CommunicationTimeout { .. } |
            Self::SynchronizationError(_) => ErrorCategory::Communication,
            
            Self::NetworkDeploymentFailed(_) |
            Self::SpikeProcessingFailed(_) => ErrorCategory::Processing,
            
            Self::VersionMismatch { .. } => ErrorCategory::Compatibility,
            
            Self::IoError(_) => ErrorCategory::IO,
            
            #[cfg(feature = "distributed")]
            Self::SerializationError(_) => ErrorCategory::Serialization,
            
            _ => ErrorCategory::Unknown,
        }
    }
}

/// Error categories for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Configuration and validation errors
    Configuration,
    /// Hardware-related errors
    Hardware,
    /// Memory allocation and management errors
    Memory,
    /// Communication and networking errors
    Communication,
    /// Data processing errors
    Processing,
    /// Compatibility and version errors
    Compatibility,
    /// Input/output errors
    IO,
    /// Serialization errors
    Serialization,
    /// Unknown or uncategorized errors
    Unknown,
}

// Hardware-specific error types
#[cfg(feature = "cuda")]
#[derive(Error, Debug)]
pub enum CudaFFIError {
    #[error("CUDA device not found")]
    DeviceNotFound,
    #[error("CUDA out of memory")]
    OutOfMemory,
    #[error("CUDA kernel launch failed: {0}")]
    KernelLaunchFailed(String),
    #[error("CUDA memory copy failed: {0}")]
    MemoryCopyFailed(String),
    #[error("CUDA context error: {0}")]
    ContextError(String),
}

#[cfg(feature = "opencl")]
#[derive(Error, Debug)]
pub enum OpenCLFFIError {
    #[error("OpenCL platform not found")]
    PlatformNotFound,
    #[error("OpenCL device not found")]
    DeviceNotFound,
    #[error("OpenCL context creation failed")]
    ContextCreationFailed,
    #[error("OpenCL kernel compilation failed: {0}")]
    KernelCompilationFailed(String),
    #[error("OpenCL buffer allocation failed")]
    BufferAllocationFailed,
}

#[cfg(feature = "fpga")]
#[derive(Error, Debug)]
pub enum FPGAFFIError {
    #[error("FPGA device not found")]
    DeviceNotFound,
    #[error("FPGA bitstream loading failed: {0}")]
    BitstreamLoadFailed(String),
    #[error("FPGA configuration failed: {0}")]
    ConfigurationFailed(String),
    #[error("FPGA timing violation: {0}")]
    TimingViolation(String),
}

#[cfg(feature = "rram")]
#[derive(Error, Debug)]
pub enum RRAMFFIError {
    #[error("RRAM array not found")]
    ArrayNotFound,
    #[error("RRAM programming failed: {0}")]
    ProgrammingFailed(String),
    #[error("RRAM read error: {0}")]
    ReadError(String),
    #[error("RRAM endurance exceeded")]
    EnduranceExceeded,
}

#[cfg(feature = "intel-loihi")]
#[derive(Error, Debug)]
pub enum LoihiFFIError {
    #[error("Loihi chip not found")]
    ChipNotFound,
    #[error("Loihi board communication failed")]
    BoardCommunicationFailed,
    #[error("Loihi network mapping failed: {0}")]
    NetworkMappingFailed(String),
    #[error("Loihi execution timeout")]
    ExecutionTimeout,
}

#[cfg(feature = "spiNNaker")]
#[derive(Error, Debug)]
pub enum SpiNNakerFFIError {
    #[error("SpiNNaker machine not found")]
    MachineNotFound,
    #[error("SpiNNaker core allocation failed")]
    CoreAllocationFailed,
    #[error("SpiNNaker routing failed: {0}")]
    RoutingFailed(String),
    #[error("SpiNNaker packet loss detected")]
    PacketLoss,
}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry the operation with the same parameters
    Retry { max_attempts: u32, delay_ms: u64 },
    /// Retry with modified parameters
    RetryWithFallback { fallback_config: String },
    /// Switch to a different hardware accelerator
    SwitchAccelerator { target_type: AcceleratorType },
    /// Graceful degradation to software implementation
    FallbackToSoftware,
    /// No recovery possible, fail permanently
    NoRecovery,
}

impl FFIError {
    /// Get the recommended recovery strategy for this error
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            Self::CommunicationTimeout { .. } => RecoveryStrategy::Retry {
                max_attempts: 3,
                delay_ms: 1000,
            },
            
            Self::MemoryAllocationFailed(_) => RecoveryStrategy::RetryWithFallback {
                fallback_config: "reduced_batch_size".to_string(),
            },
            
            Self::UnsupportedHardware(_) => RecoveryStrategy::FallbackToSoftware,
            
            Self::ResourceLimitExceeded { .. } => RecoveryStrategy::RetryWithFallback {
                fallback_config: "reduced_network_size".to_string(),
            },
            
            Self::DriverError(_) => RecoveryStrategy::SwitchAccelerator {
                target_type: AcceleratorType::CPU, // Fallback to CPU
            },
            
            _ => RecoveryStrategy::NoRecovery,
        }
    }
}

/// Convert FFI errors to C-compatible error codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FFIErrorCode {
    Success = 0,
    UnsupportedHardware = 1001,
    InvalidAcceleratorId = 1002,
    InvalidNetworkId = 1003,
    HardwareInitializationFailed = 1004,
    MemoryAllocationFailed = 1005,
    NetworkDeploymentFailed = 1006,
    SpikeProcessingFailed = 1007,
    InvalidConfiguration = 1008,
    DriverError = 1009,
    CommunicationTimeout = 1010,
    ResourceLimitExceeded = 1011,
    SynchronizationError = 1012,
    VersionMismatch = 1013,
    UnknownError = 9999,
}

impl From<&FFIError> for FFIErrorCode {
    fn from(error: &FFIError) -> Self {
        match error {
            FFIError::UnsupportedHardware(_) => Self::UnsupportedHardware,
            FFIError::InvalidAcceleratorId(_) => Self::InvalidAcceleratorId,
            FFIError::InvalidNetworkId(_) => Self::InvalidNetworkId,
            FFIError::HardwareInitializationFailed(_) => Self::HardwareInitializationFailed,
            FFIError::MemoryAllocationFailed(_) => Self::MemoryAllocationFailed,
            FFIError::NetworkDeploymentFailed(_) => Self::NetworkDeploymentFailed,
            FFIError::SpikeProcessingFailed(_) => Self::SpikeProcessingFailed,
            FFIError::InvalidConfiguration(_) => Self::InvalidConfiguration,
            FFIError::DriverError(_) => Self::DriverError,
            FFIError::CommunicationTimeout { .. } => Self::CommunicationTimeout,
            FFIError::ResourceLimitExceeded { .. } => Self::ResourceLimitExceeded,
            FFIError::SynchronizationError(_) => Self::SynchronizationError,
            FFIError::VersionMismatch { .. } => Self::VersionMismatch,
            _ => Self::UnknownError,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let error = FFIError::hardware_init("Test message");
        assert!(matches!(error, FFIError::HardwareInitializationFailed(_)));
        assert!(!error.is_recoverable());
        assert_eq!(error.category(), ErrorCategory::Hardware);
    }
    
    #[test]
    fn test_error_recovery_strategy() {
        let timeout_error = FFIError::communication_timeout(5000);
        match timeout_error.recovery_strategy() {
            RecoveryStrategy::Retry { max_attempts, delay_ms } => {
                assert_eq!(max_attempts, 3);
                assert_eq!(delay_ms, 1000);
            }
            _ => panic!("Expected retry strategy"),
        }
    }
    
    #[test]
    fn test_error_code_conversion() {
        let error = FFIError::invalid_config("test");
        let error_code = FFIErrorCode::from(&error);
        assert_eq!(error_code, FFIErrorCode::InvalidConfiguration);
    }
    
    #[test]
    fn test_error_categories() {
        assert_eq!(
            FFIError::invalid_config("test").category(),
            ErrorCategory::Configuration
        );
        assert_eq!(
            FFIError::memory_allocation("test").category(),
            ErrorCategory::Memory
        );
        assert_eq!(
            FFIError::communication_timeout(1000).category(),
            ErrorCategory::Communication
        );
    }
}