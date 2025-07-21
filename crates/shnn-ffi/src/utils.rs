//! Utility functions for FFI operations
//!
//! This module provides helper functions for data conversion, memory management,
//! and other common operations used throughout the FFI interface.

use crate::{
    error::{FFIError, FFIResult},
    types::{SpikeData, NeuronState, NetworkConfig, PerformanceMetrics},
};
use std::{
    ffi::{CStr, CString},
    os::raw::c_char,
    ptr,
    slice,
    time::{SystemTime, UNIX_EPOCH},
};

/// Utility functions for C string handling
pub mod cstring {
    use super::*;
    
    /// Convert Rust string to C string
    pub fn to_c_string(s: &str) -> FFIResult<CString> {
        CString::new(s).map_err(|_| FFIError::invalid_config("Invalid string for C conversion"))
    }
    
    /// Convert C string to Rust string
    pub unsafe fn from_c_string(ptr: *const c_char) -> FFIResult<String> {
        if ptr.is_null() {
            return Err(FFIError::invalid_config("Null pointer for string"));
        }
        
        let c_str = CStr::from_ptr(ptr);
        c_str.to_str()
            .map(|s| s.to_owned())
            .map_err(|_| FFIError::invalid_config("Invalid UTF-8 in C string"))
    }
    
    /// Get length of C string
    pub unsafe fn c_string_len(ptr: *const c_char) -> usize {
        if ptr.is_null() {
            return 0;
        }
        CStr::from_ptr(ptr).to_bytes().len()
    }
}

/// Utility functions for array/slice handling
pub mod array {
    use super::*;
    
    /// Convert C array to Rust slice
    pub unsafe fn from_c_array<T>(ptr: *const T, len: usize) -> Option<&'static [T]> {
        if ptr.is_null() || len == 0 {
            None
        } else {
            Some(slice::from_raw_parts(ptr, len))
        }
    }
    
    /// Convert mutable C array to Rust slice
    pub unsafe fn from_c_array_mut<T>(ptr: *mut T, len: usize) -> Option<&'static mut [T]> {
        if ptr.is_null() || len == 0 {
            None
        } else {
            Some(slice::from_raw_parts_mut(ptr, len))
        }
    }
    
    /// Copy Rust slice to C array
    pub unsafe fn to_c_array<T: Copy>(
        src: &[T],
        dst: *mut T,
        max_len: usize,
    ) -> FFIResult<usize> {
        if dst.is_null() {
            return Err(FFIError::invalid_config("Null destination pointer"));
        }
        
        let copy_len = src.len().min(max_len);
        if copy_len > 0 {
            ptr::copy_nonoverlapping(src.as_ptr(), dst, copy_len);
        }
        
        Ok(copy_len)
    }
    
    /// Allocate and copy Rust Vec to C array
    pub fn vec_to_c_array<T: Copy>(vec: Vec<T>) -> (*mut T, usize) {
        if vec.is_empty() {
            return (ptr::null_mut(), 0);
        }
        
        let len = vec.len();
        let ptr = vec.as_ptr() as *mut T;
        std::mem::forget(vec); // Prevent deallocation
        (ptr, len)
    }
    
    /// Free C array allocated by vec_to_c_array
    pub unsafe fn free_c_array<T>(ptr: *mut T, len: usize) {
        if !ptr.is_null() && len > 0 {
            Vec::from_raw_parts(ptr, len, len);
            // Vec will be dropped and memory freed
        }
    }
}

/// Utility functions for error handling in C context
pub mod error_handling {
    use super::*;
    use crate::error::FFIErrorCode;
    
    /// Convert FFIResult to C-style error code and optional error message
    pub fn result_to_c_error(
        result: FFIResult<()>,
        error_msg_ptr: *mut *mut c_char,
    ) -> i32 {
        match result {
            Ok(()) => FFIErrorCode::Success as i32,
            Err(error) => {
                // Set error message if pointer provided
                if !error_msg_ptr.is_null() {
                    let error_msg = error.to_string();
                    match CString::new(error_msg) {
                        Ok(c_string) => unsafe {
                            *error_msg_ptr = c_string.into_raw();
                        },
                        Err(_) => unsafe {
                            *error_msg_ptr = ptr::null_mut();
                        },
                    }
                }
                
                FFIErrorCode::from(&error) as i32
            }
        }
    }
    
    /// Free error message allocated by result_to_c_error
    pub unsafe fn free_error_message(error_msg: *mut c_char) {
        if !error_msg.is_null() {
            CString::from_raw(error_msg);
            // CString will be dropped and memory freed
        }
    }
    
    /// Get error message for error code
    pub fn get_error_message(error_code: i32) -> &'static str {
        match error_code {
            x if x == FFIErrorCode::Success as i32 => "Success",
            x if x == FFIErrorCode::UnsupportedHardware as i32 => "Unsupported hardware accelerator",
            x if x == FFIErrorCode::InvalidAcceleratorId as i32 => "Invalid accelerator ID",
            x if x == FFIErrorCode::InvalidNetworkId as i32 => "Invalid network ID",
            x if x == FFIErrorCode::HardwareInitializationFailed as i32 => "Hardware initialization failed",
            x if x == FFIErrorCode::MemoryAllocationFailed as i32 => "Memory allocation failed",
            x if x == FFIErrorCode::NetworkDeploymentFailed as i32 => "Network deployment failed",
            x if x == FFIErrorCode::SpikeProcessingFailed as i32 => "Spike processing failed",
            x if x == FFIErrorCode::InvalidConfiguration as i32 => "Invalid configuration",
            x if x == FFIErrorCode::DriverError as i32 => "Hardware driver error",
            x if x == FFIErrorCode::CommunicationTimeout as i32 => "Communication timeout",
            x if x == FFIErrorCode::ResourceLimitExceeded as i32 => "Resource limit exceeded",
            x if x == FFIErrorCode::SynchronizationError as i32 => "Synchronization error",
            x if x == FFIErrorCode::VersionMismatch as i32 => "Version mismatch",
            _ => "Unknown error",
        }
    }
}

/// Utility functions for time handling
pub mod time {
    use super::*;
    
    /// Get current timestamp in milliseconds
    pub fn current_timestamp_ms() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64() * 1000.0
    }
    
    /// Get high-resolution timestamp in microseconds
    pub fn current_timestamp_us() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }
    
    /// Convert timestamp to human-readable string
    pub fn timestamp_to_string(timestamp_ms: f64) -> String {
        let secs = (timestamp_ms / 1000.0) as u64;
        let nanos = ((timestamp_ms % 1000.0) * 1_000_000.0) as u32;
        
        match UNIX_EPOCH.checked_add(std::time::Duration::new(secs, nanos)) {
            Some(time) => {
                format!("{:?}", time)
            }
            None => "Invalid timestamp".to_string(),
        }
    }
}

/// Utility functions for data validation
pub mod validation {
    use super::*;
    
    /// Validate network configuration
    pub fn validate_network_config(config: &NetworkConfig) -> FFIResult<()> {
        if config.num_neurons == 0 {
            return Err(FFIError::invalid_config("Number of neurons must be greater than 0"));
        }
        
        if config.num_connections == 0 {
            return Err(FFIError::invalid_config("Number of connections must be greater than 0"));
        }
        
        if config.connectivity < 0.0 || config.connectivity > 1.0 {
            return Err(FFIError::invalid_config("Connectivity must be between 0.0 and 1.0"));
        }
        
        if config.dt <= 0.0 {
            return Err(FFIError::invalid_config("Time step must be greater than 0"));
        }
        
        if config.input_size == 0 {
            return Err(FFIError::invalid_config("Input size must be greater than 0"));
        }
        
        if config.output_size == 0 {
            return Err(FFIError::invalid_config("Output size must be greater than 0"));
        }
        
        // Validate neuron configuration
        validate_neuron_config(&config.neuron_config)?;
        
        // Validate plasticity configuration
        validate_plasticity_config(&config.plasticity_config)?;
        
        Ok(())
    }
    
    /// Validate neuron configuration
    pub fn validate_neuron_config(config: &crate::types::NeuronConfig) -> FFIResult<()> {
        if config.tau_m <= 0.0 {
            return Err(FFIError::invalid_config("Membrane time constant must be positive"));
        }
        
        if config.v_thresh <= config.v_rest {
            return Err(FFIError::invalid_config("Threshold must be greater than resting potential"));
        }
        
        if config.tau_ref < 0.0 {
            return Err(FFIError::invalid_config("Refractory period must be non-negative"));
        }
        
        Ok(())
    }
    
    /// Validate plasticity configuration
    pub fn validate_plasticity_config(config: &crate::types::PlasticityConfig) -> FFIResult<()> {
        if config.enabled {
            if config.learning_rate <= 0.0 {
                return Err(FFIError::invalid_config("Learning rate must be positive"));
            }
            
            if config.stdp_window <= 0.0 {
                return Err(FFIError::invalid_config("STDP window must be positive"));
            }
            
            if config.weight_min > config.weight_max {
                return Err(FFIError::invalid_config("Minimum weight must not exceed maximum weight"));
            }
        }
        
        Ok(())
    }
    
    /// Validate spike data
    pub fn validate_spike_data(spikes: &[SpikeData]) -> FFIResult<()> {
        for (i, spike) in spikes.iter().enumerate() {
            if spike.timestamp < 0.0 {
                return Err(FFIError::invalid_config(
                    format!("Spike {} has negative timestamp", i)
                ));
            }
            
            if spike.amplitude < 0.0 {
                return Err(FFIError::invalid_config(
                    format!("Spike {} has negative amplitude", i)
                ));
            }
        }
        
        Ok(())
    }
}

/// Utility functions for memory management
pub mod memory {
    use super::*;
    
    /// Calculate optimal batch size based on available memory
    pub fn calculate_optimal_batch_size(
        total_spikes: usize,
        available_memory: u64,
        spike_size: usize,
    ) -> usize {
        let max_spikes = (available_memory as usize / spike_size).max(1);
        total_spikes.min(max_spikes)
    }
    
    /// Estimate memory usage for spike processing
    pub fn estimate_spike_memory_usage(
        num_spikes: usize,
        num_neurons: u32,
        buffer_multiplier: f32,
    ) -> u64 {
        let spike_memory = num_spikes * std::mem::size_of::<SpikeData>();
        let neuron_memory = num_neurons as usize * std::mem::size_of::<NeuronState>();
        let buffer_memory = (spike_memory as f32 * buffer_multiplier) as usize;
        
        (spike_memory + neuron_memory + buffer_memory) as u64
    }
    
    /// Check if system has enough memory for operation
    pub fn check_memory_availability(required_memory: u64) -> bool {
        // This is a simplified check - in practice, you'd query system memory
        let available_memory = get_available_memory();
        available_memory >= required_memory
    }
    
    /// Get available system memory (simplified implementation)
    fn get_available_memory() -> u64 {
        // In a real implementation, this would query the OS
        // For now, assume 4GB available
        4 * 1024 * 1024 * 1024
    }
}

/// Utility functions for performance optimization
pub mod performance {
    use super::*;
    
    /// Calculate throughput in spikes per second
    pub fn calculate_throughput(num_spikes: usize, duration_ms: f64) -> f64 {
        if duration_ms <= 0.0 {
            return 0.0;
        }
        (num_spikes as f64) / (duration_ms / 1000.0)
    }
    
    /// Calculate latency per spike in microseconds
    pub fn calculate_latency_per_spike(total_latency_us: f64, num_spikes: usize) -> f64 {
        if num_spikes == 0 {
            return 0.0;
        }
        total_latency_us / (num_spikes as f64)
    }
    
    /// Estimate optimal number of processing threads
    pub fn estimate_optimal_threads(workload_size: usize, min_work_per_thread: usize) -> usize {
        let max_threads = num_cpus::get();
        let optimal_threads = (workload_size / min_work_per_thread).max(1);
        optimal_threads.min(max_threads)
    }
    
    /// Create performance summary
    pub fn create_performance_summary(metrics: &PerformanceMetrics) -> String {
        format!(
            "Performance Summary:\n\
             - Execution Time: {:.2} ms\n\
             - Throughput: {:.0} spikes/sec\n\
             - Memory Usage: {} MB\n\
             - Power Consumption: {:.1} W\n\
             - GPU Utilization: {:.1}%\n\
             - Memory Utilization: {:.1}%",
            metrics.execution_time_ms,
            metrics.spikes_per_second,
            metrics.memory_usage / (1024 * 1024),
            metrics.power_consumption,
            metrics.gpu_utilization * 100.0,
            metrics.memory_utilization * 100.0
        )
    }
}

/// Utility functions for logging and debugging
pub mod logging {
    use super::*;
    
    /// Log levels for FFI operations
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum LogLevel {
        Trace,
        Debug,
        Info,
        Warn,
        Error,
    }
    
    /// Simple logging function (in practice, would integrate with proper logging framework)
    pub fn log_message(level: LogLevel, component: &str, message: &str) {
        let timestamp = time::current_timestamp_ms();
        let level_str = match level {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
        };
        
        eprintln!("[{:.3}] {} [{}] {}", timestamp, level_str, component, message);
    }
    
    /// Log network configuration
    pub fn log_network_config(config: &NetworkConfig) {
        log_message(
            LogLevel::Info,
            "CONFIG",
            &format!(
                "Network: {} neurons, {} connections, connectivity: {:.3}, dt: {:.6}",
                config.num_neurons,
                config.num_connections,
                config.connectivity,
                config.dt
            ),
        );
    }
    
    /// Log performance metrics
    pub fn log_performance_metrics(metrics: &PerformanceMetrics) {
        log_message(
            LogLevel::Info,
            "PERF",
            &format!(
                "Execution: {:.2}ms, Throughput: {:.0} spikes/s, Memory: {}MB",
                metrics.execution_time_ms,
                metrics.spikes_per_second,
                metrics.memory_usage / (1024 * 1024)
            ),
        );
    }
}

/// Utility functions for testing and benchmarking
pub mod testing {
    use super::*;
    
    /// Generate random spike data for testing
    pub fn generate_random_spikes(
        count: usize,
        max_neuron_id: u32,
        max_timestamp: f64,
    ) -> Vec<SpikeData> {
        let mut spikes = Vec::with_capacity(count);
        
        for _ in 0..count {
            spikes.push(SpikeData {
                neuron_id: fastrand::u32(0..max_neuron_id),
                timestamp: fastrand::f64() * max_timestamp,
                amplitude: fastrand::f32() * 2.0, // 0.0 to 2.0
            });
        }
        
        // Sort by timestamp for realistic ordering
        spikes.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());
        spikes
    }
    
    /// Create a test network configuration
    pub fn create_test_network_config(
        num_neurons: u32,
        connectivity: f32,
    ) -> NetworkConfig {
        NetworkConfig {
            num_neurons,
            num_connections: (num_neurons as f32 * connectivity) as u32,
            connectivity,
            dt: 0.001,
            input_size: num_neurons / 10,
            output_size: num_neurons / 20,
            hidden_layers: vec![num_neurons / 2],
            ..Default::default()
        }
    }
    
    /// Measure execution time of a function
    pub fn measure_execution_time<F, R>(f: F) -> (R, f64)
    where
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = f();
        let duration = start.elapsed().as_secs_f64() * 1000.0; // Convert to milliseconds
        (result, duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cstring_conversion() {
        let rust_str = "Hello, World!";
        let c_string = cstring::to_c_string(rust_str).unwrap();
        
        unsafe {
            let converted_back = cstring::from_c_string(c_string.as_ptr()).unwrap();
            assert_eq!(rust_str, converted_back);
        }
    }
    
    #[test]
    fn test_array_operations() {
        let data = vec![1, 2, 3, 4, 5];
        let (ptr, len) = array::vec_to_c_array(data.clone());
        
        unsafe {
            let slice = array::from_c_array(ptr, len).unwrap();
            assert_eq!(slice, &data);
            
            array::free_c_array(ptr, len);
        }
    }
    
    #[test]
    fn test_validation() {
        let mut config = NetworkConfig::default();
        assert!(validation::validate_network_config(&config).is_ok());
        
        config.num_neurons = 0;
        assert!(validation::validate_network_config(&config).is_err());
    }
    
    #[test]
    fn test_performance_calculations() {
        let throughput = performance::calculate_throughput(1000, 500.0);
        assert_eq!(throughput, 2000.0); // 1000 spikes in 0.5 seconds = 2000 spikes/sec
        
        let latency = performance::calculate_latency_per_spike(1000.0, 100);
        assert_eq!(latency, 10.0); // 1000 us / 100 spikes = 10 us/spike
    }
    
    #[test]
    fn test_spike_generation() {
        let spikes = testing::generate_random_spikes(100, 1000, 10.0);
        assert_eq!(spikes.len(), 100);
        
        // Check that spikes are sorted by timestamp
        for i in 1..spikes.len() {
            assert!(spikes[i].timestamp >= spikes[i - 1].timestamp);
        }
    }
    
    #[test]
    fn test_memory_estimation() {
        let memory_usage = memory::estimate_spike_memory_usage(1000, 500, 2.0);
        assert!(memory_usage > 0);
        
        let batch_size = memory::calculate_optimal_batch_size(
            10000,
            1024 * 1024, // 1 MB
            std::mem::size_of::<SpikeData>(),
        );
        assert!(batch_size > 0);
        assert!(batch_size <= 10000);
    }
}