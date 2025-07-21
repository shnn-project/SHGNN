//! Comprehensive benchmarking suite for SHNN neuromorphic computing library
//!
//! This crate provides extensive benchmarking capabilities for all components of the SHNN
//! ecosystem, including performance comparisons against other frameworks, scalability
//! analysis, memory profiling, and hardware acceleration benchmarks.
//!
//! # Features
//!
//! - **Performance Benchmarking**: Comprehensive performance testing of all SHNN components
//! - **Scalability Analysis**: Testing performance across different network sizes and configurations
//! - **Memory Profiling**: Detailed memory usage analysis and optimization recommendations
//! - **Hardware Acceleration**: Benchmarking across different hardware accelerators
//! - **Comparison Framework**: Performance comparisons with other neuromorphic frameworks
//! - **Statistical Analysis**: Advanced statistical analysis of benchmark results
//! - **Visualization**: Rich plotting and visualization of benchmark results
//! - **Automated Reporting**: Generate comprehensive benchmark reports
//!
//! # Quick Start
//!
//! ```rust
//! use shnn_bench::{BenchmarkSuite, BenchmarkConfig};
//!
//! // Create benchmark configuration
//! let config = BenchmarkConfig::default()
//!     .with_network_sizes(vec![100, 1000, 10000])
//!     .with_iterations(100)
//!     .with_hardware_acceleration(true);
//!
//! // Run comprehensive benchmark suite
//! let mut suite = BenchmarkSuite::new(config);
//! let results = suite.run_all_benchmarks()?;
//!
//! // Generate report
//! results.generate_report("benchmark_results.html")?;
//! ```

pub mod benchmark;
pub mod comparison;
pub mod memory;
pub mod hardware;
pub mod statistics;
pub mod reporting;
pub mod visualization;
pub mod utils;

// Re-export main types
pub use benchmark::{BenchmarkSuite, BenchmarkConfig, BenchmarkResult, BenchmarkMetrics};
pub use comparison::{ComparisonFramework, FrameworkComparison, ComparisonResult};
pub use memory::{MemoryProfiler, MemoryReport, AllocationTracker};
pub use hardware::{HardwareBenchmark, AcceleratorBenchmark, HardwareReport};
pub use statistics::{StatisticalAnalysis, PerformanceStats, TrendAnalysis};
pub use reporting::{ReportGenerator, BenchmarkReport, ReportConfig};
pub use visualization::{PlotGenerator, ChartType, VisualizationConfig};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Error types for benchmarking operations
#[derive(thiserror::Error, Debug)]
pub enum BenchmarkError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Benchmark execution failed: {0}")]
    Execution(String),
    
    #[error("Hardware not available: {0}")]
    HardwareUnavailable(String),
    
    #[error("Memory profiling failed: {0}")]
    MemoryProfiling(String),
    
    #[error("Statistical analysis failed: {0}")]
    StatisticalAnalysis(String),
    
    #[error("Report generation failed: {0}")]
    ReportGeneration(String),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Main benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Network sizes to test
    pub network_sizes: Vec<usize>,
    
    /// Number of iterations per benchmark
    pub iterations: usize,
    
    /// Duration to run each benchmark
    pub duration: Duration,
    
    /// Enable hardware acceleration testing
    pub hardware_acceleration: bool,
    
    /// Enable memory profiling
    pub memory_profiling: bool,
    
    /// Enable statistical analysis
    pub statistical_analysis: bool,
    
    /// Output directory for results
    pub output_dir: String,
    
    /// Benchmark-specific configurations
    pub neuron_configs: NeuronBenchmarkConfig,
    pub network_configs: NetworkBenchmarkConfig,
    pub spike_configs: SpikeBenchmarkConfig,
    pub plasticity_configs: PlasticityBenchmarkConfig,
    
    /// Hardware configurations
    pub hardware_configs: Vec<HardwareConfig>,
    
    /// Comparison frameworks to test against
    pub comparison_frameworks: Vec<String>,
    
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    
    /// Warmup iterations before actual benchmarking
    pub warmup_iterations: usize,
    
    /// Enable verbose output
    pub verbose: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            network_sizes: vec![100, 500, 1000, 5000, 10000],
            iterations: 100,
            duration: Duration::from_secs(10),
            hardware_acceleration: true,
            memory_profiling: true,
            statistical_analysis: true,
            output_dir: "benchmark_results".to_string(),
            neuron_configs: NeuronBenchmarkConfig::default(),
            network_configs: NetworkBenchmarkConfig::default(),
            spike_configs: SpikeBenchmarkConfig::default(),
            plasticity_configs: PlasticityBenchmarkConfig::default(),
            hardware_configs: vec![
                HardwareConfig::cpu(),
                HardwareConfig::cuda(),
                HardwareConfig::opencl(),
            ],
            comparison_frameworks: vec![
                "brian2".to_string(),
                "nest".to_string(),
                "neuron".to_string(),
                "spinnaker".to_string(),
            ],
            seed: Some(42),
            warmup_iterations: 10,
            verbose: false,
        }
    }
}

impl BenchmarkConfig {
    /// Create new benchmark configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set network sizes to test
    pub fn with_network_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.network_sizes = sizes;
        self
    }
    
    /// Set number of iterations
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }
    
    /// Set benchmark duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }
    
    /// Enable/disable hardware acceleration testing
    pub fn with_hardware_acceleration(mut self, enabled: bool) -> Self {
        self.hardware_acceleration = enabled;
        self
    }
    
    /// Enable/disable memory profiling
    pub fn with_memory_profiling(mut self, enabled: bool) -> Self {
        self.memory_profiling = enabled;
        self
    }
    
    /// Set output directory
    pub fn with_output_dir<S: Into<String>>(mut self, dir: S) -> Self {
        self.output_dir = dir.into();
        self
    }
    
    /// Add hardware configuration
    pub fn add_hardware_config(mut self, config: HardwareConfig) -> Self {
        self.hardware_configs.push(config);
        self
    }
    
    /// Add comparison framework
    pub fn add_comparison_framework<S: Into<String>>(mut self, framework: S) -> Self {
        self.comparison_frameworks.push(framework.into());
        self
    }
    
    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Enable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.network_sizes.is_empty() {
            return Err(BenchmarkError::Configuration("No network sizes specified".to_string()));
        }
        
        if self.iterations == 0 {
            return Err(BenchmarkError::Configuration("Iterations must be greater than 0".to_string()));
        }
        
        if self.duration.is_zero() {
            return Err(BenchmarkError::Configuration("Duration must be greater than 0".to_string()));
        }
        
        // Validate hardware configurations
        for hw_config in &self.hardware_configs {
            hw_config.validate()?;
        }
        
        Ok(())
    }
}

/// Neuron-specific benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronBenchmarkConfig {
    /// Neuron types to benchmark
    pub neuron_types: Vec<String>,
    
    /// Input current ranges
    pub current_ranges: Vec<(f32, f32)>,
    
    /// Time step sizes
    pub time_steps: Vec<f32>,
    
    /// Test different parameter sets
    pub parameter_variations: bool,
}

impl Default for NeuronBenchmarkConfig {
    fn default() -> Self {
        Self {
            neuron_types: vec!["LIF".to_string(), "AdEx".to_string(), "Izhikevich".to_string()],
            current_ranges: vec![(0.0, 1.0), (1.0, 5.0), (5.0, 10.0)],
            time_steps: vec![0.0001, 0.001, 0.01],
            parameter_variations: true,
        }
    }
}

/// Network-specific benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkBenchmarkConfig {
    /// Network topologies to test
    pub topologies: Vec<String>,
    
    /// Connectivity patterns
    pub connectivity_patterns: Vec<f32>,
    
    /// Test different architectures
    pub architectures: Vec<String>,
    
    /// Enable plasticity during benchmarks
    pub enable_plasticity: bool,
}

impl Default for NetworkBenchmarkConfig {
    fn default() -> Self {
        Self {
            topologies: vec!["random".to_string(), "small_world".to_string(), "scale_free".to_string()],
            connectivity_patterns: vec![0.01, 0.05, 0.1, 0.2],
            architectures: vec!["feedforward".to_string(), "recurrent".to_string(), "reservoir".to_string()],
            enable_plasticity: true,
        }
    }
}

/// Spike processing benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeBenchmarkConfig {
    /// Spike rates to test (Hz)
    pub spike_rates: Vec<f32>,
    
    /// Spike patterns
    pub patterns: Vec<String>,
    
    /// Encoding methods
    pub encoding_methods: Vec<String>,
    
    /// Buffer sizes
    pub buffer_sizes: Vec<usize>,
}

impl Default for SpikeBenchmarkConfig {
    fn default() -> Self {
        Self {
            spike_rates: vec![1.0, 10.0, 100.0, 1000.0],
            patterns: vec!["poisson".to_string(), "regular".to_string(), "burst".to_string()],
            encoding_methods: vec!["rate".to_string(), "temporal".to_string(), "population".to_string()],
            buffer_sizes: vec![1000, 10000, 100000],
        }
    }
}

/// Plasticity benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityBenchmarkConfig {
    /// Plasticity rules to test
    pub rules: Vec<String>,
    
    /// Learning rates
    pub learning_rates: Vec<f32>,
    
    /// STDP window parameters
    pub stdp_windows: Vec<(f32, f32)>,
    
    /// Test homeostatic mechanisms
    pub test_homeostasis: bool,
}

impl Default for PlasticityBenchmarkConfig {
    fn default() -> Self {
        Self {
            rules: vec!["STDP".to_string(), "BCM".to_string(), "Oja".to_string(), "Hebbian".to_string()],
            learning_rates: vec![0.001, 0.01, 0.1],
            stdp_windows: vec![(10.0, 10.0), (20.0, 20.0), (50.0, 50.0)],
            test_homeostasis: true,
        }
    }
}

/// Hardware configuration for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Hardware type
    pub hardware_type: String,
    
    /// Device ID (for multi-device systems)
    pub device_id: Option<usize>,
    
    /// Hardware-specific parameters
    pub parameters: HashMap<String, String>,
    
    /// Enable this hardware for testing
    pub enabled: bool,
}

impl HardwareConfig {
    /// Create CPU configuration
    pub fn cpu() -> Self {
        Self {
            hardware_type: "CPU".to_string(),
            device_id: None,
            parameters: HashMap::new(),
            enabled: true,
        }
    }
    
    /// Create CUDA configuration
    pub fn cuda() -> Self {
        Self {
            hardware_type: "CUDA".to_string(),
            device_id: Some(0),
            parameters: HashMap::new(),
            enabled: true,
        }
    }
    
    /// Create OpenCL configuration
    pub fn opencl() -> Self {
        Self {
            hardware_type: "OpenCL".to_string(),
            device_id: Some(0),
            parameters: HashMap::new(),
            enabled: true,
        }
    }
    
    /// Create FPGA configuration
    pub fn fpga() -> Self {
        Self {
            hardware_type: "FPGA".to_string(),
            device_id: Some(0),
            parameters: HashMap::new(),
            enabled: false, // Disabled by default as FPGA might not be available
        }
    }
    
    /// Create Loihi configuration
    pub fn loihi() -> Self {
        Self {
            hardware_type: "Loihi".to_string(),
            device_id: Some(0),
            parameters: HashMap::new(),
            enabled: false, // Disabled by default as Loihi might not be available
        }
    }
    
    /// Create SpiNNaker configuration
    pub fn spinnaker() -> Self {
        Self {
            hardware_type: "SpiNNaker".to_string(),
            device_id: Some(0),
            parameters: HashMap::new(),
            enabled: false, // Disabled by default as SpiNNaker might not be available
        }
    }
    
    /// Validate hardware configuration
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.hardware_type.is_empty() {
            return Err(BenchmarkError::Configuration("Hardware type cannot be empty".to_string()));
        }
        
        Ok(())
    }
}

/// System information for benchmark context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    
    /// CPU information
    pub cpu: CpuInfo,
    
    /// Memory information
    pub memory: MemoryInfo,
    
    /// GPU information
    pub gpus: Vec<GpuInfo>,
    
    /// Rust version
    pub rust_version: String,
    
    /// SHNN version
    pub shnn_version: String,
    
    /// Benchmark timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub name: String,
    pub cores: usize,
    pub threads: usize,
    pub frequency: u64,
    pub cache_size: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total: u64,
    pub available: u64,
    pub swap_total: u64,
    pub swap_free: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub memory: u64,
    pub compute_capability: Option<String>,
    pub driver_version: Option<String>,
}

impl SystemInfo {
    /// Collect system information
    pub fn collect() -> Result<Self> {
        use sysinfo::{System, SystemExt, CpuExt};
        
        let mut system = System::new_all();
        system.refresh_all();
        
        let cpu_info = if let Some(cpu) = system.cpus().first() {
            CpuInfo {
                name: cpu.brand().to_string(),
                cores: system.physical_core_count().unwrap_or(0),
                threads: system.cpus().len(),
                frequency: cpu.frequency(),
                cache_size: None,
            }
        } else {
            CpuInfo {
                name: "Unknown".to_string(),
                cores: 0,
                threads: 0,
                frequency: 0,
                cache_size: None,
            }
        };
        
        let memory_info = MemoryInfo {
            total: system.total_memory(),
            available: system.available_memory(),
            swap_total: system.total_swap(),
            swap_free: system.free_swap(),
        };
        
        // GPU information would be collected here using hardware-specific APIs
        let gpus = Vec::new();
        
        Ok(SystemInfo {
            os: system.long_os_version().unwrap_or_else(|| "Unknown".to_string()),
            cpu: cpu_info,
            memory: memory_info,
            gpus,
            rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
            shnn_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now(),
        })
    }
}

/// Utility functions for benchmarking
pub mod bench_utils {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    /// Generate reproducible test data
    pub fn generate_test_spikes(
        num_neurons: usize,
        duration: f64,
        rate: f32,
        seed: Option<u64>,
    ) -> Vec<(u32, f64, f32)> {
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        
        let mut spikes = Vec::new();
        
        for neuron_id in 0..num_neurons {
            let mut time = 0.0;
            while time < duration {
                let interval = -((1.0 - rng.gen::<f32>()).ln()) / rate;
                time += interval;
                
                if time < duration {
                    spikes.push((neuron_id as u32, time, 1.0));
                }
            }
        }
        
        // Sort by time
        spikes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        spikes
    }
    
    /// Measure execution time with high precision
    pub fn measure_time<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }
    
    /// Run function multiple times and collect statistics
    pub fn benchmark_function<F, R>(
        f: F,
        iterations: usize,
        warmup: usize,
    ) -> (Vec<R>, Vec<Duration>)
    where
        F: Fn() -> R,
    {
        // Warmup
        for _ in 0..warmup {
            let _ = f();
        }
        
        // Actual benchmarking
        let mut results = Vec::with_capacity(iterations);
        let mut durations = Vec::with_capacity(iterations);
        
        for _ in 0..iterations {
            let (result, duration) = measure_time(&f);
            results.push(result);
            durations.push(duration);
        }
        
        (results, durations)
    }
    
    /// Calculate basic statistics from durations
    pub fn calculate_stats(durations: &[Duration]) -> (Duration, Duration, Duration, Duration) {
        if durations.is_empty() {
            return (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);
        }
        
        let mut sorted = durations.to_vec();
        sorted.sort();
        
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let median = sorted[sorted.len() / 2];
        
        let sum: Duration = sorted.iter().sum();
        let mean = sum / sorted.len() as u32;
        
        (min, max, mean, median)
    }
    
    /// Format duration for human-readable output
    pub fn format_duration(duration: Duration) -> String {
        let nanos = duration.as_nanos();
        
        if nanos < 1_000 {
            format!("{}ns", nanos)
        } else if nanos < 1_000_000 {
            format!("{:.2}μs", nanos as f64 / 1_000.0)
        } else if nanos < 1_000_000_000 {
            format!("{:.2}ms", nanos as f64 / 1_000_000.0)
        } else {
            format!("{:.2}s", nanos as f64 / 1_000_000_000.0)
        }
    }
    
    /// Create progress bar for long-running benchmarks
    pub fn create_progress_bar(length: usize, message: &str) -> indicatif::ProgressBar {
        let pb = indicatif::ProgressBar::new(length as u64);
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-")
        );
        pb.set_message(message.to_string());
        pb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert!(!config.network_sizes.is_empty());
        assert!(config.iterations > 0);
        assert!(!config.duration.is_zero());
    }
    
    #[test]
    fn test_benchmark_config_builder() {
        let config = BenchmarkConfig::new()
            .with_network_sizes(vec![100, 1000])
            .with_iterations(50)
            .with_hardware_acceleration(false);
        
        assert_eq!(config.network_sizes, vec![100, 1000]);
        assert_eq!(config.iterations, 50);
        assert!(!config.hardware_acceleration);
    }
    
    #[test]
    fn test_hardware_config_creation() {
        let cpu_config = HardwareConfig::cpu();
        assert_eq!(cpu_config.hardware_type, "CPU");
        assert!(cpu_config.enabled);
        
        let cuda_config = HardwareConfig::cuda();
        assert_eq!(cuda_config.hardware_type, "CUDA");
        assert_eq!(cuda_config.device_id, Some(0));
    }
    
    #[test]
    fn test_config_validation() {
        let valid_config = BenchmarkConfig::default();
        assert!(valid_config.validate().is_ok());
        
        let invalid_config = BenchmarkConfig {
            network_sizes: vec![],
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }
    
    #[test]
    fn test_system_info_collection() {
        let system_info = SystemInfo::collect();
        assert!(system_info.is_ok());
        
        let info = system_info.unwrap();
        assert!(!info.os.is_empty());
        assert!(!info.rust_version.is_empty());
    }
    
    #[test]
    fn test_bench_utils() {
        use bench_utils::*;
        
        // Test spike generation
        let spikes = generate_test_spikes(10, 1.0, 10.0, Some(42));
        assert!(!spikes.is_empty());
        
        // Test time measurement
        let (result, duration) = measure_time(|| {
            std::thread::sleep(Duration::from_millis(1));
            42
        });
        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(1));
        
        // Test statistics calculation
        let durations = vec![
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(3),
        ];
        let (min, max, mean, median) = calculate_stats(&durations);
        assert_eq!(min, Duration::from_millis(1));
        assert_eq!(max, Duration::from_millis(3));
        assert_eq!(median, Duration::from_millis(2));
        
        // Test duration formatting
        assert_eq!(format_duration(Duration::from_nanos(500)), "500ns");
        assert_eq!(format_duration(Duration::from_micros(500)), "500.00μs");
        assert_eq!(format_duration(Duration::from_millis(500)), "500.00ms");
    }
}