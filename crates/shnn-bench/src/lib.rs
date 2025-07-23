//! Performance benchmarks for SHNN zero-dependency implementations
//!
//! This crate provides comprehensive benchmarks to validate that our custom
//! zero-dependency implementations meet or exceed the performance of the
//! original heavy dependencies (tokio, nalgebra, ndarray, crossbeam, serde).

#![warn(missing_docs)]

use std::time::{Duration, Instant};

pub mod compilation;
pub mod math;
pub mod async_runtime;
pub mod memory;
pub mod neuromorphic;

/// Benchmark result containing timing and memory metrics
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Test name
    pub name: String,
    /// Duration of the operation
    pub duration: Duration,
    /// Memory used in bytes (if available)
    pub memory_bytes: Option<usize>,
    /// Number of operations performed
    pub operations: u64,
    /// Operations per second
    pub ops_per_sec: f64,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    pub fn new(name: &str, duration: Duration, operations: u64) -> Self {
        let ops_per_sec = operations as f64 / duration.as_secs_f64();
        Self {
            name: name.to_string(),
            duration,
            memory_bytes: None,
            operations,
            ops_per_sec,
        }
    }

    /// Add memory usage information
    pub fn with_memory(mut self, bytes: usize) -> Self {
        self.memory_bytes = Some(bytes);
        self
    }

    /// Get throughput in MB/s (assuming operations represent data processed)
    pub fn throughput_mbps(&self, bytes_per_op: usize) -> f64 {
        let total_bytes = self.operations as f64 * bytes_per_op as f64;
        let mb = total_bytes / (1024.0 * 1024.0);
        mb / self.duration.as_secs_f64()
    }
}

/// Simple benchmark runner for measuring performance
pub struct BenchmarkRunner {
    /// Warmup iterations
    pub warmup_iterations: usize,
    /// Benchmark iterations
    pub benchmark_iterations: usize,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            benchmark_iterations: 10,
        }
    }
}

impl BenchmarkRunner {
    /// Run a benchmark function and return the result
    pub fn run<F>(&self, name: &str, mut benchmark_fn: F) -> BenchmarkResult
    where
        F: FnMut() -> u64,
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            benchmark_fn();
        }

        // Actual benchmark
        let start = Instant::now();
        let mut total_operations = 0;
        
        for _ in 0..self.benchmark_iterations {
            total_operations += benchmark_fn();
        }
        
        let duration = start.elapsed();
        BenchmarkResult::new(name, duration, total_operations)
    }

    /// Run a benchmark with memory measurement
    pub fn run_with_memory<F>(&self, name: &str, mut benchmark_fn: F) -> BenchmarkResult
    where
        F: FnMut() -> (u64, usize),
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            benchmark_fn();
        }

        // Actual benchmark
        let start = Instant::now();
        let mut total_operations = 0;
        let mut total_memory = 0;
        
        for _ in 0..self.benchmark_iterations {
            let (ops, mem) = benchmark_fn();
            total_operations += ops;
            total_memory += mem;
        }
        
        let duration = start.elapsed();
        let avg_memory = total_memory / self.benchmark_iterations;
        
        BenchmarkResult::new(name, duration, total_operations).with_memory(avg_memory)
    }
}

/// Comparison results between zero-dependency and legacy implementations
#[derive(Debug)]
pub struct ComparisonResult {
    /// Zero-dependency implementation result
    pub zero_deps: BenchmarkResult,
    /// Legacy implementation result (if available)
    pub legacy: Option<BenchmarkResult>,
    /// Performance improvement ratio (zero_deps / legacy)
    pub improvement_ratio: Option<f64>,
}

impl ComparisonResult {
    /// Create a new comparison result
    pub fn new(zero_deps: BenchmarkResult, legacy: Option<BenchmarkResult>) -> Self {
        let improvement_ratio = legacy.as_ref().map(|legacy_result| {
            legacy_result.ops_per_sec / zero_deps.ops_per_sec
        });

        Self {
            zero_deps,
            legacy,
            improvement_ratio,
        }
    }

    /// Get improvement percentage (positive means zero-deps is faster)
    pub fn improvement_percentage(&self) -> Option<f64> {
        self.improvement_ratio.map(|ratio| (ratio - 1.0) * 100.0)
    }

    /// Check if zero-deps implementation is faster
    pub fn is_faster(&self) -> Option<bool> {
        self.improvement_ratio.map(|ratio| ratio > 1.0)
    }
}

/// Memory usage tracker for benchmarks
#[derive(Debug, Default)]
pub struct MemoryTracker {
    peak_usage: usize,
    current_usage: usize,
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record memory allocation
    pub fn allocate(&mut self, bytes: usize) {
        self.current_usage += bytes;
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }

    /// Record memory deallocation
    pub fn deallocate(&mut self, bytes: usize) {
        self.current_usage = self.current_usage.saturating_sub(bytes);
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Get allocation count (simplified metric)
    pub fn allocation_count(&self) -> u64 {
        // For benchmarking purposes, use peak usage as proxy for allocation activity
        (self.peak_usage / 1024) as u64
    }

    /// Get deallocation count (simplified metric)
    pub fn deallocation_count(&self) -> u64 {
        // For benchmarking purposes, calculate estimated deallocations
        let allocated = self.allocation_count();
        let current_kb = (self.current_usage / 1024) as u64;
        allocated.saturating_sub(current_kb)
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        self.current_usage = 0;
        self.peak_usage = 0;
    }
}

/// Compilation time measurement utilities
pub mod compilation_utils {
    use std::process::Command;
    use std::time::{Duration, Instant};

    /// Measure compilation time for a specific crate with given features
    pub fn measure_compile_time(crate_path: &str, features: &[&str]) -> Result<Duration, Box<dyn std::error::Error>> {
        // Clean first
        let mut clean_cmd = Command::new("cargo");
        clean_cmd.arg("clean").current_dir(crate_path);
        clean_cmd.output()?;

        // Build with timing
        let mut build_cmd = Command::new("cargo");
        build_cmd.arg("build").current_dir(crate_path);
        
        if !features.is_empty() {
            build_cmd.arg("--features").arg(features.join(","));
        }

        let start = Instant::now();
        let output = build_cmd.output()?;
        let duration = start.elapsed();

        if !output.status.success() {
            return Err(format!("Compilation failed: {}", String::from_utf8_lossy(&output.stderr)).into());
        }

        Ok(duration)
    }

    /// Compare compilation times between zero-deps and legacy features
    pub fn compare_compilation_times(crate_path: &str) -> Result<(Duration, Duration), Box<dyn std::error::Error>> {
        let zero_deps_time = measure_compile_time(crate_path, &["zero-deps"])?;
        let legacy_time = measure_compile_time(crate_path, &["legacy-deps"])?;
        
        Ok((zero_deps_time, legacy_time))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result() {
        let result = BenchmarkResult::new("test", Duration::from_millis(100), 1000);
        assert_eq!(result.name, "test");
        assert_eq!(result.operations, 1000);
        assert!((result.ops_per_sec - 10000.0).abs() < 0.1);
    }

    #[test]
    fn test_benchmark_runner() {
        let runner = BenchmarkRunner::default();
        let result = runner.run("test", || 100);
        assert_eq!(result.operations, 100 * runner.benchmark_iterations as u64);
    }

    #[test]
    fn test_comparison_result() {
        let zero_deps = BenchmarkResult::new("zero", Duration::from_millis(50), 1000);
        let legacy = BenchmarkResult::new("legacy", Duration::from_millis(100), 1000);
        
        let comparison = ComparisonResult::new(zero_deps, Some(legacy));
        assert!(comparison.is_faster().unwrap());
        assert!((comparison.improvement_percentage().unwrap() - 100.0).abs() < 0.1);
    }
}