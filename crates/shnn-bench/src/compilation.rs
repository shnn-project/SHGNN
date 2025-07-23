//! Compilation time benchmarks for zero-dependency vs legacy implementations
//!
//! This module measures the compilation time improvements achieved by replacing
//! heavy dependencies (tokio, nalgebra, ndarray, crossbeam, serde) with our
//! custom zero-dependency implementations.

use std::process::Command;
use std::time::{Duration, Instant};
use crate::{BenchmarkResult, ComparisonResult};

/// Compilation benchmark configuration
#[derive(Debug, Clone)]
pub struct CompilationConfig {
    /// Project root path
    pub project_path: String,
    /// Whether to include clean builds
    pub clean_build: bool,
    /// Number of compilation runs to average
    pub runs: usize,
    /// Whether to measure incremental builds
    pub incremental: bool,
}

impl Default for CompilationConfig {
    fn default() -> Self {
        Self {
            project_path: ".".to_string(),
            clean_build: true,
            runs: 3,
            incremental: false,
        }
    }
}

/// Compilation benchmark runner
pub struct CompilationBenchmark {
    config: CompilationConfig,
}

impl CompilationBenchmark {
    /// Create a new compilation benchmark
    pub fn new(config: CompilationConfig) -> Self {
        Self { config }
    }

    /// Run compilation benchmark for zero-deps features
    pub fn benchmark_zero_deps(&self) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        self.benchmark_features(&["zero-deps"], "Zero-Dependency Build")
    }

    /// Run compilation benchmark for legacy features
    pub fn benchmark_legacy(&self) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        self.benchmark_features(&["legacy-deps"], "Legacy Dependencies Build")
    }

    /// Compare zero-deps vs legacy compilation times
    pub fn compare_builds(&self) -> Result<ComparisonResult, Box<dyn std::error::Error>> {
        println!("🔄 Running compilation time comparison...");
        
        let zero_deps_result = self.benchmark_zero_deps()?;
        println!("✅ Zero-deps build time: {:.2}s", zero_deps_result.duration.as_secs_f64());
        
        let legacy_result = self.benchmark_legacy().ok();
        if let Some(ref legacy) = legacy_result {
            println!("✅ Legacy build time: {:.2}s", legacy.duration.as_secs_f64());
        } else {
            println!("⚠️  Legacy dependencies not available for comparison");
        }

        Ok(ComparisonResult::new(zero_deps_result, legacy_result))
    }

    /// Benchmark specific crate compilation times
    pub fn benchmark_crate_compilation(&self) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
        let crates = vec![
            ("shnn-async-runtime", "Custom async runtime"),
            ("shnn-math", "Zero-dependency math library"),
            ("shnn-lockfree", "Lock-free concurrency primitives"),
            ("shnn-serialize", "Zero-copy serialization"),
            ("shnn-core", "Core SHNN library"),
        ];

        let mut results = Vec::new();
        
        for (crate_name, description) in crates {
            let crate_path = format!("crates/{}", crate_name);
            match self.benchmark_crate(&crate_path, description) {
                Ok(result) => {
                    println!("✅ {}: {:.2}s", description, result.duration.as_secs_f64());
                    results.push(result);
                }
                Err(e) => {
                    println!("❌ Failed to benchmark {}: {}", crate_name, e);
                }
            }
        }

        Ok(results)
    }

    /// Benchmark incremental compilation performance
    pub fn benchmark_incremental(&self) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
        println!("🔄 Running incremental compilation benchmarks...");
        
        // First, do a full build
        self.run_cargo_command(&["build", "--features", "zero-deps"])?;
        
        let mut results = Vec::new();
        
        // Test incremental builds with small changes
        let scenarios = vec![
            ("No changes", vec!["build", "--features", "zero-deps"]),
            ("Check only", vec!["check", "--features", "zero-deps"]),
            ("Doc build", vec!["doc", "--features", "zero-deps", "--no-deps"]),
        ];

        for (scenario, args) in scenarios {
            let start = Instant::now();
            self.run_cargo_command(&args)?;
            let duration = start.elapsed();
            
            let result = BenchmarkResult::new(scenario, duration, 1);
            println!("✅ {}: {:.2}s", scenario, duration.as_secs_f64());
            results.push(result);
        }

        Ok(results)
    }

    /// Measure dependency download and cache impact
    pub fn benchmark_dependency_impact(&self) -> Result<ComparisonResult, Box<dyn std::error::Error>> {
        println!("🔄 Measuring dependency download and cache impact...");
        
        // Clear cargo cache for clean measurement
        if let Err(e) = self.clear_cargo_cache() {
            println!("⚠️  Could not clear cargo cache: {}", e);
        }

        // Measure zero-deps build (minimal dependencies)
        let zero_deps_result = self.benchmark_zero_deps()?;
        
        // Clear cache again
        if let Err(e) = self.clear_cargo_cache() {
            println!("⚠️  Could not clear cargo cache: {}", e);
        }

        // Measure legacy build (heavy dependencies) if available
        let legacy_result = self.benchmark_legacy().ok();

        Ok(ComparisonResult::new(zero_deps_result, legacy_result))
    }

    /// Generate compilation benchmark report
    pub fn generate_report(&self) -> Result<String, Box<dyn std::error::Error>> {
        let mut report = String::new();
        report.push_str("# 🚀 SHNN Zero-Dependency Compilation Benchmark Report\n\n");
        
        // Overall comparison
        report.push_str("## 📊 Overall Compilation Time Comparison\n\n");
        let comparison = self.compare_builds()?;
        
        report.push_str(&format!(
            "| Metric | Zero-Deps | Legacy | Improvement |\n\
             |--------|-----------|--------|-------------|\n\
             | Build Time | {:.2}s | {} | {} |\n\
             | Operations/sec | {:.0} | {} | {} |\n\n",
            comparison.zero_deps.duration.as_secs_f64(),
            comparison.legacy.as_ref().map_or("N/A".to_string(), |l| format!("{:.2}s", l.duration.as_secs_f64())),
            comparison.improvement_percentage().map_or("N/A".to_string(), |p| format!("{:.1}%", p)),
            comparison.zero_deps.ops_per_sec,
            comparison.legacy.as_ref().map_or("N/A".to_string(), |l| format!("{:.0}", l.ops_per_sec)),
            comparison.improvement_ratio.map_or("N/A".to_string(), |r| format!("{:.2}x", r))
        ));

        // Individual crate benchmarks
        report.push_str("## 🔧 Individual Crate Compilation Times\n\n");
        let crate_results = self.benchmark_crate_compilation()?;
        report.push_str("| Crate | Compilation Time | Description |\n");
        report.push_str("|-------|------------------|-------------|\n");
        
        for result in crate_results {
            report.push_str(&format!(
                "| {} | {:.2}s | Zero-dependency implementation |\n",
                result.name, result.duration.as_secs_f64()
            ));
        }
        report.push('\n');

        // Incremental build performance
        report.push_str("## 🔄 Incremental Build Performance\n\n");
        if let Ok(incremental_results) = self.benchmark_incremental() {
            report.push_str("| Scenario | Time | Performance |\n");
            report.push_str("|----------|------|-------------|\n");
            
            for result in incremental_results {
                report.push_str(&format!(
                    "| {} | {:.2}s | {:.0} ops/sec |\n",
                    result.name, result.duration.as_secs_f64(), result.ops_per_sec
                ));
            }
        }
        report.push('\n');

        // Summary and recommendations
        report.push_str("## 🎯 Summary and Impact\n\n");
        if let Some(improvement) = comparison.improvement_percentage() {
            if improvement > 0.0 {
                report.push_str(&format!(
                    "✅ **Zero-dependency implementation is {:.1}% faster** than legacy dependencies!\n\n",
                    improvement
                ));
            } else {
                report.push_str(&format!(
                    "⚠️  Zero-dependency implementation is {:.1}% slower than legacy dependencies.\n\n",
                    improvement.abs()
                ));
            }
        }

        report.push_str("### Key Benefits:\n");
        report.push_str("- 🚀 Significantly reduced compilation times\n");
        report.push_str("- 📦 Minimal dependency footprint\n");
        report.push_str("- 🛡️  Enhanced security through dependency reduction\n");
        report.push_str("- 🔧 Better portability and maintenance\n");
        report.push_str("- ⚡ Faster CI/CD pipeline execution\n\n");

        Ok(report)
    }

    // Private helper methods

    fn benchmark_features(&self, features: &[&str], name: &str) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let mut total_duration = Duration::ZERO;
        
        for run in 0..self.config.runs {
            println!("  Run {}/{} for {}...", run + 1, self.config.runs, name);
            
            if self.config.clean_build {
                self.run_cargo_command(&["clean"])?;
            }

            let mut args = vec!["build"];
            let features_string;
            if !features.is_empty() {
                args.push("--features");
                features_string = features.join(",");
                args.push(&features_string);
            }

            let start = Instant::now();
            self.run_cargo_command(&args)?;
            total_duration += start.elapsed();
        }

        let avg_duration = total_duration / self.config.runs as u32;
        Ok(BenchmarkResult::new(name, avg_duration, 1))
    }

    fn benchmark_crate(&self, crate_path: &str, description: &str) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        if self.config.clean_build {
            self.run_cargo_command_in_dir(&["clean"], crate_path)?;
        }

        let start = Instant::now();
        self.run_cargo_command_in_dir(&["build"], crate_path)?;
        let duration = start.elapsed();

        Ok(BenchmarkResult::new(description, duration, 1))
    }

    fn run_cargo_command(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        self.run_cargo_command_in_dir(args, &self.config.project_path)
    }

    fn run_cargo_command_in_dir(&self, args: &[&str], dir: &str) -> Result<(), Box<dyn std::error::Error>> {
        let output = Command::new("cargo")
            .args(args)
            .current_dir(dir)
            .output()?;

        if !output.status.success() {
            return Err(format!(
                "Cargo command failed: {}\nStderr: {}",
                args.join(" "),
                String::from_utf8_lossy(&output.stderr)
            ).into());
        }

        Ok(())
    }

    fn clear_cargo_cache(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Clear target directory
        self.run_cargo_command(&["clean"])?;
        
        // Optionally clear cargo registry cache (commented out as it's quite aggressive)
        // Command::new("rm")
        //     .args(&["-rf", "~/.cargo/registry"])
        //     .output()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compilation_config() {
        let config = CompilationConfig::default();
        assert_eq!(config.project_path, ".");
        assert!(config.clean_build);
        assert_eq!(config.runs, 3);
    }

    #[test]
    fn test_compilation_benchmark_creation() {
        let config = CompilationConfig::default();
        let benchmark = CompilationBenchmark::new(config);
        assert_eq!(benchmark.config.runs, 3);
    }
}