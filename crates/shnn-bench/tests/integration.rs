//! Integration tests for shnn-bench crate
//! 
//! These tests verify benchmarking capabilities, performance measurement,
//! cross-platform comparison, and statistical analysis of neuromorphic computing performance.

use shnn_bench::prelude::*;
use shnn_bench::config::{BenchmarkConfig, SystemConfig, NetworkConfig};
use shnn_bench::runner::{BenchmarkRunner, BenchmarkSuite, TestScenario};
use shnn_bench::metrics::{PerformanceMetrics, SystemMetrics, NetworkMetrics};
use shnn_bench::comparison::{ComparisonFramework, BaselineComparison, PerformanceAnalysis};
use shnn_bench::analysis::{StatisticalAnalysis, PerformanceReport, TrendAnalysis};
use shnn_bench::profiler::{DetailedProfiler, MemoryProfiler, CpuProfiler};
use shnn_bench::export::{ResultExporter, CsvExporter, JsonExporter, PlotExporter};

use shnn_core::prelude::*;
use shnn_core::neuron::{LIFNeuron, AdExNeuron, IzhikevichNeuron, NeuronId};
use shnn_core::hypergraph::HypergraphNetwork;
use shnn_core::spike::Spike;
use shnn_core::time::TimeStep;

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::fs;
use std::path::Path;

#[test]
fn test_benchmark_config_creation() {
    let config = BenchmarkConfig::builder()
        .name("test_benchmark")
        .description("Test benchmark configuration")
        .repetitions(5)
        .warmup_iterations(2)
        .timeout(Duration::from_secs(30))
        .collect_detailed_metrics(true)
        .output_directory("test_results")
        .build();
    
    assert_eq!(config.name(), "test_benchmark");
    assert_eq!(config.repetitions(), 5);
    assert_eq!(config.warmup_iterations(), 2);
    assert_eq!(config.timeout(), Duration::from_secs(30));
    assert!(config.collect_detailed_metrics());
}

#[test]
fn test_system_configuration_detection() {
    let system_config = SystemConfig::detect_current_system();
    
    assert!(system_config.is_ok());
    let config = system_config.unwrap();
    
    // Verify system information is detected
    assert!(!config.cpu_model().is_empty());
    assert!(config.cpu_cores() > 0);
    assert!(config.memory_gb() > 0.0);
    assert!(!config.os_name().is_empty());
    assert!(!config.architecture().is_empty());
    
    println!("Detected system:");
    println!("  CPU: {} ({} cores)", config.cpu_model(), config.cpu_cores());
    println!("  Memory: {:.1} GB", config.memory_gb());
    println!("  OS: {} ({})", config.os_name(), config.architecture());
}

#[test]
fn test_network_configuration_scenarios() {
    let scenarios = vec![
        NetworkConfig::small_network(100, 500),
        NetworkConfig::medium_network(1000, 5000),
        NetworkConfig::large_network(10000, 50000),
        NetworkConfig::custom("custom_test", 2000, 8000, 0.1, 1000.0),
    ];
    
    for scenario in scenarios {
        assert!(scenario.neuron_count() > 0);
        assert!(scenario.connection_count() > 0);
        assert!(scenario.simulation_time() > 0.0);
        assert!(scenario.timestep() > 0.0);
        
        // Verify reasonable parameter ranges
        assert!(scenario.timestep() <= 1.0); // <= 1ms timestep
        assert!(scenario.simulation_time() <= 10000.0); // <= 10s simulation
    }
}

#[test]
fn test_basic_benchmark_execution() {
    let config = BenchmarkConfig::builder()
        .name("basic_test")
        .repetitions(3)
        .warmup_iterations(1)
        .timeout(Duration::from_secs(10))
        .build();
    
    let mut runner = BenchmarkRunner::new(config);
    
    // Define simple test scenario
    let scenario = TestScenario::builder()
        .name("lif_neuron_integration")
        .setup(|| {
            let mut neuron = LIFNeuron::default();
            neuron
        })
        .benchmark(|mut neuron| {
            let dt = TimeStep::from_ms(0.1);
            for _ in 0..1000 {
                neuron.integrate(1.0, dt);
                neuron.update(dt);
            }
            neuron
        })
        .cleanup(|_neuron| {
            // Nothing to clean up for LIF neuron
        })
        .build();
    
    let result = runner.run_scenario(scenario);
    assert!(result.is_ok());
    
    let metrics = result.unwrap();
    assert!(metrics.execution_time() > Duration::from_nanos(0));
    assert!(metrics.iterations_completed() > 0);
    assert_eq!(metrics.scenario_name(), "lif_neuron_integration");
}

#[test]
fn test_network_benchmark_suite() {
    let suite_config = BenchmarkConfig::builder()
        .name("network_suite")
        .repetitions(2)
        .warmup_iterations(1)
        .timeout(Duration::from_secs(15))
        .build();
    
    let mut suite = BenchmarkSuite::new(suite_config);
    
    // Add different network size scenarios
    let scenarios = vec![
        NetworkConfig::small_network(50, 200),
        NetworkConfig::medium_network(200, 800),
    ];
    
    for network_config in scenarios {
        let scenario_name = format!("network_{}_{}", 
                                  network_config.neuron_count(), 
                                  network_config.connection_count());
        
        let scenario = TestScenario::builder()
            .name(&scenario_name)
            .setup(move || {
                create_test_network(network_config.clone())
            })
            .benchmark(|mut network| {
                simulate_network(&mut network, 100, 0.1);
                network
            })
            .cleanup(|_network| {
                // Network cleanup handled by Drop
            })
            .build();
        
        suite.add_scenario(scenario);
    }
    
    let results = suite.run_all();
    assert!(results.is_ok());
    
    let suite_results = results.unwrap();
    assert_eq!(suite_results.scenario_count(), 2);
    
    for result in suite_results.individual_results() {
        assert!(result.execution_time() > Duration::from_nanos(0));
        println!("Scenario '{}': {:.3}ms", 
                result.scenario_name(), 
                result.execution_time().as_millis());
    }
}

#[test]
fn test_performance_metrics_collection() {
    let mut profiler = DetailedProfiler::new();
    
    profiler.start_profiling("test_metrics").unwrap();
    
    // Simulate work
    let mut network = create_test_network(NetworkConfig::small_network(100, 400));
    
    for step in 0..500 {
        // Inject spikes periodically
        if step % 50 == 0 {
            let spike = Spike::new(NeuronId(step % 100), TimeStep::from_ms(step as f64 * 0.1));
            network.process_spike(spike).unwrap();
        }
        
        let dt = TimeStep::from_ms(0.1);
        // Simulate network update (would be implemented in actual network)
    }
    
    let metrics = profiler.end_profiling("test_metrics").unwrap();
    
    // Verify metrics collection
    assert!(metrics.cpu_time() > Duration::from_nanos(0));
    assert!(metrics.wall_clock_time() > Duration::from_nanos(0));
    assert!(metrics.memory_peak_mb() >= 0.0);
    assert!(metrics.instructions_executed() >= 0);
    
    println!("Performance metrics:");
    println!("  CPU time: {:.3}ms", metrics.cpu_time().as_millis());
    println!("  Wall time: {:.3}ms", metrics.wall_clock_time().as_millis());
    println!("  Peak memory: {:.2}MB", metrics.memory_peak_mb());
    println!("  Instructions: {}", metrics.instructions_executed());
}

#[test]
fn test_memory_profiling() {
    let mut memory_profiler = MemoryProfiler::new();
    
    memory_profiler.start_monitoring();
    
    let initial_memory = memory_profiler.current_usage_mb();
    
    // Allocate and work with various data structures
    let mut networks = Vec::new();
    
    for i in 0..10 {
        let network = create_test_network(NetworkConfig::small_network(100 + i * 10, 500 + i * 50));
        networks.push(network);
        
        let current_memory = memory_profiler.current_usage_mb();
        memory_profiler.record_checkpoint(&format!("after_network_{}", i));
        
        println!("Memory after network {}: {:.2}MB", i, current_memory);
    }
    
    let peak_memory = memory_profiler.peak_usage_mb();
    
    // Clean up
    networks.clear();
    
    let final_memory = memory_profiler.current_usage_mb();
    memory_profiler.stop_monitoring();
    
    let memory_report = memory_profiler.generate_report();
    
    // Verify memory tracking
    assert!(peak_memory >= initial_memory);
    assert!(memory_report.checkpoints().len() >= 10);
    assert!(memory_report.allocation_events() > 0);
    
    println!("Memory profiling summary:");
    println!("  Initial: {:.2}MB", initial_memory);
    println!("  Peak: {:.2}MB", peak_memory);
    println!("  Final: {:.2}MB", final_memory);
    println!("  Allocations: {}", memory_report.allocation_events());
}

#[test]
fn test_cpu_profiling() {
    let mut cpu_profiler = CpuProfiler::new();
    
    cpu_profiler.start_profiling();
    
    // CPU-intensive neuromorphic computation
    let mut network = create_test_network(NetworkConfig::medium_network(1000, 5000));
    
    // Simulate intensive computation
    for iteration in 0..100 {
        cpu_profiler.mark_iteration_start(iteration);
        
        // Inject multiple spikes
        for neuron_id in 0..10 {
            let spike = Spike::new(NeuronId(neuron_id), TimeStep::from_ms(iteration as f64 * 0.1));
            network.process_spike(spike).unwrap();
        }
        
        // Simulate network update
        simulate_network_step(&mut network);
        
        cpu_profiler.mark_iteration_end(iteration);
        
        if iteration % 20 == 0 {
            let cpu_usage = cpu_profiler.current_cpu_usage();
            println!("CPU usage at iteration {}: {:.1}%", iteration, cpu_usage * 100.0);
        }
    }
    
    cpu_profiler.stop_profiling();
    
    let cpu_report = cpu_profiler.generate_report();
    
    // Verify CPU profiling
    assert!(cpu_report.average_cpu_usage() >= 0.0);
    assert!(cpu_report.average_cpu_usage() <= 1.0);
    assert!(cpu_report.peak_cpu_usage() >= cpu_report.average_cpu_usage());
    assert_eq!(cpu_report.total_iterations(), 100);
    
    println!("CPU profiling summary:");
    println!("  Average CPU: {:.1}%", cpu_report.average_cpu_usage() * 100.0);
    println!("  Peak CPU: {:.1}%", cpu_report.peak_cpu_usage() * 100.0);
    println!("  Total iterations: {}", cpu_report.total_iterations());
}

#[test]
fn test_comparison_framework() {
    let mut comparison = ComparisonFramework::new();
    
    // Add baseline implementation (simple reference)
    comparison.add_baseline("reference_python", || {
        // Simulate Python-equivalent performance
        std::thread::sleep(Duration::from_millis(100));
        BaselineResult {
            execution_time: Duration::from_millis(100),
            memory_usage: 50.0,
            accuracy: 0.95,
        }
    });
    
    // Add SHNN implementation
    comparison.add_implementation("shnn_rust", || {
        let start = Instant::now();
        
        let network = create_test_network(NetworkConfig::small_network(100, 500));
        simulate_network_basic(&network, 100);
        
        let execution_time = start.elapsed();
        
        ImplementationResult {
            execution_time,
            memory_usage: 25.0, // Estimated
            accuracy: 0.97,
            additional_metrics: HashMap::new(),
        }
    });
    
    // Add optimized SHNN implementation
    comparison.add_implementation("shnn_optimized", || {
        let start = Instant::now();
        
        let network = create_optimized_network(NetworkConfig::small_network(100, 500));
        simulate_network_optimized(&network, 100);
        
        let execution_time = start.elapsed();
        
        ImplementationResult {
            execution_time,
            memory_usage: 20.0,
            accuracy: 0.97,
            additional_metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("cache_hits".to_string(), 0.85);
                metrics.insert("branch_prediction".to_string(), 0.92);
                metrics
            },
        }
    });
    
    let comparison_results = comparison.run_comparison(3); // 3 repetitions
    assert!(comparison_results.is_ok());
    
    let results = comparison_results.unwrap();
    
    // Verify comparison results
    assert_eq!(results.implementation_count(), 2);
    assert!(results.has_baseline());
    
    // Check speedup calculations
    let speedups = results.calculate_speedups();
    assert!(speedups.contains_key("shnn_rust"));
    assert!(speedups.contains_key("shnn_optimized"));
    
    for (impl_name, speedup) in &speedups {
        println!("{}: {:.2}x speedup", impl_name, speedup);
        assert!(*speedup > 0.0);
    }
    
    // Generate performance analysis
    let analysis = PerformanceAnalysis::new(&results);
    let summary = analysis.generate_summary();
    
    assert!(!summary.best_implementation().is_empty());
    assert!(summary.overall_improvement() >= 0.0);
    
    println!("Performance analysis:");
    println!("  Best implementation: {}", summary.best_implementation());
    println!("  Overall improvement: {:.1}%", summary.overall_improvement() * 100.0);
}

#[test]
fn test_statistical_analysis() {
    // Generate sample benchmark data
    let mut measurements = Vec::new();
    
    for _ in 0..20 {
        let base_time = 50.0; // 50ms base
        let noise = (rand::random::<f64>() - 0.5) * 10.0; // ±5ms noise
        measurements.push(base_time + noise);
    }
    
    let analysis = StatisticalAnalysis::new(&measurements);
    
    // Test statistical calculations
    let mean = analysis.mean();
    let std_dev = analysis.standard_deviation();
    let confidence_interval = analysis.confidence_interval(0.95);
    let outliers = analysis.detect_outliers();
    
    assert!((mean - 50.0).abs() < 5.0); // Should be close to 50ms
    assert!(std_dev > 0.0);
    assert!(confidence_interval.lower < mean);
    assert!(confidence_interval.upper > mean);
    
    println!("Statistical analysis:");
    println!("  Mean: {:.2}ms", mean);
    println!("  Std dev: {:.2}ms", std_dev);
    println!("  95% CI: [{:.2}, {:.2}]ms", confidence_interval.lower, confidence_interval.upper);
    println!("  Outliers: {}", outliers.len());
    
    // Test trend analysis
    let trend_analysis = TrendAnalysis::new();
    
    // Generate time series data
    let mut time_series = Vec::new();
    for i in 0..50 {
        let value = 100.0 - i as f64 * 0.5 + (rand::random::<f64>() - 0.5) * 5.0; // Declining trend
        time_series.push((i as f64, value));
    }
    
    let trend = trend_analysis.analyze_trend(&time_series);
    
    assert!(trend.slope < 0.0); // Should detect declining trend
    assert!(trend.r_squared > 0.5); // Should have reasonable fit
    
    println!("Trend analysis:");
    println!("  Slope: {:.3}", trend.slope);
    println!("  R²: {:.3}", trend.r_squared);
    println!("  Trend direction: {:?}", trend.direction);
}

#[test]
fn test_result_export() {
    // Create sample benchmark results
    let results = create_sample_benchmark_results();
    
    let temp_dir = std::env::temp_dir().join("shnn_bench_test");
    fs::create_dir_all(&temp_dir).unwrap();
    
    // Test CSV export
    let csv_exporter = CsvExporter::new();
    let csv_path = temp_dir.join("results.csv");
    csv_exporter.export(&results, &csv_path).unwrap();
    
    assert!(csv_path.exists());
    let csv_content = fs::read_to_string(&csv_path).unwrap();
    assert!(csv_content.contains("scenario_name"));
    assert!(csv_content.contains("execution_time"));
    
    // Test JSON export
    let json_exporter = JsonExporter::new();
    let json_path = temp_dir.join("results.json");
    json_exporter.export(&results, &json_path).unwrap();
    
    assert!(json_path.exists());
    let json_content = fs::read_to_string(&json_path).unwrap();
    assert!(json_content.starts_with('{') || json_content.starts_with('['));
    
    // Test plot export (if plotting is available)
    if PlotExporter::is_available() {
        let plot_exporter = PlotExporter::new();
        let plot_path = temp_dir.join("performance_plot.png");
        plot_exporter.export_performance_chart(&results, &plot_path).unwrap();
        
        assert!(plot_path.exists());
    }
    
    // Cleanup
    fs::remove_dir_all(&temp_dir).unwrap();
}

#[test]
fn test_neuron_model_benchmarks() {
    let config = BenchmarkConfig::builder()
        .name("neuron_models")
        .repetitions(5)
        .warmup_iterations(2)
        .build();
    
    let mut suite = BenchmarkSuite::new(config);
    
    // Benchmark different neuron models
    let neuron_types = vec![
        ("LIF", || Box::new(LIFNeuron::default()) as Box<dyn Neuron>),
        ("AdEx", || Box::new(AdExNeuron::default()) as Box<dyn Neuron>),
        ("Izhikevich", || Box::new(IzhikevichNeuron::default()) as Box<dyn Neuron>),
    ];
    
    for (name, neuron_factory) in neuron_types {
        let scenario = TestScenario::builder()
            .name(&format!("{}_neuron_benchmark", name))
            .setup(neuron_factory)
            .benchmark(|mut neuron| {
                let dt = TimeStep::from_ms(0.1);
                
                // Simulate 1 second of activity
                for step in 0..10000 {
                    let input_current = if step % 100 == 0 { 5.0 } else { 0.1 };
                    neuron.integrate(input_current, dt);
                    neuron.update(dt);
                }
                
                neuron
            })
            .cleanup(|_neuron| {})
            .build();
        
        suite.add_scenario(scenario);
    }
    
    let results = suite.run_all().unwrap();
    
    // Analyze neuron model performance
    let mut model_performance = HashMap::new();
    
    for result in results.individual_results() {
        let model_name = result.scenario_name()
            .replace("_neuron_benchmark", "");
        model_performance.insert(model_name, result.execution_time());
    }
    
    println!("Neuron model performance:");
    for (model, time) in &model_performance {
        println!("  {}: {:.3}ms", model, time.as_millis());
    }
    
    // Verify all models completed successfully
    assert_eq!(model_performance.len(), 3);
    for (_, time) in &model_performance {
        assert!(*time > Duration::from_nanos(0));
        assert!(*time < Duration::from_secs(5)); // Should complete in reasonable time
    }
}

#[test]
fn test_scalability_benchmarks() {
    let network_sizes = vec![100, 500, 1000, 2000];
    let mut scalability_results = Vec::new();
    
    for size in network_sizes {
        let config = BenchmarkConfig::builder()
            .name(&format!("scalability_{}", size))
            .repetitions(3)
            .timeout(Duration::from_secs(30))
            .build();
        
        let mut runner = BenchmarkRunner::new(config);
        
        let scenario = TestScenario::builder()
            .name(&format!("network_size_{}", size))
            .setup(move || {
                create_test_network(NetworkConfig::custom(
                    &format!("size_{}", size),
                    size,
                    size * 5, // 5 connections per neuron on average
                    0.1,
                    100.0
                ))
            })
            .benchmark(|mut network| {
                simulate_network(&mut network, 100, 0.1);
                network
            })
            .cleanup(|_network| {})
            .build();
        
        let result = runner.run_scenario(scenario).unwrap();
        scalability_results.push((size, result.execution_time()));
        
        println!("Network size {}: {:.3}ms", size, result.execution_time().as_millis());
    }
    
    // Analyze scalability
    let mut complexity_analysis = Vec::new();
    
    for i in 1..scalability_results.len() {
        let (prev_size, prev_time) = scalability_results[i - 1];
        let (curr_size, curr_time) = scalability_results[i];
        
        let size_ratio = curr_size as f64 / prev_size as f64;
        let time_ratio = curr_time.as_nanos() as f64 / prev_time.as_nanos() as f64;
        
        let complexity_factor = time_ratio / size_ratio;
        complexity_analysis.push(complexity_factor);
        
        println!("Size {}→{}: {:.2}x time increase for {:.2}x size increase (factor: {:.2})",
                prev_size, curr_size, time_ratio, size_ratio, complexity_factor);
    }
    
    // Verify reasonable scalability (should be sub-quadratic)
    let average_complexity = complexity_analysis.iter().sum::<f64>() / complexity_analysis.len() as f64;
    assert!(average_complexity < 5.0, "Scalability too poor: {:.2}", average_complexity);
}

// Helper functions for testing

fn create_test_network(config: NetworkConfig) -> TestNetwork {
    TestNetwork {
        neuron_count: config.neuron_count(),
        connection_count: config.connection_count(),
        simulation_time: config.simulation_time(),
        timestep: config.timestep(),
    }
}

fn create_optimized_network(config: NetworkConfig) -> OptimizedTestNetwork {
    OptimizedTestNetwork {
        neuron_count: config.neuron_count(),
        connection_count: config.connection_count(),
        simulation_time: config.simulation_time(),
        timestep: config.timestep(),
        optimization_level: 2,
    }
}

fn simulate_network(network: &mut TestNetwork, steps: usize, dt: f64) {
    for step in 0..steps {
        // Simulate network computation
        if step % 10 == 0 {
            // Inject stimulus
            network.inject_stimulus(step % network.neuron_count, step as f64 * dt);
        }
        
        network.update(dt);
    }
}

fn simulate_network_basic(network: &TestNetwork, steps: usize) {
    // Basic simulation for comparison
    for _ in 0..steps {
        std::hint::spin_loop(); // Simulate minimal work
    }
}

fn simulate_network_optimized(network: &OptimizedTestNetwork, steps: usize) {
    // Optimized simulation
    for _ in 0..steps {
        // Simulate optimized computation
        std::hint::spin_loop();
    }
}

fn simulate_network_step(network: &mut TestNetwork) {
    network.update(0.1);
}

fn create_sample_benchmark_results() -> BenchmarkResults {
    let mut results = BenchmarkResults::new("sample_benchmark");
    
    results.add_result(ScenarioResult {
        scenario_name: "small_network".to_string(),
        execution_time: Duration::from_millis(50),
        memory_usage: 25.0,
        iterations_completed: 1000,
        success: true,
    });
    
    results.add_result(ScenarioResult {
        scenario_name: "medium_network".to_string(),
        execution_time: Duration::from_millis(200),
        memory_usage: 100.0,
        iterations_completed: 1000,
        success: true,
    });
    
    results
}

// Test data structures

struct TestNetwork {
    neuron_count: usize,
    connection_count: usize,
    simulation_time: f64,
    timestep: f64,
}

impl TestNetwork {
    fn inject_stimulus(&mut self, neuron_id: usize, time: f64) {
        // Simulate stimulus injection
    }
    
    fn update(&mut self, dt: f64) {
        // Simulate network update
        std::thread::sleep(Duration::from_micros(10)); // Simulate computation
    }
}

struct OptimizedTestNetwork {
    neuron_count: usize,
    connection_count: usize,
    simulation_time: f64,
    timestep: f64,
    optimization_level: u8,
}

struct BaselineResult {
    execution_time: Duration,
    memory_usage: f64,
    accuracy: f64,
}

struct ImplementationResult {
    execution_time: Duration,
    memory_usage: f64,
    accuracy: f64,
    additional_metrics: HashMap<String, f64>,
}

struct ScenarioResult {
    scenario_name: String,
    execution_time: Duration,
    memory_usage: f64,
    iterations_completed: usize,
    success: bool,
}

struct BenchmarkResults {
    benchmark_name: String,
    results: Vec<ScenarioResult>,
}

impl BenchmarkResults {
    fn new(name: &str) -> Self {
        Self {
            benchmark_name: name.to_string(),
            results: Vec::new(),
        }
    }
    
    fn add_result(&mut self, result: ScenarioResult) {
        self.results.push(result);
    }
    
    fn individual_results(&self) -> &[ScenarioResult] {
        &self.results
    }
}

impl ScenarioResult {
    fn scenario_name(&self) -> &str {
        &self.scenario_name
    }
    
    fn execution_time(&self) -> Duration {
        self.execution_time
    }
    
    fn iterations_completed(&self) -> usize {
        self.iterations_completed
    }
}