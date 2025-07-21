//! Performance profiling and monitoring for hardware acceleration
//!
//! This module provides comprehensive profiling capabilities for analyzing
//! the performance of neural network operations across different hardware platforms.

use crate::{
    error::{FFIError, FFIResult},
    types::{AcceleratorType, PerformanceMetrics, TimingBreakdown},
    AcceleratorId, NetworkId,
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[cfg(feature = "profiling")]
use tracing::{info, debug, warn, error};

/// Performance profiler for tracking hardware acceleration metrics
#[derive(Debug)]
pub struct PerformanceProfiler {
    /// Active profiling sessions
    sessions: Arc<Mutex<HashMap<ProfilerSessionId, ProfilerSession>>>,
    /// Next session ID
    next_session_id: Arc<Mutex<u64>>,
    /// Global metrics aggregator
    global_metrics: Arc<Mutex<GlobalMetrics>>,
    /// Profiling configuration
    config: ProfilerConfig,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            next_session_id: Arc::new(Mutex::new(1)),
            global_metrics: Arc::new(Mutex::new(GlobalMetrics::new())),
            config,
        }
    }
    
    /// Start a new profiling session
    pub fn start_session(
        &self,
        accelerator_id: AcceleratorId,
        accelerator_type: AcceleratorType,
        network_id: Option<NetworkId>,
    ) -> FFIResult<ProfilerSessionId> {
        let mut next_id = self.next_session_id.lock().unwrap();
        let session_id = ProfilerSessionId(*next_id);
        *next_id += 1;
        
        let session = ProfilerSession::new(
            session_id,
            accelerator_id,
            accelerator_type,
            network_id,
            self.config.clone(),
        );
        
        let mut sessions = self.sessions.lock().unwrap();
        sessions.insert(session_id, session);
        
        #[cfg(feature = "profiling")]
        info!("Started profiling session {:?} for accelerator {:?}", session_id, accelerator_id);
        
        Ok(session_id)
    }
    
    /// End a profiling session and get results
    pub fn end_session(&self, session_id: ProfilerSessionId) -> FFIResult<ProfilerResults> {
        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions.remove(&session_id)
            .ok_or_else(|| FFIError::invalid_config("Invalid profiler session ID"))?;
        
        let results = session.finalize()?;
        
        // Update global metrics
        if let Ok(mut global_metrics) = self.global_metrics.lock() {
            global_metrics.update_from_session(&results);
        }
        
        #[cfg(feature = "profiling")]
        info!("Ended profiling session {:?}, total duration: {:.2}ms", 
              session_id, results.total_duration.as_secs_f64() * 1000.0);
        
        Ok(results)
    }
    
    /// Record a timing event in a session
    pub fn record_timing(
        &self,
        session_id: ProfilerSessionId,
        event_type: TimingEventType,
        duration: Duration,
    ) -> FFIResult<()> {
        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions.get_mut(&session_id)
            .ok_or_else(|| FFIError::invalid_config("Invalid profiler session ID"))?;
        
        session.record_timing(event_type, duration);
        Ok(())
    }
    
    /// Record a performance metric in a session
    pub fn record_metric(
        &self,
        session_id: ProfilerSessionId,
        metric_type: MetricType,
        value: f64,
    ) -> FFIResult<()> {
        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions.get_mut(&session_id)
            .ok_or_else(|| FFIError::invalid_config("Invalid profiler session ID"))?;
        
        session.record_metric(metric_type, value);
        Ok(())
    }
    
    /// Get current session status
    pub fn get_session_status(&self, session_id: ProfilerSessionId) -> FFIResult<SessionStatus> {
        let sessions = self.sessions.lock().unwrap();
        let session = sessions.get(&session_id)
            .ok_or_else(|| FFIError::invalid_config("Invalid profiler session ID"))?;
        
        Ok(session.get_status())
    }
    
    /// Get global performance statistics
    pub fn get_global_statistics(&self) -> GlobalStatistics {
        self.global_metrics.lock()
            .map(|metrics| metrics.get_statistics())
            .unwrap_or_default()
    }
    
    /// Generate performance report
    pub fn generate_report(&self, session_id: ProfilerSessionId) -> FFIResult<PerformanceReport> {
        let sessions = self.sessions.lock().unwrap();
        let session = sessions.get(&session_id)
            .ok_or_else(|| FFIError::invalid_config("Invalid profiler session ID"))?;
        
        session.generate_report()
    }
    
    /// Export profiling data to file
    pub fn export_data(
        &self,
        session_id: ProfilerSessionId,
        format: ExportFormat,
        filename: &str,
    ) -> FFIResult<()> {
        let sessions = self.sessions.lock().unwrap();
        let session = sessions.get(&session_id)
            .ok_or_else(|| FFIError::invalid_config("Invalid profiler session ID"))?;
        
        session.export_data(format, filename)
    }
}

/// Unique identifier for profiler sessions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProfilerSessionId(pub u64);

/// Individual profiling session
#[derive(Debug)]
pub struct ProfilerSession {
    /// Session ID
    id: ProfilerSessionId,
    /// Associated accelerator
    accelerator_id: AcceleratorId,
    /// Accelerator type
    accelerator_type: AcceleratorType,
    /// Associated network (if any)
    network_id: Option<NetworkId>,
    /// Session start time
    start_time: Instant,
    /// Timing events
    timing_events: Vec<TimingEvent>,
    /// Performance metrics
    metrics: HashMap<MetricType, Vec<MetricSample>>,
    /// Memory usage tracking
    memory_samples: Vec<MemorySample>,
    /// Configuration
    config: ProfilerConfig,
    /// Session state
    state: SessionState,
}

impl ProfilerSession {
    /// Create a new profiling session
    pub fn new(
        id: ProfilerSessionId,
        accelerator_id: AcceleratorId,
        accelerator_type: AcceleratorType,
        network_id: Option<NetworkId>,
        config: ProfilerConfig,
    ) -> Self {
        Self {
            id,
            accelerator_id,
            accelerator_type,
            network_id,
            start_time: Instant::now(),
            timing_events: Vec::new(),
            metrics: HashMap::new(),
            memory_samples: Vec::new(),
            config,
            state: SessionState::Active,
        }
    }
    
    /// Record a timing event
    pub fn record_timing(&mut self, event_type: TimingEventType, duration: Duration) {
        if !self.config.collect_timing {
            return;
        }
        
        let event = TimingEvent {
            event_type,
            duration,
            timestamp: self.start_time.elapsed(),
        };
        
        self.timing_events.push(event);
        
        #[cfg(feature = "profiling")]
        debug!("Recorded timing event {:?}: {:.3}ms", event_type, duration.as_secs_f64() * 1000.0);
    }
    
    /// Record a performance metric
    pub fn record_metric(&mut self, metric_type: MetricType, value: f64) {
        if !self.config.collect_metrics {
            return;
        }
        
        let sample = MetricSample {
            value,
            timestamp: self.start_time.elapsed(),
        };
        
        self.metrics.entry(metric_type)
            .or_insert_with(Vec::new)
            .push(sample);
        
        #[cfg(feature = "profiling")]
        debug!("Recorded metric {:?}: {:.3}", metric_type, value);
    }
    
    /// Record memory usage sample
    pub fn record_memory_usage(&mut self, usage_bytes: u64) {
        if !self.config.collect_memory {
            return;
        }
        
        let sample = MemorySample {
            usage_bytes,
            timestamp: self.start_time.elapsed(),
        };
        
        self.memory_samples.push(sample);
    }
    
    /// Get current session status
    pub fn get_status(&self) -> SessionStatus {
        SessionStatus {
            id: self.id,
            accelerator_id: self.accelerator_id,
            accelerator_type: self.accelerator_type,
            network_id: self.network_id,
            state: self.state,
            duration: self.start_time.elapsed(),
            event_count: self.timing_events.len(),
            metric_count: self.metrics.values().map(|v| v.len()).sum(),
            memory_sample_count: self.memory_samples.len(),
        }
    }
    
    /// Finalize the session and produce results
    pub fn finalize(mut self) -> FFIResult<ProfilerResults> {
        self.state = SessionState::Finalized;
        
        let total_duration = self.start_time.elapsed();
        let timing_breakdown = self.calculate_timing_breakdown();
        let performance_metrics = self.calculate_performance_metrics()?;
        
        Ok(ProfilerResults {
            session_id: self.id,
            accelerator_id: self.accelerator_id,
            accelerator_type: self.accelerator_type,
            network_id: self.network_id,
            total_duration,
            timing_breakdown,
            performance_metrics,
            timing_events: self.timing_events,
            metric_samples: self.metrics,
            memory_samples: self.memory_samples,
        })
    }
    
    /// Calculate timing breakdown
    fn calculate_timing_breakdown(&self) -> TimingBreakdown {
        let mut breakdown = TimingBreakdown::default();
        
        for event in &self.timing_events {
            let duration_ms = event.duration.as_secs_f64() * 1000.0;
            
            match event.event_type {
                TimingEventType::Initialization => breakdown.initialization_ms += duration_ms,
                TimingEventType::SpikeProcessing => breakdown.processing_ms += duration_ms,
                TimingEventType::MemoryTransfer => breakdown.memory_transfer_ms += duration_ms,
                TimingEventType::PlasticityUpdate => breakdown.plasticity_update_ms += duration_ms,
                TimingEventType::Synchronization => breakdown.synchronization_ms += duration_ms,
                _ => {} // Other event types not included in standard breakdown
            }
        }
        
        breakdown
    }
    
    /// Calculate aggregate performance metrics
    fn calculate_performance_metrics(&self) -> FFIResult<PerformanceMetrics> {
        let mut metrics = PerformanceMetrics::default();
        
        metrics.execution_time_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;
        
        // Calculate averages for various metrics
        if let Some(samples) = self.metrics.get(&MetricType::SpikesPerSecond) {
            if !samples.is_empty() {
                metrics.spikes_per_second = samples.iter().map(|s| s.value).sum::<f64>() / samples.len() as f64;
            }
        }
        
        if let Some(samples) = self.metrics.get(&MetricType::PowerConsumption) {
            if !samples.is_empty() {
                metrics.power_consumption = (samples.iter().map(|s| s.value).sum::<f64>() / samples.len() as f64) as f32;
            }
        }
        
        if let Some(samples) = self.metrics.get(&MetricType::GpuUtilization) {
            if !samples.is_empty() {
                metrics.gpu_utilization = (samples.iter().map(|s| s.value).sum::<f64>() / samples.len() as f64) as f32;
            }
        }
        
        if let Some(samples) = self.metrics.get(&MetricType::MemoryUtilization) {
            if !samples.is_empty() {
                metrics.memory_utilization = (samples.iter().map(|s| s.value).sum::<f64>() / samples.len() as f64) as f32;
            }
        }
        
        // Memory usage from memory samples
        if !self.memory_samples.is_empty() {
            metrics.memory_usage = self.memory_samples.iter()
                .map(|s| s.usage_bytes)
                .max()
                .unwrap_or(0);
        }
        
        metrics.timing_breakdown = self.calculate_timing_breakdown();
        
        Ok(metrics)
    }
    
    /// Generate a comprehensive performance report
    pub fn generate_report(&self) -> FFIResult<PerformanceReport> {
        let timing_summary = self.generate_timing_summary();
        let metric_summary = self.generate_metric_summary();
        let memory_summary = self.generate_memory_summary();
        let recommendations = self.generate_recommendations();
        
        Ok(PerformanceReport {
            session_id: self.id,
            accelerator_type: self.accelerator_type,
            total_duration: self.start_time.elapsed(),
            timing_summary,
            metric_summary,
            memory_summary,
            recommendations,
        })
    }
    
    /// Generate timing summary
    fn generate_timing_summary(&self) -> TimingSummary {
        let mut summary = TimingSummary::default();
        
        for event in &self.timing_events {
            let duration_ms = event.duration.as_secs_f64() * 1000.0;
            
            summary.total_events += 1;
            summary.total_time_ms += duration_ms;
            
            if duration_ms > summary.max_event_time_ms {
                summary.max_event_time_ms = duration_ms;
                summary.slowest_event_type = Some(event.event_type);
            }
            
            if summary.min_event_time_ms == 0.0 || duration_ms < summary.min_event_time_ms {
                summary.min_event_time_ms = duration_ms;
            }
        }
        
        if summary.total_events > 0 {
            summary.avg_event_time_ms = summary.total_time_ms / summary.total_events as f64;
        }
        
        summary
    }
    
    /// Generate metric summary
    fn generate_metric_summary(&self) -> MetricSummary {
        let mut summary = MetricSummary::default();
        
        for (metric_type, samples) in &self.metrics {
            if samples.is_empty() {
                continue;
            }
            
            let values: Vec<f64> = samples.iter().map(|s| s.value).collect();
            let sum: f64 = values.iter().sum();
            let avg = sum / values.len() as f64;
            let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            summary.metrics.insert(*metric_type, MetricStatistics {
                count: values.len(),
                average: avg,
                minimum: min,
                maximum: max,
                total: sum,
            });
        }
        
        summary
    }
    
    /// Generate memory summary
    fn generate_memory_summary(&self) -> MemorySummary {
        let mut summary = MemorySummary::default();
        
        if self.memory_samples.is_empty() {
            return summary;
        }
        
        let usage_values: Vec<u64> = self.memory_samples.iter().map(|s| s.usage_bytes).collect();
        
        summary.sample_count = usage_values.len();
        summary.peak_usage_bytes = usage_values.iter().max().copied().unwrap_or(0);
        summary.min_usage_bytes = usage_values.iter().min().copied().unwrap_or(0);
        summary.avg_usage_bytes = usage_values.iter().sum::<u64>() / usage_values.len() as u64;
        
        summary
    }
    
    /// Generate performance recommendations
    fn generate_recommendations(&self) -> Vec<PerformanceRecommendation> {
        let mut recommendations = Vec::new();
        
        // Analyze timing patterns
        let breakdown = self.calculate_timing_breakdown();
        
        if breakdown.memory_transfer_ms > breakdown.processing_ms * 0.5 {
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::Memory,
                priority: RecommendationPriority::High,
                description: "High memory transfer overhead detected. Consider batch processing or data locality optimizations.".to_string(),
                potential_improvement: "20-50% reduction in execution time".to_string(),
            });
        }
        
        if breakdown.synchronization_ms > breakdown.processing_ms * 0.2 {
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::Parallelization,
                priority: RecommendationPriority::Medium,
                description: "High synchronization overhead. Consider async execution or reducing synchronization points.".to_string(),
                potential_improvement: "10-30% reduction in execution time".to_string(),
            });
        }
        
        // Analyze GPU utilization
        if let Some(samples) = self.metrics.get(&MetricType::GpuUtilization) {
            if !samples.is_empty() {
                let avg_utilization = samples.iter().map(|s| s.value).sum::<f64>() / samples.len() as f64;
                if avg_utilization < 0.5 {
                    recommendations.push(PerformanceRecommendation {
                        category: RecommendationCategory::ResourceUtilization,
                        priority: RecommendationPriority::Medium,
                        description: format!("Low GPU utilization ({:.1}%). Consider increasing batch size or workload complexity.", avg_utilization * 100.0),
                        potential_improvement: "Improved hardware efficiency".to_string(),
                    });
                }
            }
        }
        
        recommendations
    }
    
    /// Export profiling data to file
    pub fn export_data(&self, format: ExportFormat, filename: &str) -> FFIResult<()> {
        match format {
            ExportFormat::Json => self.export_json(filename),
            ExportFormat::Csv => self.export_csv(filename),
            ExportFormat::Binary => self.export_binary(filename),
        }
    }
    
    fn export_json(&self, filename: &str) -> FFIResult<()> {
        let data = serde_json::to_string_pretty(&ExportData {
            session_id: self.id,
            accelerator_type: self.accelerator_type,
            timing_events: &self.timing_events,
            metrics: &self.metrics,
            memory_samples: &self.memory_samples,
        }).map_err(|e| FFIError::invalid_config(format!("JSON serialization failed: {}", e)))?;
        
        std::fs::write(filename, data)
            .map_err(|e| FFIError::invalid_config(format!("Failed to write file: {}", e)))
    }
    
    fn export_csv(&self, filename: &str) -> FFIResult<()> {
        let mut csv_data = String::new();
        csv_data.push_str("timestamp_ms,event_type,duration_ms\n");
        
        for event in &self.timing_events {
            csv_data.push_str(&format!(
                "{:.3},{:?},{:.3}\n",
                event.timestamp.as_secs_f64() * 1000.0,
                event.event_type,
                event.duration.as_secs_f64() * 1000.0
            ));
        }
        
        std::fs::write(filename, csv_data)
            .map_err(|e| FFIError::invalid_config(format!("Failed to write CSV: {}", e)))
    }
    
    fn export_binary(&self, filename: &str) -> FFIResult<()> {
        let data = bincode::serialize(&ExportData {
            session_id: self.id,
            accelerator_type: self.accelerator_type,
            timing_events: &self.timing_events,
            metrics: &self.metrics,
            memory_samples: &self.memory_samples,
        }).map_err(|e| FFIError::invalid_config(format!("Binary serialization failed: {}", e)))?;
        
        std::fs::write(filename, data)
            .map_err(|e| FFIError::invalid_config(format!("Failed to write binary file: {}", e)))
    }
}

/// Profiler configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Collect timing information
    pub collect_timing: bool,
    /// Collect performance metrics
    pub collect_metrics: bool,
    /// Collect memory usage data
    pub collect_memory: bool,
    /// Maximum number of samples per metric
    pub max_samples_per_metric: usize,
    /// Sampling interval for continuous metrics
    pub sampling_interval_ms: u64,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            collect_timing: true,
            collect_metrics: true,
            collect_memory: true,
            max_samples_per_metric: 10000,
            sampling_interval_ms: 100,
        }
    }
}

/// Types of timing events that can be recorded
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TimingEventType {
    /// Hardware initialization
    Initialization,
    /// Network deployment
    NetworkDeployment,
    /// Spike processing
    SpikeProcessing,
    /// Memory transfer (host-device)
    MemoryTransfer,
    /// Plasticity rule updates
    PlasticityUpdate,
    /// Synchronization operations
    Synchronization,
    /// Data preprocessing
    Preprocessing,
    /// Results postprocessing
    Postprocessing,
    /// Custom user-defined event
    Custom(u32),
}

/// Types of performance metrics that can be recorded
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MetricType {
    /// Spikes processed per second
    SpikesPerSecond,
    /// Power consumption in watts
    PowerConsumption,
    /// GPU utilization (0.0 to 1.0)
    GpuUtilization,
    /// Memory utilization (0.0 to 1.0)
    MemoryUtilization,
    /// Temperature in Celsius
    Temperature,
    /// Clock frequency in MHz
    ClockFrequency,
    /// Custom metric
    Custom(u32),
}

/// Individual timing event record
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TimingEvent {
    /// Type of event
    pub event_type: TimingEventType,
    /// Duration of the event
    pub duration: Duration,
    /// Timestamp relative to session start
    pub timestamp: Duration,
}

/// Individual metric sample
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricSample {
    /// Metric value
    pub value: f64,
    /// Timestamp relative to session start
    pub timestamp: Duration,
}

/// Individual memory usage sample
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemorySample {
    /// Memory usage in bytes
    pub usage_bytes: u64,
    /// Timestamp relative to session start
    pub timestamp: Duration,
}

/// Session state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session is actively collecting data
    Active,
    /// Session has been finalized
    Finalized,
    /// Session encountered an error
    Error,
}

/// Current status of a profiling session
#[derive(Debug, Clone)]
pub struct SessionStatus {
    /// Session ID
    pub id: ProfilerSessionId,
    /// Associated accelerator
    pub accelerator_id: AcceleratorId,
    /// Accelerator type
    pub accelerator_type: AcceleratorType,
    /// Associated network
    pub network_id: Option<NetworkId>,
    /// Current state
    pub state: SessionState,
    /// Session duration so far
    pub duration: Duration,
    /// Number of timing events recorded
    pub event_count: usize,
    /// Number of metric samples recorded
    pub metric_count: usize,
    /// Number of memory samples recorded
    pub memory_sample_count: usize,
}

/// Complete results from a profiling session
#[derive(Debug)]
pub struct ProfilerResults {
    /// Session ID
    pub session_id: ProfilerSessionId,
    /// Associated accelerator
    pub accelerator_id: AcceleratorId,
    /// Accelerator type
    pub accelerator_type: AcceleratorType,
    /// Associated network
    pub network_id: Option<NetworkId>,
    /// Total session duration
    pub total_duration: Duration,
    /// Timing breakdown
    pub timing_breakdown: TimingBreakdown,
    /// Aggregate performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// All timing events
    pub timing_events: Vec<TimingEvent>,
    /// All metric samples
    pub metric_samples: HashMap<MetricType, Vec<MetricSample>>,
    /// All memory samples
    pub memory_samples: Vec<MemorySample>,
}

/// Global metrics aggregator
#[derive(Debug)]
struct GlobalMetrics {
    /// Total number of sessions
    total_sessions: u64,
    /// Total execution time across all sessions
    total_execution_time: Duration,
    /// Average metrics by accelerator type
    accelerator_metrics: HashMap<AcceleratorType, AcceleratorMetrics>,
}

impl GlobalMetrics {
    fn new() -> Self {
        Self {
            total_sessions: 0,
            total_execution_time: Duration::default(),
            accelerator_metrics: HashMap::new(),
        }
    }
    
    fn update_from_session(&mut self, results: &ProfilerResults) {
        self.total_sessions += 1;
        self.total_execution_time += results.total_duration;
        
        let entry = self.accelerator_metrics
            .entry(results.accelerator_type)
            .or_insert_with(AcceleratorMetrics::new);
        
        entry.update_from_results(results);
    }
    
    fn get_statistics(&self) -> GlobalStatistics {
        GlobalStatistics {
            total_sessions: self.total_sessions,
            total_execution_time: self.total_execution_time,
            accelerator_statistics: self.accelerator_metrics.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct AcceleratorMetrics {
    session_count: u64,
    total_execution_time: Duration,
    avg_spikes_per_second: f64,
    avg_power_consumption: f32,
}

impl AcceleratorMetrics {
    fn new() -> Self {
        Self {
            session_count: 0,
            total_execution_time: Duration::default(),
            avg_spikes_per_second: 0.0,
            avg_power_consumption: 0.0,
        }
    }
    
    fn update_from_results(&mut self, results: &ProfilerResults) {
        self.session_count += 1;
        self.total_execution_time += results.total_duration;
        
        // Update running averages
        let weight = 1.0 / self.session_count as f64;
        let inv_weight = 1.0 - weight;
        
        self.avg_spikes_per_second = self.avg_spikes_per_second * inv_weight + 
            results.performance_metrics.spikes_per_second * weight;
        
        self.avg_power_consumption = self.avg_power_consumption * inv_weight as f32 + 
            results.performance_metrics.power_consumption * weight as f32;
    }
}

/// Export formats for profiling data
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Binary format
    Binary,
}

/// Data structure for export
#[derive(serde::Serialize, serde::Deserialize)]
struct ExportData<'a> {
    session_id: ProfilerSessionId,
    accelerator_type: AcceleratorType,
    timing_events: &'a [TimingEvent],
    metrics: &'a HashMap<MetricType, Vec<MetricSample>>,
    memory_samples: &'a [MemorySample],
}

/// Global performance statistics
#[derive(Debug, Clone, Default)]
pub struct GlobalStatistics {
    /// Total number of profiling sessions
    pub total_sessions: u64,
    /// Total execution time across all sessions
    pub total_execution_time: Duration,
    /// Statistics by accelerator type
    pub accelerator_statistics: HashMap<AcceleratorType, AcceleratorMetrics>,
}

/// Comprehensive performance report
#[derive(Debug)]
pub struct PerformanceReport {
    /// Session ID
    pub session_id: ProfilerSessionId,
    /// Accelerator type
    pub accelerator_type: AcceleratorType,
    /// Total duration
    pub total_duration: Duration,
    /// Timing analysis
    pub timing_summary: TimingSummary,
    /// Metric analysis
    pub metric_summary: MetricSummary,
    /// Memory analysis
    pub memory_summary: MemorySummary,
    /// Performance recommendations
    pub recommendations: Vec<PerformanceRecommendation>,
}

/// Summary of timing analysis
#[derive(Debug, Default)]
pub struct TimingSummary {
    /// Total number of timing events
    pub total_events: usize,
    /// Total time spent in all events
    pub total_time_ms: f64,
    /// Average time per event
    pub avg_event_time_ms: f64,
    /// Minimum event time
    pub min_event_time_ms: f64,
    /// Maximum event time
    pub max_event_time_ms: f64,
    /// Type of slowest event
    pub slowest_event_type: Option<TimingEventType>,
}

/// Summary of metric analysis
#[derive(Debug, Default)]
pub struct MetricSummary {
    /// Statistics for each metric type
    pub metrics: HashMap<MetricType, MetricStatistics>,
}

/// Statistics for a specific metric
#[derive(Debug)]
pub struct MetricStatistics {
    /// Number of samples
    pub count: usize,
    /// Average value
    pub average: f64,
    /// Minimum value
    pub minimum: f64,
    /// Maximum value
    pub maximum: f64,
    /// Total/sum of all values
    pub total: f64,
}

/// Summary of memory analysis
#[derive(Debug, Default)]
pub struct MemorySummary {
    /// Number of memory samples
    pub sample_count: usize,
    /// Peak memory usage
    pub peak_usage_bytes: u64,
    /// Minimum memory usage
    pub min_usage_bytes: u64,
    /// Average memory usage
    pub avg_usage_bytes: u64,
}

/// Performance improvement recommendation
#[derive(Debug)]
pub struct PerformanceRecommendation {
    /// Category of recommendation
    pub category: RecommendationCategory,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Description of the issue and suggested fix
    pub description: String,
    /// Potential improvement estimate
    pub potential_improvement: String,
}

/// Categories of performance recommendations
#[derive(Debug, Clone, Copy)]
pub enum RecommendationCategory {
    /// Memory optimization
    Memory,
    /// Parallelization improvements
    Parallelization,
    /// Resource utilization
    ResourceUtilization,
    /// Algorithm optimization
    Algorithm,
    /// Hardware configuration
    Hardware,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Create a global profiler instance
static GLOBAL_PROFILER: once_cell::sync::Lazy<Arc<Mutex<Option<PerformanceProfiler>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(None)));

/// Initialize the global profiler
pub fn init_profiler(config: ProfilerConfig) -> FFIResult<()> {
    let mut profiler = GLOBAL_PROFILER.lock().unwrap();
    *profiler = Some(PerformanceProfiler::new(config));
    Ok(())
}

/// Get the global profiler
pub fn get_profiler() -> FFIResult<Arc<Mutex<PerformanceProfiler>>> {
    let profiler_guard = GLOBAL_PROFILER.lock().unwrap();
    match profiler_guard.as_ref() {
        Some(profiler) => {
            // Clone the profiler into a new Arc<Mutex<>>
            let config = profiler.config.clone();
            drop(profiler_guard);
            Ok(Arc::new(Mutex::new(PerformanceProfiler::new(config))))
        }
        None => Err(FFIError::invalid_config("Profiler not initialized")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_profiler_session() {
        let config = ProfilerConfig::default();
        let profiler = PerformanceProfiler::new(config);
        
        let session_id = profiler.start_session(
            crate::AcceleratorId(1),
            AcceleratorType::CPU,
            None,
        ).unwrap();
        
        // Record some events
        profiler.record_timing(
            session_id,
            TimingEventType::SpikeProcessing,
            Duration::from_millis(10),
        ).unwrap();
        
        profiler.record_metric(
            session_id,
            MetricType::SpikesPerSecond,
            1000.0,
        ).unwrap();
        
        let status = profiler.get_session_status(session_id).unwrap();
        assert_eq!(status.state, SessionState::Active);
        assert_eq!(status.event_count, 1);
        assert_eq!(status.metric_count, 1);
        
        let results = profiler.end_session(session_id).unwrap();
        assert_eq!(results.timing_events.len(), 1);
        assert_eq!(results.metric_samples.len(), 1);
    }
    
    #[test]
    fn test_timing_breakdown_calculation() {
        let config = ProfilerConfig::default();
        let mut session = ProfilerSession::new(
            ProfilerSessionId(1),
            crate::AcceleratorId(1),
            AcceleratorType::CPU,
            None,
            config,
        );
        
        session.record_timing(TimingEventType::SpikeProcessing, Duration::from_millis(50));
        session.record_timing(TimingEventType::MemoryTransfer, Duration::from_millis(20));
        session.record_timing(TimingEventType::Synchronization, Duration::from_millis(5));
        
        let breakdown = session.calculate_timing_breakdown();
        assert_eq!(breakdown.processing_ms, 50.0);
        assert_eq!(breakdown.memory_transfer_ms, 20.0);
        assert_eq!(breakdown.synchronization_ms, 5.0);
    }
    
    #[test]
    fn test_performance_recommendations() {
        let config = ProfilerConfig::default();
        let mut session = ProfilerSession::new(
            ProfilerSessionId(1),
            crate::AcceleratorId(1),
            AcceleratorType::CUDA,
            None,
            config,
        );
        
        // Simulate high memory transfer overhead
        session.record_timing(TimingEventType::SpikeProcessing, Duration::from_millis(10));
        session.record_timing(TimingEventType::MemoryTransfer, Duration::from_millis(20));
        
        // Simulate low GPU utilization
        session.record_metric(MetricType::GpuUtilization, 0.3);
        
        let recommendations = session.generate_recommendations();
        assert!(!recommendations.is_empty());
        
        // Should have recommendations for memory and resource utilization
        let has_memory_rec = recommendations.iter()
            .any(|r| matches!(r.category, RecommendationCategory::Memory));
        let has_resource_rec = recommendations.iter()
            .any(|r| matches!(r.category, RecommendationCategory::ResourceUtilization));
        
        assert!(has_memory_rec);
        assert!(has_resource_rec);
    }
}