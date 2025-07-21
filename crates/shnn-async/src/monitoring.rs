//! Network monitoring and metrics collection
//!
//! This module provides comprehensive monitoring capabilities for async
//! neural network performance and health.

use crate::error::{AsyncError, AsyncResult};
use shnn_core::time::{Time, Duration};

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::Instant,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn, error};

/// Monitoring configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MonitoringConfig {
    /// Collection interval
    pub collection_interval: Duration,
    /// Enable performance metrics
    pub enable_performance: bool,
    /// Enable health checks
    pub enable_health_checks: bool,
    /// Enable alerting
    pub enable_alerting: bool,
    /// Maximum metric history size
    pub max_history_size: usize,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_millis(1000),
            enable_performance: true,
            enable_health_checks: true,
            enable_alerting: false,
            max_history_size: 1000,
        }
    }
}

/// Processing metrics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProcessingMetrics {
    /// Number of spikes processed
    pub spikes_processed: u64,
    /// Total processing time
    pub total_processing_time: std::time::Duration,
    /// Average processing time per spike
    pub avg_processing_time: std::time::Duration,
    /// Peak processing rate (spikes/second)
    pub peak_processing_rate: f64,
    /// Current processing rate
    pub current_processing_rate: f64,
    /// Memory usage (bytes)
    pub memory_usage: u64,
    /// Error count
    pub error_count: u64,
    /// Last update timestamp
    pub last_update: Instant,
}

impl ProcessingMetrics {
    /// Create new processing metrics
    pub fn new() -> Self {
        Self {
            last_update: Instant::now(),
            ..Default::default()
        }
    }
    
    /// Update processing rate
    pub fn update_processing_rate(&mut self) {
        let elapsed = self.last_update.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.current_processing_rate = self.spikes_processed as f64 / elapsed;
            self.peak_processing_rate = self.peak_processing_rate.max(self.current_processing_rate);
        }
        self.last_update = Instant::now();
    }
    
    /// Update average processing time
    pub fn update_avg_processing_time(&mut self) {
        if self.spikes_processed > 0 {
            self.avg_processing_time = self.total_processing_time / self.spikes_processed as u32;
        }
    }
}

/// Health status
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HealthStatus {
    /// System is healthy
    Healthy,
    /// System has warnings
    Warning,
    /// System has errors
    Error,
    /// System is critical
    Critical,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self::Healthy
    }
}

/// Health check result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HealthCheck {
    /// Health status
    pub status: HealthStatus,
    /// Check name
    pub name: String,
    /// Status message
    pub message: String,
    /// Check timestamp
    pub timestamp: Instant,
    /// Check duration
    pub duration: std::time::Duration,
}

impl HealthCheck {
    /// Create a new health check
    pub fn new(name: String, status: HealthStatus, message: String) -> Self {
        Self {
            name,
            status,
            message,
            timestamp: Instant::now(),
            duration: std::time::Duration::from_millis(0),
        }
    }
    
    /// Check if healthy
    pub fn is_healthy(&self) -> bool {
        self.status == HealthStatus::Healthy
    }
}

/// Network monitor
pub struct NetworkMonitor {
    /// Configuration
    config: MonitoringConfig,
    /// Current metrics
    metrics: Arc<RwLock<ProcessingMetrics>>,
    /// Metric history
    metric_history: Arc<RwLock<Vec<ProcessingMetrics>>>,
    /// Health checks
    health_checks: Arc<RwLock<HashMap<String, HealthCheck>>>,
    /// Monitor state
    state: Arc<RwLock<MonitorState>>,
}

/// Monitor state
#[derive(Debug, Clone)]
struct MonitorState {
    /// Whether monitoring is active
    active: bool,
    /// Last collection time
    last_collection: Instant,
    /// Alert count
    alert_count: u64,
}

impl NetworkMonitor {
    /// Create a new network monitor
    pub fn new() -> Self {
        Self::with_config(MonitoringConfig::default())
    }
    
    /// Create with specific configuration
    pub fn with_config(config: MonitoringConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(ProcessingMetrics::new())),
            metric_history: Arc::new(RwLock::new(Vec::new())),
            health_checks: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(RwLock::new(MonitorState {
                active: true,
                last_collection: Instant::now(),
                alert_count: 0,
            })),
        }
    }
    
    /// Collect metrics
    pub async fn collect_metrics(&self, new_metrics: &ProcessingMetrics) -> AsyncResult<()> {
        if !self.is_active() {
            return Ok(());
        }
        
        // Update current metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            *metrics = new_metrics.clone();
            metrics.update_processing_rate();
            metrics.update_avg_processing_time();
        }
        
        // Add to history
        {
            let mut history = self.metric_history.write().unwrap();
            history.push(new_metrics.clone());
            
            // Trim history if too large
            if history.len() > self.config.max_history_size {
                history.remove(0);
            }
        }
        
        // Update state
        {
            let mut state = self.state.write().unwrap();
            state.last_collection = Instant::now();
        }
        
        #[cfg(feature = "tracing")]
        debug!(
            spikes_processed = new_metrics.spikes_processed,
            processing_rate = new_metrics.current_processing_rate,
            "Collected metrics"
        );
        
        Ok(())
    }
    
    /// Get current metrics
    pub fn get_metrics(&self) -> ProcessingMetrics {
        self.metrics.read().unwrap().clone()
    }
    
    /// Get metric history
    pub fn get_metric_history(&self) -> Vec<ProcessingMetrics> {
        self.metric_history.read().unwrap().clone()
    }
    
    /// Add health check
    pub async fn add_health_check(&self, check: HealthCheck) -> AsyncResult<()> {
        if !self.config.enable_health_checks {
            return Ok(());
        }
        
        let name = check.name.clone();
        let is_healthy = check.is_healthy();
        
        {
            let mut checks = self.health_checks.write().unwrap();
            checks.insert(name.clone(), check);
        }
        
        if !is_healthy && self.config.enable_alerting {
            self.trigger_alert(&name).await?;
        }
        
        #[cfg(feature = "tracing")]
        debug!(check_name = %name, is_healthy, "Added health check");
        
        Ok(())
    }
    
    /// Get all health checks
    pub fn get_health_checks(&self) -> HashMap<String, HealthCheck> {
        self.health_checks.read().unwrap().clone()
    }
    
    /// Get overall health status
    pub fn get_overall_health(&self) -> HealthStatus {
        let checks = self.health_checks.read().unwrap();
        
        if checks.is_empty() {
            return HealthStatus::Healthy;
        }
        
        let mut has_critical = false;
        let mut has_error = false;
        let mut has_warning = false;
        
        for check in checks.values() {
            match check.status {
                HealthStatus::Critical => has_critical = true,
                HealthStatus::Error => has_error = true,
                HealthStatus::Warning => has_warning = true,
                HealthStatus::Healthy => {}
            }
        }
        
        if has_critical {
            HealthStatus::Critical
        } else if has_error {
            HealthStatus::Error
        } else if has_warning {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        }
    }
    
    /// Run performance health check
    pub async fn run_performance_check(&self) -> AsyncResult<HealthCheck> {
        let metrics = self.get_metrics();
        let avg_time_ms = metrics.avg_processing_time.as_millis();
        
        let (status, message) = if avg_time_ms > 100 {
            (HealthStatus::Critical, format!("High processing latency: {}ms", avg_time_ms))
        } else if avg_time_ms > 50 {
            (HealthStatus::Warning, format!("Elevated processing latency: {}ms", avg_time_ms))
        } else if metrics.error_count > 10 {
            (HealthStatus::Error, format!("High error count: {}", metrics.error_count))
        } else {
            (HealthStatus::Healthy, "Performance is healthy".to_string())
        };
        
        Ok(HealthCheck::new("performance".to_string(), status, message))
    }
    
    /// Trigger alert
    async fn trigger_alert(&self, check_name: &str) -> AsyncResult<()> {
        {
            let mut state = self.state.write().unwrap();
            state.alert_count += 1;
        }
        
        #[cfg(feature = "tracing")]
        warn!(check_name, "Health check alert triggered");
        
        // In a real implementation, this would send notifications
        // (email, webhook, etc.)
        
        Ok(())
    }
    
    /// Check if monitor is active
    pub fn is_active(&self) -> bool {
        self.state.read().unwrap().active
    }
    
    /// Start monitoring
    pub fn start(&self) {
        let mut state = self.state.write().unwrap();
        state.active = true;
        
        #[cfg(feature = "tracing")]
        info!("Network monitoring started");
    }
    
    /// Stop monitoring
    pub fn stop(&self) {
        let mut state = self.state.write().unwrap();
        state.active = false;
        
        #[cfg(feature = "tracing")]
        info!("Network monitoring stopped");
    }
    
    /// Get monitor statistics
    pub fn get_stats(&self) -> MonitorStats {
        let state = self.state.read().unwrap();
        let checks = self.health_checks.read().unwrap();
        let history = self.metric_history.read().unwrap();
        
        MonitorStats {
            active: state.active,
            uptime: state.last_collection.elapsed(),
            alert_count: state.alert_count,
            health_check_count: checks.len(),
            metric_history_size: history.len(),
            overall_health: self.get_overall_health(),
        }
    }
}

impl Default for NetworkMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Monitor statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MonitorStats {
    /// Whether monitor is active
    pub active: bool,
    /// Monitor uptime
    pub uptime: std::time::Duration,
    /// Alert count
    pub alert_count: u64,
    /// Number of health checks
    pub health_check_count: usize,
    /// Metric history size
    pub metric_history_size: usize,
    /// Overall health status
    pub overall_health: HealthStatus,
}

impl std::fmt::Display for MonitorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Monitor: {} (uptime: {:.1}s, {} alerts, {} checks, health: {:?})",
            if self.active { "active" } else { "inactive" },
            self.uptime.as_secs_f64(),
            self.alert_count,
            self.health_check_count,
            self.overall_health
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_network_monitor() {
        let monitor = NetworkMonitor::new();
        assert!(monitor.is_active());
        
        let metrics = ProcessingMetrics::new();
        assert!(monitor.collect_metrics(&metrics).await.is_ok());
        
        let retrieved = monitor.get_metrics();
        assert_eq!(retrieved.spikes_processed, metrics.spikes_processed);
    }
    
    #[tokio::test]
    async fn test_health_checks() {
        let monitor = NetworkMonitor::new();
        
        let check = HealthCheck::new(
            "test".to_string(),
            HealthStatus::Healthy,
            "All good".to_string(),
        );
        
        assert!(monitor.add_health_check(check).await.is_ok());
        
        let checks = monitor.get_health_checks();
        assert_eq!(checks.len(), 1);
        assert!(checks.contains_key("test"));
    }
    
    #[tokio::test]
    async fn test_performance_check() {
        let monitor = NetworkMonitor::new();
        let check = monitor.run_performance_check().await.unwrap();
        
        assert_eq!(check.name, "performance");
        assert!(check.is_healthy()); // Should be healthy for new metrics
    }
}