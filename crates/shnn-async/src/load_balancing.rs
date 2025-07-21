//! Load balancing for distributed neural network processing
//!
//! This module provides intelligent load balancing strategies for optimal
//! resource utilization across distributed nodes.

use crate::error::{AsyncError, AsyncResult};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Load balancing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Weighted distribution
    Weighted,
    /// Resource-aware balancing
    ResourceAware,
}

impl Default for BalancingStrategy {
    fn default() -> Self {
        Self::RoundRobin
    }
}

/// Workload metrics for load balancing decisions
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkloadMetrics {
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f64,
    /// Current active tasks
    pub active_tasks: u32,
    /// Average response time
    pub avg_response_time: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
}

impl WorkloadMetrics {
    /// Calculate overall load score
    pub fn load_score(&self) -> f64 {
        // Weighted combination of metrics
        let cpu_weight = 0.3;
        let memory_weight = 0.2;
        let task_weight = 0.2;
        let response_weight = 0.2;
        let error_weight = 0.1;
        
        cpu_weight * self.cpu_utilization
            + memory_weight * self.memory_utilization
            + task_weight * (self.active_tasks as f64 / 100.0).min(1.0)
            + response_weight * (self.avg_response_time / 1000.0).min(1.0)
            + error_weight * self.error_rate
    }
}

/// Load balancer for distributing workload
pub struct LoadBalancer {
    /// Balancing strategy
    strategy: BalancingStrategy,
    /// Node metrics
    node_metrics: HashMap<String, WorkloadMetrics>,
    /// Round-robin counter
    round_robin_counter: std::sync::atomic::AtomicUsize,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(strategy: BalancingStrategy) -> Self {
        Self {
            strategy,
            node_metrics: HashMap::new(),
            round_robin_counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    
    /// Update node metrics
    pub fn update_metrics(&mut self, node_id: String, metrics: WorkloadMetrics) {
        self.node_metrics.insert(node_id, metrics);
    }
    
    /// Select best node for new task
    pub fn select_node(&self, available_nodes: &[String]) -> Option<String> {
        if available_nodes.is_empty() {
            return None;
        }
        
        match self.strategy {
            BalancingStrategy::RoundRobin => {
                let index = self.round_robin_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(available_nodes[index % available_nodes.len()].clone())
            }
            BalancingStrategy::LeastConnections => {
                available_nodes.iter()
                    .min_by_key(|&node_id| {
                        self.node_metrics.get(node_id)
                            .map(|m| m.active_tasks)
                            .unwrap_or(0)
                    })
                    .cloned()
            }
            BalancingStrategy::ResourceAware => {
                available_nodes.iter()
                    .min_by(|&a, &b| {
                        let load_a = self.node_metrics.get(a)
                            .map(|m| m.load_score())
                            .unwrap_or(0.0);
                        let load_b = self.node_metrics.get(b)
                            .map(|m| m.load_score())
                            .unwrap_or(0.0);
                        load_a.partial_cmp(&load_b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .cloned()
            }
            BalancingStrategy::Weighted => {
                // Simplified weighted selection
                self.select_node_weighted(available_nodes)
            }
        }
    }
    
    /// Weighted node selection
    fn select_node_weighted(&self, available_nodes: &[String]) -> Option<String> {
        // Simple implementation: inverse of load score as weight
        let mut total_weight = 0.0;
        let weights: Vec<f64> = available_nodes.iter()
            .map(|node_id| {
                let load = self.node_metrics.get(node_id)
                    .map(|m| m.load_score())
                    .unwrap_or(0.5);
                let weight = 1.0 / (load + 0.1); // Avoid division by zero
                total_weight += weight;
                weight
            })
            .collect();
        
        if total_weight == 0.0 {
            return available_nodes.first().cloned();
        }
        
        // Simple random selection based on weights (would use proper RNG in real impl)
        let mut cumulative = 0.0;
        let target = total_weight * 0.5; // Simplified selection
        
        for (i, weight) in weights.iter().enumerate() {
            cumulative += weight;
            if cumulative >= target {
                return Some(available_nodes[i].clone());
            }
        }
        
        available_nodes.last().cloned()
    }
    
    /// Get current strategy
    pub fn strategy(&self) -> BalancingStrategy {
        self.strategy
    }
    
    /// Set new strategy
    pub fn set_strategy(&mut self, strategy: BalancingStrategy) {
        self.strategy = strategy;
    }
    
    /// Get node metrics
    pub fn get_metrics(&self, node_id: &str) -> Option<&WorkloadMetrics> {
        self.node_metrics.get(node_id)
    }
    
    /// Remove node metrics
    pub fn remove_node(&mut self, node_id: &str) {
        self.node_metrics.remove(node_id);
    }
}

impl Default for LoadBalancer {
    fn default() -> Self {
        Self::new(BalancingStrategy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_load_balancer_round_robin() {
        let balancer = LoadBalancer::new(BalancingStrategy::RoundRobin);
        let nodes = vec!["node1".to_string(), "node2".to_string(), "node3".to_string()];
        
        // Should cycle through nodes
        assert_eq!(balancer.select_node(&nodes), Some("node1".to_string()));
        assert_eq!(balancer.select_node(&nodes), Some("node2".to_string()));
        assert_eq!(balancer.select_node(&nodes), Some("node3".to_string()));
        assert_eq!(balancer.select_node(&nodes), Some("node1".to_string()));
    }
    
    #[test]
    fn test_workload_metrics() {
        let metrics = WorkloadMetrics {
            cpu_utilization: 0.5,
            memory_utilization: 0.3,
            active_tasks: 10,
            avg_response_time: 100.0,
            error_rate: 0.01,
        };
        
        let load_score = metrics.load_score();
        assert!(load_score > 0.0 && load_score < 1.0);
    }
    
    #[test]
    fn test_least_connections_strategy() {
        let mut balancer = LoadBalancer::new(BalancingStrategy::LeastConnections);
        let nodes = vec!["node1".to_string(), "node2".to_string()];
        
        // Set different task counts
        balancer.update_metrics("node1".to_string(), WorkloadMetrics {
            active_tasks: 5,
            ..Default::default()
        });
        balancer.update_metrics("node2".to_string(), WorkloadMetrics {
            active_tasks: 2,
            ..Default::default()
        });
        
        // Should select node with fewer connections
        assert_eq!(balancer.select_node(&nodes), Some("node2".to_string()));
    }
    
    #[test]
    fn test_empty_nodes() {
        let balancer = LoadBalancer::new(BalancingStrategy::RoundRobin);
        let nodes = vec![];
        
        assert_eq!(balancer.select_node(&nodes), None);
    }
}