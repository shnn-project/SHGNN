//! Memory-efficient data structures for embedded neuromorphic systems
//!
//! This module provides specialized data structures optimized for constrained
//! memory environments in embedded systems, using fixed-size arrays and
//! minimal heap allocation.

use crate::{
    error::{EmbeddedError, EmbeddedResult},
    fixed_point::{FixedPoint, Q16_16, FixedSpike},
};
use heapless::{Vec, FnvIndexMap, Deque, pool::{Pool, Node}};
use core::{mem::MaybeUninit, slice};

/// Maximum number of hyperedges in embedded hypergraph
pub const MAX_HYPEREDGES: usize = 128;

/// Maximum number of nodes per hyperedge
pub const MAX_NODES_PER_EDGE: usize = 8;

/// Maximum spike buffer size
pub const MAX_SPIKE_BUFFER_SIZE: usize = 256;

/// Maximum memory pool size for dynamic allocation
pub const MEMORY_POOL_SIZE: usize = 1024;

/// Embedded hypergraph implementation using static allocation
#[derive(Debug)]
pub struct EmbeddedHypergraph<T: FixedPoint> {
    /// Hyperedges stored as arrays of node indices
    edges: Vec<EmbeddedHyperedge<T>, MAX_HYPEREDGES>,
    /// Node connection lookup for efficient access
    node_connections: FnvIndexMap<u16, Vec<u16, 16>, 64>, // node_id -> edge_ids
    /// Edge weights for learning
    edge_weights: Vec<T, MAX_HYPEREDGES>,
    /// Edge activation states
    edge_activations: Vec<bool, MAX_HYPEREDGES>,
    /// Next available edge ID
    next_edge_id: u16,
}

/// Embedded hyperedge representation
#[derive(Debug, Clone)]
pub struct EmbeddedHyperedge<T: FixedPoint> {
    /// Edge ID
    pub id: u16,
    /// Connected node IDs
    pub nodes: Vec<u16, MAX_NODES_PER_EDGE>,
    /// Edge weight
    pub weight: T,
    /// Edge type
    pub edge_type: HyperedgeType,
    /// Last activation time
    pub last_activation: T,
}

/// Types of hyperedges in embedded systems
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HyperedgeType {
    /// Excitatory connection
    Excitatory,
    /// Inhibitory connection
    Inhibitory,
    /// Modulatory connection
    Modulatory,
    /// Plasticity rule connection
    Plasticity,
}

impl<T: FixedPoint> EmbeddedHypergraph<T> {
    /// Create a new embedded hypergraph
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            node_connections: FnvIndexMap::new(),
            edge_weights: Vec::new(),
            edge_activations: Vec::new(),
            next_edge_id: 0,
        }
    }
    
    /// Add a hyperedge to the graph
    pub fn add_hyperedge(
        &mut self, 
        nodes: &[u16], 
        weight: T, 
        edge_type: HyperedgeType
    ) -> EmbeddedResult<u16> {
        if nodes.len() > MAX_NODES_PER_EDGE {
            return Err(EmbeddedError::BufferFull);
        }
        
        let edge_id = self.next_edge_id;
        
        // Create node vector
        let mut edge_nodes = Vec::new();
        for &node in nodes {
            edge_nodes.push(node)
                .map_err(|_| EmbeddedError::BufferFull)?;
        }
        
        // Create hyperedge
        let hyperedge = EmbeddedHyperedge {
            id: edge_id,
            nodes: edge_nodes,
            weight,
            edge_type,
            last_activation: T::zero(),
        };
        
        // Add to storage
        self.edges.push(hyperedge)
            .map_err(|_| EmbeddedError::BufferFull)?;
        self.edge_weights.push(weight)
            .map_err(|_| EmbeddedError::BufferFull)?;
        self.edge_activations.push(false)
            .map_err(|_| EmbeddedError::BufferFull)?;
        
        // Update node connections
        for &node in nodes {
            let connections = self.node_connections.entry(node)
                .or_insert(Vec::new());
            connections.push(edge_id)
                .map_err(|_| EmbeddedError::BufferFull)?;
        }
        
        self.next_edge_id += 1;
        Ok(edge_id)
    }
    
    /// Get hyperedges connected to a node
    pub fn get_node_edges(&self, node_id: u16) -> Option<&Vec<u16, 16>> {
        self.node_connections.get(&node_id)
    }
    
    /// Activate a hyperedge
    pub fn activate_edge(&mut self, edge_id: u16, activation_time: T) -> EmbeddedResult<()> {
        if let Some(edge) = self.edges.iter_mut().find(|e| e.id == edge_id) {
            edge.last_activation = activation_time;
            if let Some(activation) = self.edge_activations.get_mut(edge_id as usize) {
                *activation = true;
            }
            Ok(())
        } else {
            Err(EmbeddedError::InvalidIndex)
        }
    }
    
    /// Get edge weight
    pub fn get_edge_weight(&self, edge_id: u16) -> Option<T> {
        self.edge_weights.get(edge_id as usize).copied()
    }
    
    /// Update edge weight
    pub fn update_edge_weight(&mut self, edge_id: u16, new_weight: T) -> EmbeddedResult<()> {
        if let Some(weight) = self.edge_weights.get_mut(edge_id as usize) {
            *weight = new_weight;
            
            // Also update the weight in the edge structure
            if let Some(edge) = self.edges.iter_mut().find(|e| e.id == edge_id) {
                edge.weight = new_weight;
            }
            
            Ok(())
        } else {
            Err(EmbeddedError::InvalidIndex)
        }
    }
    
    /// Clear all activations
    pub fn clear_activations(&mut self) {
        for activation in &mut self.edge_activations {
            *activation = false;
        }
    }
    
    /// Get number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
    
    /// Get number of connected nodes
    pub fn node_count(&self) -> usize {
        self.node_connections.len()
    }
}

impl<T: FixedPoint> Default for EmbeddedHypergraph<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Circular buffer for spike storage in embedded systems
#[derive(Debug)]
pub struct EmbeddedSpikeBuffer<T: FixedPoint> {
    /// Spike storage
    buffer: [Option<FixedSpike<T>>; MAX_SPIKE_BUFFER_SIZE],
    /// Current write position
    write_pos: usize,
    /// Current read position
    read_pos: usize,
    /// Number of spikes currently stored
    count: usize,
    /// Total spikes processed (for statistics)
    total_processed: u32,
}

impl<T: FixedPoint> EmbeddedSpikeBuffer<T> {
    /// Create a new spike buffer
    pub fn new() -> Self {
        Self {
            buffer: [None; MAX_SPIKE_BUFFER_SIZE],
            write_pos: 0,
            read_pos: 0,
            count: 0,
            total_processed: 0,
        }
    }
    
    /// Add a spike to the buffer
    pub fn add_spike(&mut self, spike: FixedSpike<T>) -> EmbeddedResult<()> {
        if self.count >= MAX_SPIKE_BUFFER_SIZE {
            // Buffer full, overwrite oldest
            self.read_pos = (self.read_pos + 1) % MAX_SPIKE_BUFFER_SIZE;
        } else {
            self.count += 1;
        }
        
        self.buffer[self.write_pos] = Some(spike);
        self.write_pos = (self.write_pos + 1) % MAX_SPIKE_BUFFER_SIZE;
        self.total_processed += 1;
        
        Ok(())
    }
    
    /// Get the next spike from the buffer
    pub fn get_spike(&mut self) -> Option<FixedSpike<T>> {
        if self.count == 0 {
            return None;
        }
        
        let spike = self.buffer[self.read_pos].take();
        self.read_pos = (self.read_pos + 1) % MAX_SPIKE_BUFFER_SIZE;
        self.count -= 1;
        
        spike
    }
    
    /// Peek at the next spike without removing it
    pub fn peek_spike(&self) -> Option<&FixedSpike<T>> {
        if self.count == 0 {
            None
        } else {
            self.buffer[self.read_pos].as_ref()
        }
    }
    
    /// Get spikes within a time window
    pub fn get_spikes_in_window(&self, start_time: T, end_time: T) -> Vec<FixedSpike<T>, 32> {
        let mut result = Vec::new();
        
        for i in 0..self.count {
            let idx = (self.read_pos + i) % MAX_SPIKE_BUFFER_SIZE;
            if let Some(spike) = &self.buffer[idx] {
                if spike.timestamp >= start_time && spike.timestamp <= end_time {
                    let _ = result.push(*spike);
                }
            }
        }
        
        result
    }
    
    /// Clear all spikes
    pub fn clear(&mut self) {
        for spike in &mut self.buffer {
            *spike = None;
        }
        self.write_pos = 0;
        self.read_pos = 0;
        self.count = 0;
    }
    
    /// Get buffer statistics
    pub fn get_stats(&self) -> SpikeBufferStats {
        SpikeBufferStats {
            current_count: self.count,
            total_processed: self.total_processed,
            buffer_utilization: (self.count as f32 / MAX_SPIKE_BUFFER_SIZE as f32) * 100.0,
        }
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.count >= MAX_SPIKE_BUFFER_SIZE
    }
}

impl<T: FixedPoint> Default for EmbeddedSpikeBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for spike buffer monitoring
#[derive(Debug, Clone)]
pub struct SpikeBufferStats {
    /// Current number of spikes in buffer
    pub current_count: usize,
    /// Total spikes processed since creation
    pub total_processed: u32,
    /// Buffer utilization percentage
    pub buffer_utilization: f32,
}

/// Memory-efficient weight matrix for small networks
#[derive(Debug)]
pub struct EmbeddedWeightMatrix<T: FixedPoint> {
    /// Flattened weight matrix storage
    weights: Vec<T, 1024>, // Support up to 32x32 matrix
    /// Matrix dimensions
    rows: usize,
    cols: usize,
    /// Sparsity pattern (for sparse matrices)
    sparse_indices: Option<Vec<(u16, u16), 512>>, // (row, col) pairs for non-zero entries
}

impl<T: FixedPoint> EmbeddedWeightMatrix<T> {
    /// Create a new dense weight matrix
    pub fn new_dense(rows: usize, cols: usize, default_weight: T) -> EmbeddedResult<Self> {
        if rows * cols > 1024 {
            return Err(EmbeddedError::BufferFull);
        }
        
        let mut weights = Vec::new();
        for _ in 0..rows * cols {
            weights.push(default_weight)
                .map_err(|_| EmbeddedError::BufferFull)?;
        }
        
        Ok(Self {
            weights,
            rows,
            cols,
            sparse_indices: None,
        })
    }
    
    /// Create a new sparse weight matrix
    pub fn new_sparse(rows: usize, cols: usize) -> Self {
        Self {
            weights: Vec::new(),
            rows,
            cols,
            sparse_indices: Some(Vec::new()),
        }
    }
    
    /// Set weight at position (row, col)
    pub fn set_weight(&mut self, row: usize, col: usize, weight: T) -> EmbeddedResult<()> {
        if row >= self.rows || col >= self.cols {
            return Err(EmbeddedError::InvalidIndex);
        }
        
        if let Some(ref mut indices) = self.sparse_indices {
            // Sparse matrix
            let key = (row as u16, col as u16);
            
            // Check if entry already exists
            for (i, &(r, c)) in indices.iter().enumerate() {
                if r == row as u16 && c == col as u16 {
                    self.weights[i] = weight;
                    return Ok(());
                }
            }
            
            // Add new entry
            indices.push(key)
                .map_err(|_| EmbeddedError::BufferFull)?;
            self.weights.push(weight)
                .map_err(|_| EmbeddedError::BufferFull)?;
        } else {
            // Dense matrix
            let index = row * self.cols + col;
            if let Some(w) = self.weights.get_mut(index) {
                *w = weight;
            } else {
                return Err(EmbeddedError::InvalidIndex);
            }
        }
        
        Ok(())
    }
    
    /// Get weight at position (row, col)
    pub fn get_weight(&self, row: usize, col: usize) -> T {
        if row >= self.rows || col >= self.cols {
            return T::zero();
        }
        
        if let Some(ref indices) = self.sparse_indices {
            // Sparse matrix
            for (i, &(r, c)) in indices.iter().enumerate() {
                if r == row as u16 && c == col as u16 {
                    return self.weights[i];
                }
            }
            T::zero() // Not found in sparse matrix
        } else {
            // Dense matrix
            let index = row * self.cols + col;
            self.weights.get(index).copied().unwrap_or(T::zero())
        }
    }
    
    /// Matrix-vector multiplication
    pub fn multiply_vector(&self, input: &[T]) -> EmbeddedResult<Vec<T, 64>> {
        if input.len() != self.cols {
            return Err(EmbeddedError::InvalidConfiguration);
        }
        
        let mut result = Vec::new();
        for _ in 0..self.rows {
            result.push(T::zero())
                .map_err(|_| EmbeddedError::BufferFull)?;
        }
        
        if let Some(ref indices) = self.sparse_indices {
            // Sparse multiplication
            for (i, &(row, col)) in indices.iter().enumerate() {
                let weight = self.weights[i];
                let input_val = input[col as usize];
                result[row as usize] = result[row as usize] + weight * input_val;
            }
        } else {
            // Dense multiplication
            for row in 0..self.rows {
                for col in 0..self.cols {
                    let weight = self.get_weight(row, col);
                    result[row] = result[row] + weight * input[col];
                }
            }
        }
        
        Ok(result)
    }
    
    /// Get matrix dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    
    /// Check if matrix is sparse
    pub fn is_sparse(&self) -> bool {
        self.sparse_indices.is_some()
    }
    
    /// Get sparsity ratio (0.0 = dense, 1.0 = completely sparse)
    pub fn sparsity_ratio(&self) -> f32 {
        if let Some(ref indices) = self.sparse_indices {
            let total_elements = self.rows * self.cols;
            let non_zero_elements = indices.len();
            1.0 - (non_zero_elements as f32 / total_elements as f32)
        } else {
            0.0 // Dense matrix
        }
    }
}

/// Ring buffer for time-series data in embedded systems
#[derive(Debug)]
pub struct EmbeddedTimeSeriesBuffer<T: FixedPoint> {
    /// Data storage
    data: [T; 128], // Fixed-size buffer for time series
    /// Timestamps
    timestamps: [T; 128],
    /// Current position
    position: usize,
    /// Number of valid entries
    count: usize,
    /// Total samples processed
    total_samples: u32,
}

impl<T: FixedPoint> EmbeddedTimeSeriesBuffer<T> {
    /// Create a new time series buffer
    pub fn new() -> Self {
        Self {
            data: [T::zero(); 128],
            timestamps: [T::zero(); 128],
            position: 0,
            count: 0,
            total_samples: 0,
        }
    }
    
    /// Add a data point
    pub fn add_point(&mut self, value: T, timestamp: T) {
        self.data[self.position] = value;
        self.timestamps[self.position] = timestamp;
        
        self.position = (self.position + 1) % 128;
        if self.count < 128 {
            self.count += 1;
        }
        self.total_samples += 1;
    }
    
    /// Get the most recent value
    pub fn latest_value(&self) -> Option<(T, T)> {
        if self.count == 0 {
            None
        } else {
            let idx = if self.position == 0 { 127 } else { self.position - 1 };
            Some((self.data[idx], self.timestamps[idx]))
        }
    }
    
    /// Calculate moving average over the last n points
    pub fn moving_average(&self, n: usize) -> T {
        if self.count == 0 {
            return T::zero();
        }
        
        let samples = n.min(self.count);
        let mut sum = T::zero();
        
        for i in 0..samples {
            let idx = if self.position >= i + 1 {
                self.position - i - 1
            } else {
                128 + self.position - i - 1
            };
            sum = sum + self.data[idx];
        }
        
        sum / T::from_int(samples as i32)
    }
    
    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        128
    }
    
    /// Get current count
    pub fn len(&self) -> usize {
        self.count
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Clear the buffer
    pub fn clear(&mut self) {
        self.position = 0;
        self.count = 0;
        self.total_samples = 0;
    }
}

impl<T: FixedPoint> Default for EmbeddedTimeSeriesBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_embedded_hypergraph() {
        let mut graph = EmbeddedHypergraph::<Q16_16>::new();
        
        let nodes = [0, 1, 2];
        let weight = Q16_16::from_float(0.5);
        let edge_id = graph.add_hyperedge(&nodes, weight, HyperedgeType::Excitatory).unwrap();
        
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.get_edge_weight(edge_id).unwrap(), weight);
    }
    
    #[test]
    fn test_spike_buffer() {
        let mut buffer = EmbeddedSpikeBuffer::<Q16_16>::new();
        
        let spike = FixedSpike::new(0, Q16_16::from_float(1.0), Q16_16::one());
        buffer.add_spike(spike).unwrap();
        
        assert!(!buffer.is_empty());
        assert_eq!(buffer.get_stats().current_count, 1);
        
        let retrieved = buffer.get_spike().unwrap();
        assert_eq!(retrieved.source, spike.source);
    }
    
    #[test]
    fn test_weight_matrix_dense() {
        let mut matrix = EmbeddedWeightMatrix::<Q16_16>::new_dense(
            3, 3, Q16_16::zero()
        ).unwrap();
        
        let weight = Q16_16::from_float(0.7);
        matrix.set_weight(1, 2, weight).unwrap();
        
        assert_eq!(matrix.get_weight(1, 2), weight);
        assert_eq!(matrix.get_weight(0, 0), Q16_16::zero());
    }
    
    #[test]
    fn test_weight_matrix_sparse() {
        let mut matrix = EmbeddedWeightMatrix::<Q16_16>::new_sparse(10, 10);
        
        let weight = Q16_16::from_float(0.5);
        matrix.set_weight(5, 7, weight).unwrap();
        
        assert_eq!(matrix.get_weight(5, 7), weight);
        assert_eq!(matrix.get_weight(0, 0), Q16_16::zero());
        assert!(matrix.is_sparse());
    }
    
    #[test]
    fn test_time_series_buffer() {
        let mut buffer = EmbeddedTimeSeriesBuffer::<Q16_16>::new();
        
        buffer.add_point(Q16_16::from_float(1.0), Q16_16::from_float(0.1));
        buffer.add_point(Q16_16::from_float(2.0), Q16_16::from_float(0.2));
        buffer.add_point(Q16_16::from_float(3.0), Q16_16::from_float(0.3));
        
        assert_eq!(buffer.len(), 3);
        
        let avg = buffer.moving_average(3);
        let expected = Q16_16::from_float(2.0); // (1+2+3)/3
        assert!((avg.to_float() - expected.to_float()).abs() < 0.01);
    }
}