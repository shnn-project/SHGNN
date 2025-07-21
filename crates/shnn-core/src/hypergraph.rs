//! Hypergraph data structures for multi-synaptic neural connections
//!
//! This module implements hypergraph-based neural connectivity that enables
//! multiple neurons to participate in single synaptic events, going beyond
//! traditional pairwise connections.

use crate::{
    error::{Result, SHNNError, HypergraphErrorKind},
    spike::{NeuronId, Spike, SpikeTarget},
    time::Time,
};
use core::fmt;

#[cfg(feature = "std")]
use std::collections::{HashMap, HashSet, BTreeMap};

#[cfg(not(feature = "std"))]
use heapless::{FnvIndexMap as HashMap, FnvIndexSet as HashSet};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Unique identifier for a hyperedge
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HyperedgeId(pub u32);

impl HyperedgeId {
    /// Create a new hyperedge ID
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
    
    /// Get the raw ID value
    pub const fn raw(&self) -> u32 {
        self.0
    }
    
    /// Invalid hyperedge ID constant
    pub const INVALID: Self = Self(u32::MAX);
    
    /// Check if this is a valid hyperedge ID
    pub const fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }
}

impl fmt::Display for HyperedgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{}", self.0)
    }
}

impl From<u32> for HyperedgeId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<HyperedgeId> for u32 {
    fn from(id: HyperedgeId) -> Self {
        id.0
    }
}

/// Weight function for hyperedge connections
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum WeightFunction {
    /// Uniform weight for all connections
    Uniform(f32),
    /// Individual weights per source-target pair
    Individual(Vec<f32>),
    /// Distance-based weight function
    DistanceBased {
        base_weight: f32,
        decay_constant: f32,
    },
    /// Custom weight computation (not serializable)
    #[cfg_attr(feature = "serde", serde(skip))]
    Custom(fn(&[NeuronId], &[NeuronId]) -> Vec<f32>),
}

impl WeightFunction {
    /// Compute weights for given source and target neurons
    pub fn compute_weights(&self, sources: &[NeuronId], targets: &[NeuronId]) -> Vec<f32> {
        match self {
            Self::Uniform(weight) => vec![*weight; sources.len() * targets.len()],
            Self::Individual(weights) => {
                let expected_len = sources.len() * targets.len();
                if weights.len() == expected_len {
                    weights.clone()
                } else {
                    // Fallback to uniform if sizes don't match
                    vec![1.0; expected_len]
                }
            }
            Self::DistanceBased { base_weight, decay_constant } => {
                let mut weights = Vec::with_capacity(sources.len() * targets.len());
                for source in sources {
                    for target in targets {
                        let distance = (source.raw() as f32 - target.raw() as f32).abs();
                        let weight = base_weight * (-distance * decay_constant).exp();
                        weights.push(weight);
                    }
                }
                weights
            }
            Self::Custom(func) => func(sources, targets),
        }
    }
    
    /// Get the default weight value
    pub fn default_weight(&self) -> f32 {
        match self {
            Self::Uniform(weight) => *weight,
            Self::Individual(weights) => weights.first().copied().unwrap_or(1.0),
            Self::DistanceBased { base_weight, .. } => *base_weight,
            Self::Custom(_) => 1.0,
        }
    }
}

impl Default for WeightFunction {
    fn default() -> Self {
        Self::Uniform(1.0)
    }
}

/// Hyperedge connection type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HyperedgeType {
    /// One-to-many: single source to multiple targets
    OneToMany,
    /// Many-to-one: multiple sources to single target
    ManyToOne,
    /// Many-to-many: multiple sources to multiple targets
    ManyToMany,
    /// All-to-all: fully connected within the hyperedge
    AllToAll,
    /// Custom connection pattern
    Custom,
}

impl fmt::Display for HyperedgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OneToMany => write!(f, "1→N"),
            Self::ManyToOne => write!(f, "N→1"),
            Self::ManyToMany => write!(f, "N→M"),
            Self::AllToAll => write!(f, "All↔All"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// A hyperedge representing multi-synaptic connections
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Hyperedge {
    /// Unique identifier
    pub id: HyperedgeId,
    /// Source neurons (presynaptic)
    pub sources: Vec<NeuronId>,
    /// Target neurons (postsynaptic)
    pub targets: Vec<NeuronId>,
    /// Connection type
    pub edge_type: HyperedgeType,
    /// Weight function
    pub weight_function: WeightFunction,
    /// Transmission delay in time units
    pub delay: crate::time::Duration,
    /// Whether this hyperedge is active
    pub active: bool,
    /// Minimum number of simultaneous spikes required for activation
    pub min_spikes: u32,
    /// Time window for spike coincidence detection
    pub coincidence_window: Time,
    /// Plasticity enabled
    pub plastic: bool,
}

impl Hyperedge {
    /// Create a new hyperedge
    pub fn new(
        id: HyperedgeId,
        sources: Vec<NeuronId>,
        targets: Vec<NeuronId>,
        edge_type: HyperedgeType,
    ) -> Result<Self> {
        if sources.is_empty() {
            return Err(SHNNError::hypergraph_error(
                HypergraphErrorKind::InvalidHyperedge,
                "Hyperedge must have at least one source"
            ));
        }
        
        if targets.is_empty() {
            return Err(SHNNError::hypergraph_error(
                HypergraphErrorKind::InvalidHyperedge,
                "Hyperedge must have at least one target"
            ));
        }
        
        // Validate edge type consistency
        match edge_type {
            HyperedgeType::OneToMany => {
                if sources.len() != 1 {
                    return Err(SHNNError::hypergraph_error(
                        HypergraphErrorKind::InvalidHyperedge,
                        "OneToMany hyperedge must have exactly one source"
                    ));
                }
            }
            HyperedgeType::ManyToOne => {
                if targets.len() != 1 {
                    return Err(SHNNError::hypergraph_error(
                        HypergraphErrorKind::InvalidHyperedge,
                        "ManyToOne hyperedge must have exactly one target"
                    ));
                }
            }
            _ => {} // Other types are flexible
        }
        
        Ok(Self {
            id,
            sources,
            targets,
            edge_type,
            weight_function: WeightFunction::default(),
            delay: crate::time::Duration::ZERO,
            active: true,
            min_spikes: 1,
            coincidence_window: Time::from_millis(1), // 1ms default
            plastic: true,
        })
    }
    
    /// Create a simple pairwise connection (traditional synapse)
    pub fn pairwise(
        id: HyperedgeId,
        source: NeuronId,
        target: NeuronId,
        weight: f32,
    ) -> Result<Self> {
        let mut edge = Self::new(
            id,
            vec![source],
            vec![target],
            HyperedgeType::OneToMany,
        )?;
        edge.weight_function = WeightFunction::Uniform(weight);
        Ok(edge)
    }
    
    /// Create a convergent hyperedge (multiple sources to one target)
    pub fn convergent(
        id: HyperedgeId,
        sources: Vec<NeuronId>,
        target: NeuronId,
    ) -> Result<Self> {
        Self::new(id, sources, vec![target], HyperedgeType::ManyToOne)
    }
    
    /// Create a divergent hyperedge (one source to multiple targets)
    pub fn divergent(
        id: HyperedgeId,
        source: NeuronId,
        targets: Vec<NeuronId>,
    ) -> Result<Self> {
        Self::new(id, vec![source], targets, HyperedgeType::OneToMany)
    }
    
    /// Create a many-to-many hyperedge
    pub fn many_to_many(
        id: HyperedgeId,
        sources: Vec<NeuronId>,
        targets: Vec<NeuronId>,
    ) -> Result<Self> {
        Self::new(id, sources, targets, HyperedgeType::ManyToMany)
    }
    
    /// Set the weight function
    pub fn with_weights(mut self, weight_function: WeightFunction) -> Self {
        self.weight_function = weight_function;
        self
    }
    
    /// Set the transmission delay
    pub fn with_delay(mut self, delay: crate::time::Duration) -> Self {
        self.delay = delay;
        self
    }
    
    /// Set minimum spikes required for activation
    pub fn with_min_spikes(mut self, min_spikes: u32) -> Self {
        self.min_spikes = min_spikes;
        self
    }
    
    /// Set coincidence detection window
    pub fn with_coincidence_window(mut self, window: Time) -> Self {
        self.coincidence_window = window;
        self
    }
    
    /// Check if a neuron is a source in this hyperedge
    pub fn has_source(&self, neuron_id: NeuronId) -> bool {
        self.sources.contains(&neuron_id)
    }
    
    /// Check if a neuron is a target in this hyperedge
    pub fn has_target(&self, neuron_id: NeuronId) -> bool {
        self.targets.contains(&neuron_id)
    }
    
    /// Check if this hyperedge connects two neurons
    pub fn connects(&self, source: NeuronId, target: NeuronId) -> bool {
        self.has_source(source) && self.has_target(target)
    }
    
    /// Get all neurons involved in this hyperedge
    pub fn all_neurons(&self) -> Vec<NeuronId> {
        let mut neurons = self.sources.clone();
        neurons.extend(&self.targets);
        neurons.sort();
        neurons.dedup();
        neurons
    }
    
    /// Compute connection weights for this hyperedge
    pub fn compute_weights(&self) -> Vec<f32> {
        self.weight_function.compute_weights(&self.sources, &self.targets)
    }
    
    /// Get the number of potential connections
    pub fn connection_count(&self) -> usize {
        match self.edge_type {
            HyperedgeType::OneToMany => self.targets.len(),
            HyperedgeType::ManyToOne => self.sources.len(),
            HyperedgeType::ManyToMany => self.sources.len() * self.targets.len(),
            HyperedgeType::AllToAll => {
                let total = self.sources.len() + self.targets.len();
                total * (total - 1) // Fully connected, excluding self-loops
            }
            HyperedgeType::Custom => self.sources.len() * self.targets.len(), // Conservative estimate
        }
    }
}

impl fmt::Display for Hyperedge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Hyperedge({}, {} [{}] → {} [{}], delay={})",
            self.id,
            self.sources.len(),
            self.edge_type,
            self.targets.len(),
            if self.active { "active" } else { "inactive" },
            self.delay
        )
    }
}

/// Spike routing information for hyperedge processing
#[derive(Debug, Clone, PartialEq)]
pub struct SpikeRoute {
    /// Source hyperedge
    pub hyperedge_id: HyperedgeId,
    /// Target neurons for this route
    pub targets: Vec<NeuronId>,
    /// Weights for each target
    pub weights: Vec<f32>,
    /// Delivery time
    pub delivery_time: Time,
}

/// Hypergraph network structure
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HypergraphNetwork {
    /// All hyperedges in the network
    hyperedges: Vec<Option<Hyperedge>>,
    /// Mapping from source neurons to hyperedge IDs
    #[cfg(feature = "std")]
    source_map: HashMap<NeuronId, Vec<HyperedgeId>>,
    /// Mapping from target neurons to hyperedge IDs
    #[cfg(feature = "std")]
    target_map: HashMap<NeuronId, Vec<HyperedgeId>>,
    /// Maximum hyperedge ID
    max_edge_id: u32,
    /// Network statistics
    stats: NetworkStats,
}

/// Network statistics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NetworkStats {
    /// Total number of hyperedges
    pub edge_count: usize,
    /// Total number of neurons
    pub neuron_count: usize,
    /// Total number of connections
    pub connection_count: usize,
    /// Average degree (connections per neuron)
    pub average_degree: f32,
    /// Maximum degree
    pub max_degree: u32,
}

impl HypergraphNetwork {
    /// Create a new empty hypergraph network
    pub fn new() -> Self {
        Self {
            hyperedges: Vec::new(),
            #[cfg(feature = "std")]
            source_map: HashMap::new(),
            #[cfg(feature = "std")]
            target_map: HashMap::new(),
            max_edge_id: 0,
            stats: NetworkStats::default(),
        }
    }
    
    /// Create with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            hyperedges: Vec::with_capacity(capacity),
            #[cfg(feature = "std")]
            source_map: HashMap::with_capacity(capacity * 2),
            #[cfg(feature = "std")]
            target_map: HashMap::with_capacity(capacity * 2),
            max_edge_id: 0,
            stats: NetworkStats::default(),
        }
    }
    
    /// Add a hyperedge to the network
    pub fn add_hyperedge(&mut self, hyperedge: Hyperedge) -> Result<()> {
        let id = hyperedge.id.raw();
        
        // Expand vector if necessary
        if id as usize >= self.hyperedges.len() {
            self.hyperedges.resize(id as usize + 1, None);
        }
        
        if self.hyperedges[id as usize].is_some() {
            return Err(SHNNError::hypergraph_error(
                HypergraphErrorKind::InvalidHyperedge,
                "Hyperedge ID already exists"
            ));
        }
        
        // Update mappings
        #[cfg(feature = "std")]
        {
            for &source in &hyperedge.sources {
                self.source_map.entry(source).or_insert_with(Vec::new).push(hyperedge.id);
            }
            
            for &target in &hyperedge.targets {
                self.target_map.entry(target).or_insert_with(Vec::new).push(hyperedge.id);
            }
        }
        
        self.hyperedges[id as usize] = Some(hyperedge);
        self.max_edge_id = self.max_edge_id.max(id);
        
        self.update_stats();
        
        Ok(())
    }
    
    /// Get a hyperedge by ID
    pub fn get_hyperedge(&self, id: HyperedgeId) -> Option<&Hyperedge> {
        let idx = id.raw() as usize;
        if idx < self.hyperedges.len() {
            self.hyperedges[idx].as_ref()
        } else {
            None
        }
    }
    
    /// Get a mutable hyperedge by ID
    pub fn get_hyperedge_mut(&mut self, id: HyperedgeId) -> Option<&mut Hyperedge> {
        let idx = id.raw() as usize;
        if idx < self.hyperedges.len() {
            self.hyperedges[idx].as_mut()
        } else {
            None
        }
    }
    
    /// Remove a hyperedge from the network
    pub fn remove_hyperedge(&mut self, id: HyperedgeId) -> Option<Hyperedge> {
        let idx = id.raw() as usize;
        if idx < self.hyperedges.len() {
            let hyperedge = self.hyperedges[idx].take();
            
            if let Some(ref edge) = hyperedge {
                // Update mappings
                #[cfg(feature = "std")]
                {
                    for &source in &edge.sources {
                        if let Some(edges) = self.source_map.get_mut(&source) {
                            edges.retain(|&eid| eid != id);
                            if edges.is_empty() {
                                self.source_map.remove(&source);
                            }
                        }
                    }
                    
                    for &target in &edge.targets {
                        if let Some(edges) = self.target_map.get_mut(&target) {
                            edges.retain(|&eid| eid != id);
                            if edges.is_empty() {
                                self.target_map.remove(&target);
                            }
                        }
                    }
                }
                
                self.update_stats();
            }
            
            hyperedge
        } else {
            None
        }
    }
    
    /// Get all hyperedges involving a neuron as source
    #[cfg(feature = "std")]
    pub fn get_source_hyperedges(&self, neuron_id: NeuronId) -> Vec<&Hyperedge> {
        if let Some(edge_ids) = self.source_map.get(&neuron_id) {
            edge_ids.iter()
                .filter_map(|&id| self.get_hyperedge(id))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get all hyperedges involving a neuron as target
    #[cfg(feature = "std")]
    pub fn get_target_hyperedges(&self, neuron_id: NeuronId) -> Vec<&Hyperedge> {
        if let Some(edge_ids) = self.target_map.get(&neuron_id) {
            edge_ids.iter()
                .filter_map(|&id| self.get_hyperedge(id))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Route a spike through the hypergraph
    pub fn route_spike(&self, spike: &Spike, current_time: Time) -> Vec<SpikeRoute> {
        let mut routes = Vec::new();
        
        #[cfg(feature = "std")]
        {
            if let Some(edge_ids) = self.source_map.get(&spike.source) {
                for &edge_id in edge_ids {
                    if let Some(hyperedge) = self.get_hyperedge(edge_id) {
                        if hyperedge.active {
                            let weights = hyperedge.compute_weights();
                            let delivery_time = current_time + hyperedge.delay;
                            
                            routes.push(SpikeRoute {
                                hyperedge_id: edge_id,
                                targets: hyperedge.targets.clone(),
                                weights,
                                delivery_time,
                            });
                        }
                    }
                }
            }
        }
        
        #[cfg(not(feature = "std"))]
        {
            // Fallback for no-std: iterate through all hyperedges
            for hyperedge_opt in &self.hyperedges {
                if let Some(hyperedge) = hyperedge_opt {
                    if hyperedge.active && hyperedge.has_source(spike.source) {
                        let weights = hyperedge.compute_weights();
                        let delivery_time = current_time + hyperedge.delay;
                        
                        routes.push(SpikeRoute {
                            hyperedge_id: hyperedge.id,
                            targets: hyperedge.targets.clone(),
                            weights,
                            delivery_time,
                        });
                    }
                }
            }
        }
        
        routes
    }
    
    /// Get all hyperedge IDs
    pub fn hyperedge_ids(&self) -> Vec<HyperedgeId> {
        let mut ids = Vec::new();
        for (i, hyperedge) in self.hyperedges.iter().enumerate() {
            if hyperedge.is_some() {
                ids.push(HyperedgeId::new(i as u32));
            }
        }
        ids
    }
    
    /// Get all unique neurons in the network
    pub fn all_neurons(&self) -> Vec<NeuronId> {
        #[cfg(feature = "std")]
        {
            let mut neurons: HashSet<NeuronId> = HashSet::new();
            for hyperedge_opt in &self.hyperedges {
                if let Some(hyperedge) = hyperedge_opt {
                    neurons.extend(&hyperedge.sources);
                    neurons.extend(&hyperedge.targets);
                }
            }
            let mut result: Vec<NeuronId> = neurons.into_iter().collect();
            result.sort();
            result
        }
        
        #[cfg(not(feature = "std"))]
        {
            let mut neurons = Vec::new();
            for hyperedge_opt in &self.hyperedges {
                if let Some(hyperedge) = hyperedge_opt {
                    neurons.extend(&hyperedge.sources);
                    neurons.extend(&hyperedge.targets);
                }
            }
            neurons.sort();
            neurons.dedup();
            neurons
        }
    }
    
    /// Update network statistics
    fn update_stats(&mut self) {
        let active_hyperedges: Vec<_> = self.hyperedges.iter()
            .filter_map(|h| h.as_ref())
            .collect();
        
        self.stats.edge_count = active_hyperedges.len();
        
        let all_neurons = self.all_neurons();
        self.stats.neuron_count = all_neurons.len();
        
        self.stats.connection_count = active_hyperedges.iter()
            .map(|h| h.connection_count())
            .sum();
        
        if self.stats.neuron_count > 0 {
            self.stats.average_degree = self.stats.connection_count as f32 / self.stats.neuron_count as f32;
        }
        
        // Calculate max degree
        #[cfg(feature = "std")]
        {
            let mut degree_counts: HashMap<NeuronId, u32> = HashMap::new();
            for hyperedge in &active_hyperedges {
                for &neuron in &hyperedge.sources {
                    *degree_counts.entry(neuron).or_insert(0) += hyperedge.targets.len() as u32;
                }
                for &neuron in &hyperedge.targets {
                    *degree_counts.entry(neuron).or_insert(0) += hyperedge.sources.len() as u32;
                }
            }
            self.stats.max_degree = degree_counts.values().max().copied().unwrap_or(0);
        }
        
        #[cfg(not(feature = "std"))]
        {
            // Simplified calculation for no-std
            self.stats.max_degree = if self.stats.neuron_count > 0 {
                (self.stats.connection_count / self.stats.neuron_count) as u32 * 2
            } else {
                0
            };
        }
    }
    
    /// Get network statistics
    pub fn stats(&self) -> &NetworkStats {
        &self.stats
    }
    
    /// Check if the network has cycles (simplified check)
    pub fn has_cycles(&self) -> bool {
        // Simplified cycle detection - in practice this would be more sophisticated
        for hyperedge_opt in &self.hyperedges {
            if let Some(hyperedge) = hyperedge_opt {
                for &source in &hyperedge.sources {
                    if hyperedge.targets.contains(&source) {
                        return true; // Self-loop detected
                    }
                }
            }
        }
        false
    }
    
    /// Validate network structure
    pub fn validate(&self) -> Result<()> {
        for hyperedge_opt in &self.hyperedges {
            if let Some(hyperedge) = hyperedge_opt {
                // Check for empty source/target lists
                if hyperedge.sources.is_empty() || hyperedge.targets.is_empty() {
                    return Err(SHNNError::hypergraph_error(
                        HypergraphErrorKind::InvalidHyperedge,
                        "Hyperedge has empty source or target list"
                    ));
                }
                
                // Check for duplicate neurons in same hyperedge
                let mut all_neurons = hyperedge.sources.clone();
                all_neurons.extend(&hyperedge.targets);
                let original_len = all_neurons.len();
                all_neurons.sort();
                all_neurons.dedup();
                
                if all_neurons.len() != original_len {
                    // This is actually allowed in hypergraphs - neurons can be both source and target
                }
            }
        }
        
        Ok(())
    }
}

impl Default for HypergraphNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for HypergraphNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HypergraphNetwork({} edges, {} neurons, {} connections, avg_degree={:.1})",
            self.stats.edge_count,
            self.stats.neuron_count,
            self.stats.connection_count,
            self.stats.average_degree
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hyperedge_creation() {
        let edge = Hyperedge::pairwise(
            HyperedgeId::new(0),
            NeuronId::new(1),
            NeuronId::new(2),
            0.5
        ).unwrap();
        
        assert_eq!(edge.sources.len(), 1);
        assert_eq!(edge.targets.len(), 1);
        assert!(edge.connects(NeuronId::new(1), NeuronId::new(2)));
    }
    
    #[test]
    fn test_hyperedge_weights() {
        let edge = Hyperedge::convergent(
            HyperedgeId::new(0),
            vec![NeuronId::new(1), NeuronId::new(2)],
            NeuronId::new(3)
        ).unwrap()
        .with_weights(WeightFunction::Uniform(0.8));
        
        let weights = edge.compute_weights();
        assert_eq!(weights.len(), 2); // 2 sources * 1 target
        assert!(weights.iter().all(|&w| (w - 0.8).abs() < f32::EPSILON));
    }
    
    #[test]
    fn test_hypergraph_network() {
        let mut network = HypergraphNetwork::new();
        
        let edge = Hyperedge::pairwise(
            HyperedgeId::new(0),
            NeuronId::new(1),
            NeuronId::new(2),
            1.0
        ).unwrap();
        
        network.add_hyperedge(edge).unwrap();
        
        assert_eq!(network.stats().edge_count, 1);
        assert_eq!(network.stats().neuron_count, 2);
        assert!(network.get_hyperedge(HyperedgeId::new(0)).is_some());
    }
    
    #[test]
    fn test_spike_routing() {
        let mut network = HypergraphNetwork::new();
        
        let edge = Hyperedge::divergent(
            HyperedgeId::new(0),
            NeuronId::new(1),
            vec![NeuronId::new(2), NeuronId::new(3)]
        ).unwrap()
        .with_delay(crate::time::Duration::from_millis(1));
        
        network.add_hyperedge(edge).unwrap();
        
        let spike = Spike::binary(NeuronId::new(1), Time::from_millis(10)).unwrap();
        let routes = network.route_spike(&spike, Time::from_millis(10));
        
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].targets.len(), 2);
        assert_eq!(routes[0].delivery_time, Time::from_millis(11));
    }
    
    #[test]
    fn test_network_validation() {
        let network = HypergraphNetwork::new();
        assert!(network.validate().is_ok());
        
        // Test with invalid hyperedge would require manually constructing invalid state
    }
}