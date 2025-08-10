# SHNN Hypergraph Modularization Plan

## Overview

This document outlines the plan to modularize the SHNN project to separate hypergraph-specific functionality from generic spiking neural network primitives, enabling users to employ any data structure for neural connectivity.

## Current State Analysis

### Hypergraph-Specific Components (843 lines, 4.2% of core code)

**📁 [`crates/shnn-core/src/hypergraph.rs`](crates/shnn-core/src/hypergraph.rs)**
- **Lines**: 843 (100% hypergraph-specific)
- **Core Types**:
  - [`HyperedgeId`](crates/shnn-core/src/hypergraph.rs:24-46) - Multi-synaptic connection identifiers
  - [`WeightFunction`](crates/shnn-core/src/hypergraph.rs:66-124) - Multi-connection weight computation
  - [`HyperedgeType`](crates/shnn-core/src/hypergraph.rs:132-158) - Connection pattern types
  - [`Hyperedge`](crates/shnn-core/src/hypergraph.rs:160-353) - Multi-synaptic connection structure
  - [`HypergraphNetwork`](crates/shnn-core/src/hypergraph.rs:383-763) - Hypergraph-based network

### Generic Neuromorphic Components (19,000+ lines, 95.8% of core code)

**All already data-structure agnostic:**
- **Neuron Models** ([`neuron.rs`](crates/shnn-core/src/neuron.rs), 786 lines) - LIF, AdEx, Izhikevich
- **Spike Processing** ([`spike.rs`](crates/shnn-core/src/spike.rs), 557 lines) - Core spike types and operations
- **Plasticity Mechanisms** ([`plasticity.rs`](crates/shnn-core/src/plasticity.rs), 697 lines) - STDP, homeostatic learning
- **Spike Encoding** ([`encoding.rs`](crates/shnn-core/src/encoding.rs), 841 lines) - Rate, temporal, population encoding
- **Supporting Infrastructure** - Math, async runtime, concurrency, serialization

## Modularization Strategy

### Phase 1: Connectivity Abstraction Layer

**Goal**: Create a generic trait for network connectivity that abstracts the data structure

#### 1.1 Create NetworkConnectivity Trait

```rust
// crates/shnn-core/src/connectivity/mod.rs
pub trait NetworkConnectivity<NodeId> {
    type ConnectionId;
    type RouteInfo;
    type Error;
    
    /// Route a spike through the connectivity structure
    fn route_spike(&self, spike: &Spike, current_time: Time) -> Result<Vec<SpikeRoute>, Self::Error>;
    
    /// Get all target neurons for a given source
    fn get_targets(&self, source: NodeId) -> Result<Vec<NodeId>, Self::Error>;
    
    /// Get all source neurons for a given target  
    fn get_sources(&self, target: NodeId) -> Result<Vec<NodeId>, Self::Error>;
    
    /// Add a connection to the network
    fn add_connection(&mut self, connection: Self::ConnectionId) -> Result<(), Self::Error>;
    
    /// Remove a connection from the network
    fn remove_connection(&mut self, connection: Self::ConnectionId) -> Result<Option<Self::RouteInfo>, Self::Error>;
    
    /// Get network statistics
    fn get_stats(&self) -> ConnectivityStats;
    
    /// Validate connectivity structure
    fn validate(&self) -> Result<(), Self::Error>;
}
```

#### 1.2 Define Common Types

```rust
// crates/shnn-core/src/connectivity/types.rs
pub struct SpikeRoute {
    pub source_connection: ConnectionId,
    pub targets: Vec<NeuronId>,
    pub weights: Vec<f32>,
    pub delivery_time: Time,
}

pub struct ConnectivityStats {
    pub connection_count: usize,
    pub node_count: usize,
    pub average_degree: f32,
    pub max_degree: u32,
}
```

### Phase 2: Implement Connectivity for Multiple Data Structures

#### 2.1 Hypergraph Implementation (Preserve Existing)

```rust
// crates/shnn-core/src/connectivity/hypergraph.rs
impl NetworkConnectivity<NeuronId> for HypergraphNetwork {
    type ConnectionId = HyperedgeId;
    type RouteInfo = HypergraphRoute;
    type Error = HypergraphError;
    
    fn route_spike(&self, spike: &Spike, current_time: Time) -> Result<Vec<SpikeRoute>, Self::Error> {
        // Existing hypergraph routing logic
        self.route_spike(spike, current_time)
    }
    
    // ... implement other trait methods
}
```

#### 2.2 Traditional Graph Implementation

```rust
// crates/shnn-core/src/connectivity/graph.rs
pub struct GraphNetwork {
    edges: Vec<Edge>,
    adjacency_map: HashMap<NeuronId, Vec<NeuronId>>,
    weights: HashMap<(NeuronId, NeuronId), f32>,
}

impl NetworkConnectivity<NeuronId> for GraphNetwork {
    type ConnectionId = EdgeId;
    type RouteInfo = EdgeRoute;
    type Error = GraphError;
    
    fn route_spike(&self, spike: &Spike, current_time: Time) -> Result<Vec<SpikeRoute>, Self::Error> {
        // Traditional graph routing logic
        let targets = self.adjacency_map.get(&spike.source).unwrap_or(&vec![]);
        let routes = targets.iter().map(|&target| {
            let weight = self.weights.get(&(spike.source, target)).copied().unwrap_or(1.0);
            SpikeRoute {
                source_connection: EdgeId::new(spike.source, target),
                targets: vec![target],
                weights: vec![weight],
                delivery_time: current_time,
            }
        }).collect();
        Ok(routes)
    }
    
    // ... implement other trait methods
}
```

#### 2.3 Matrix Implementation (for Dense Networks)

```rust
// crates/shnn-core/src/connectivity/matrix.rs
pub struct MatrixNetwork {
    adjacency_matrix: Vec<Vec<f32>>,
    size: usize,
}

impl NetworkConnectivity<NeuronId> for MatrixNetwork {
    type ConnectionId = (usize, usize);
    type RouteInfo = MatrixRoute;
    type Error = MatrixError;
    
    fn route_spike(&self, spike: &Spike, current_time: Time) -> Result<Vec<SpikeRoute>, Self::Error> {
        let source_idx = spike.source.raw() as usize;
        if source_idx >= self.size {
            return Err(MatrixError::InvalidNeuronId(spike.source));
        }
        
        let mut routes = Vec::new();
        for (target_idx, &weight) in self.adjacency_matrix[source_idx].iter().enumerate() {
            if weight > 0.0 {
                routes.push(SpikeRoute {
                    source_connection: (source_idx, target_idx),
                    targets: vec![NeuronId::new(target_idx as u32)],
                    weights: vec![weight],
                    delivery_time: current_time,
                });
            }
        }
        Ok(routes)
    }
    
    // ... implement other trait methods
}
```

#### 2.4 Sparse Matrix Implementation (for Large Sparse Networks)

```rust
// crates/shnn-core/src/connectivity/sparse.rs
use shnn_math::SparseMatrix;

pub struct SparseMatrixNetwork {
    connectivity: SparseMatrix<f32>,
    neuron_count: usize,
}

impl NetworkConnectivity<NeuronId> for SparseMatrixNetwork {
    type ConnectionId = (usize, usize);
    type RouteInfo = SparseRoute;
    type Error = SparseError;
    
    fn route_spike(&self, spike: &Spike, current_time: Time) -> Result<Vec<SpikeRoute>, Self::Error> {
        let source_idx = spike.source.raw() as usize;
        let mut routes = Vec::new();
        
        // Use sparse matrix row iteration
        for (target_idx, weight) in self.connectivity.row_iter(source_idx) {
            if weight > 0.0 {
                routes.push(SpikeRoute {
                    source_connection: (source_idx, target_idx),
                    targets: vec![NeuronId::new(target_idx as u32)],
                    weights: vec![weight],
                    delivery_time: current_time,
                });
            }
        }
        Ok(routes)
    }
    
    // ... implement other trait methods
}
```

### Phase 3: Generic Network Container

#### 3.1 Create Generic SpikeNetwork

```rust
// crates/shnn-core/src/network/mod.rs
pub struct SpikeNetwork<C, N> 
where 
    C: NetworkConnectivity<NeuronId>,
    N: Neuron,
{
    /// Connectivity structure (hypergraph, graph, matrix, etc.)
    connectivity: C,
    
    /// Neuron collection
    neurons: NeuronPool<N>,
    
    /// Plasticity management
    plasticity: PlasticityManager,
    
    /// Spike encoding/decoding
    encoder: MultiModalEncoder,
    
    /// Runtime statistics
    stats: NetworkStats,
    
    /// Current simulation time
    current_time: Time,
}

impl<C, N> SpikeNetwork<C, N> 
where 
    C: NetworkConnectivity<NeuronId>,
    N: Neuron,
{
    /// Create a new network with specified connectivity and neurons
    pub fn new(
        connectivity: C,
        neurons: NeuronPool<N>,
        plasticity: PlasticityManager,
        encoder: MultiModalEncoder,
    ) -> Self {
        Self {
            connectivity,
            neurons,
            plasticity,
            encoder,
            stats: NetworkStats::default(),
            current_time: Time::ZERO,
        }
    }
    
    /// Process input spikes through the network
    pub fn process_spikes(&mut self, input_spikes: &[Spike]) -> Result<Vec<Spike>, NetworkError> {
        let mut output_spikes = Vec::new();
        
        for spike in input_spikes {
            // Route spike through connectivity structure
            let routes = self.connectivity.route_spike(spike, self.current_time)?;
            
            for route in routes {
                // Apply weighted inputs to target neurons
                for (&target, &weight) in route.targets.iter().zip(route.weights.iter()) {
                    if let Some(neuron) = self.neurons.get_neuron_mut(target.into()) {
                        // Integrate weighted input
                        neuron.integrate(weight as f64, 1);
                        
                        // Check for output spike
                        if let Some(output_spike) = neuron.update(1) {
                            output_spikes.push(output_spike);
                            
                            // Apply plasticity if enabled
                            if self.plasticity.is_enabled() {
                                let new_weight = self.plasticity.process_spike_pair(
                                    target.into(),
                                    weight,
                                    spike.timestamp,
                                    output_spike.timestamp,
                                );
                                
                                // Update connection weight in connectivity structure
                                // This would require extending the NetworkConnectivity trait
                                // with a method to update weights
                            }
                        }
                    }
                }
            }
        }
        
        // Update statistics
        self.stats.total_spikes_processed += input_spikes.len();
        self.stats.total_spikes_generated += output_spikes.len();
        
        Ok(output_spikes)
    }
    
    /// Get network statistics
    pub fn get_stats(&self) -> &NetworkStats {
        &self.stats
    }
    
    /// Reset network state
    pub fn reset(&mut self) {
        self.neurons.reset_all();
        self.plasticity.reset();
        self.current_time = Time::ZERO;
        self.stats = NetworkStats::default();
    }
}
```

#### 3.2 Convenient Builder Pattern

```rust
// crates/shnn-core/src/network/builder.rs
pub struct NetworkBuilder<C> {
    connectivity: Option<C>,
    plasticity_config: PlasticityConfig,
    encoding_config: EncodingConfig,
}

impl<C> NetworkBuilder<C> 
where 
    C: NetworkConnectivity<NeuronId>,
{
    pub fn new() -> Self {
        Self {
            connectivity: None,
            plasticity_config: PlasticityConfig::default(),
            encoding_config: EncodingConfig::default(),
        }
    }
    
    pub fn with_connectivity(mut self, connectivity: C) -> Self {
        self.connectivity = Some(connectivity);
        self
    }
    
    pub fn with_plasticity(mut self, config: PlasticityConfig) -> Self {
        self.plasticity_config = config;
        self
    }
    
    pub fn build<N: Neuron>(self, neurons: NeuronPool<N>) -> Result<SpikeNetwork<C, N>, BuildError> {
        let connectivity = self.connectivity.ok_or(BuildError::MissingConnectivity)?;
        let plasticity = PlasticityManager::from_config(self.plasticity_config);
        let encoder = MultiModalEncoder::from_config(self.encoding_config);
        
        Ok(SpikeNetwork::new(connectivity, neurons, plasticity, encoder))
    }
}
```

### Phase 4: Update Existing APIs

#### 4.1 Preserve Backward Compatibility

```rust
// crates/shnn-core/src/lib.rs
pub mod connectivity {
    pub mod hypergraph;
    pub mod graph;
    pub mod matrix;
    pub mod sparse;
    pub use hypergraph::HypergraphNetwork; // Re-export for compatibility
}

pub mod network;

// Convenience type aliases for backward compatibility
pub type HypergraphSNN = SpikeNetwork<HypergraphNetwork, LIFNeuron>;
pub type GraphSNN = SpikeNetwork<GraphNetwork, LIFNeuron>;
pub type MatrixSNN = SpikeNetwork<MatrixNetwork, LIFNeuron>;

// Legacy API wrapper (deprecated)
#[deprecated(since = "0.2.0", note = "Use SpikeNetwork with specific connectivity instead")]
pub type LegacyHypergraphNetwork = HypergraphNetwork;
```

#### 4.2 Update Python Bindings

```rust
// crates/shnn-python/src/network.rs
#[pyclass(name = "Network")]
pub struct PyNetwork {
    // Support multiple connectivity types
    inner: NetworkType,
}

enum NetworkType {
    Hypergraph(SpikeNetwork<HypergraphNetwork, LIFNeuron>),
    Graph(SpikeNetwork<GraphNetwork, LIFNeuron>),
    Matrix(SpikeNetwork<MatrixNetwork, LIFNeuron>),
    Sparse(SpikeNetwork<SparseMatrixNetwork, LIFNeuron>),
}

#[pymethods]
impl PyNetwork {
    #[new]
    #[pyo3(signature = (connectivity_type="hypergraph", **kwargs))]
    fn new(connectivity_type: &str, kwargs: Option<&PyDict>) -> PyResult<Self> {
        let inner = match connectivity_type {
            "hypergraph" => NetworkType::Hypergraph(/* construct hypergraph network */),
            "graph" => NetworkType::Graph(/* construct graph network */),
            "matrix" => NetworkType::Matrix(/* construct matrix network */),
            "sparse" => NetworkType::Sparse(/* construct sparse network */),
            _ => return Err(PyValueError::new_err(format!("Unknown connectivity type: {}", connectivity_type))),
        };
        
        Ok(Self { inner })
    }
    
    /// Create a hypergraph-based network (backward compatibility)
    #[classmethod]
    fn hypergraph(_cls: &PyType, **kwargs) -> PyResult<Self> {
        Self::new("hypergraph", kwargs)
    }
    
    /// Create a traditional graph-based network
    #[classmethod]
    fn graph(_cls: &PyType, **kwargs) -> PyResult<Self> {
        Self::new("graph", kwargs)
    }
    
    /// Create a matrix-based network
    #[classmethod]
    fn matrix(_cls: &PyType, **kwargs) -> PyResult<Self> {
        Self::new("matrix", kwargs)
    }
}
```

## Implementation Timeline

### Phase 1: Foundation (2-3 weeks)
- [ ] Create `NetworkConnectivity` trait and common types
- [ ] Refactor `HypergraphNetwork` to implement the trait
- [ ] Update core processing loops to use trait methods
- [ ] Comprehensive testing of hypergraph compatibility

### Phase 2: Alternative Implementations (3-4 weeks)
- [ ] Implement `GraphNetwork` with traditional directed graph structure
- [ ] Implement `MatrixNetwork` for dense connectivity patterns
- [ ] Implement `SparseMatrixNetwork` for large sparse networks
- [ ] Performance benchmarking across implementations

### Phase 3: Generic Container (2-3 weeks)
- [ ] Create `SpikeNetwork<C, N>` generic container
- [ ] Implement builder pattern for easy construction
- [ ] Update plasticity integration to work with generic connectivity
- [ ] Migrate existing functionality to use generic interface

### Phase 4: API Updates (2-3 weeks)
- [ ] Update Python bindings to support multiple connectivity types
- [ ] Create migration guide and compatibility layer
- [ ] Update examples and documentation
- [ ] Performance optimization and final testing

### Phase 5: Documentation & Examples (1-2 weeks)
- [ ] Create comprehensive usage examples for each connectivity type
- [ ] Write performance comparison guide
- [ ] Create migration documentation
- [ ] Update README and API documentation

## Migration Guide

### For Existing Users

**Old API (deprecated but supported):**
```rust
let mut network = HypergraphNetwork::new();
network.add_hyperedge(hyperedge);
let output = network.process_spikes(input_spikes);
```

**New API (recommended):**
```rust
let connectivity = HypergraphNetwork::new();
let neurons = NeuronPool::with_neurons(lif_neurons);
let mut network = SpikeNetwork::new(connectivity, neurons, plasticity, encoder);
let output = network.process_spikes(input_spikes);
```

**Or using builder pattern:**
```rust
let network = NetworkBuilder::new()
    .with_connectivity(HypergraphNetwork::new())
    .with_plasticity(STDPConfig::default())
    .build(neuron_pool)?;
```

### For New Users

**Choose your connectivity structure:**

```rust
// Hypergraph (multi-synaptic connections)
let network = SpikeNetwork::new(
    HypergraphNetwork::new(),
    neuron_pool,
    plasticity,
    encoder,
);

// Traditional graph (pairwise connections)
let network = SpikeNetwork::new(
    GraphNetwork::new(), 
    neuron_pool,
    plasticity,
    encoder,
);

// Dense matrix (fully connected)
let network = SpikeNetwork::new(
    MatrixNetwork::with_size(1000),
    neuron_pool,
    plasticity,
    encoder,
);

// Sparse matrix (large sparse networks)
let network = SpikeNetwork::new(
    SparseMatrixNetwork::new(),
    neuron_pool,
    plasticity,
    encoder,
);
```

## Benefits

1. **🎯 Flexibility**: Choose optimal data structures for specific use cases
2. **⚡ Performance**: Specialized data structures for different connectivity patterns
3. **🔄 Compatibility**: Existing hypergraph functionality preserved
4. **🔧 Extensibility**: Easy to add new connectivity implementations
5. **📏 Scalability**: Better support for large-scale networks with appropriate data structures

## Testing Strategy

### Unit Tests
- [ ] Trait implementation compliance for each connectivity type
- [ ] Spike routing correctness across all implementations
- [ ] Weight update functionality
- [ ] Edge case handling

### Integration Tests  
- [ ] End-to-end spike processing with different connectivity types
- [ ] Plasticity interaction with various data structures
- [ ] Memory safety under concurrent access
- [ ] Serialization/deserialization compatibility

### Performance Tests
- [ ] Routing performance comparison across implementations
- [ ] Memory usage analysis for different network sizes
- [ ] Concurrent processing benchmarks
- [ ] Compilation time impact measurement

### Compatibility Tests
- [ ] Backward compatibility with existing APIs
- [ ] Python binding functionality across connectivity types
- [ ] Example code validation
- [ ] Migration guide accuracy

## Risk Assessment

### Low Risk
- **Trait abstraction**: Well-established Rust pattern
- **Performance**: Expected equal or better performance
- **Testing**: Comprehensive existing test suite provides validation

### Medium Risk  
- **API complexity**: Generic interfaces may increase learning curve
- **Migration effort**: Users need to update to new patterns over time

### Mitigation Strategies
- **Deprecation timeline**: 6-month overlap period for old APIs
- **Documentation**: Comprehensive migration guides and examples
- **Performance validation**: Extensive benchmarking to ensure no regressions
- **Community feedback**: Early alpha releases for user validation

## Success Metrics

- [ ] **Backward Compatibility**: 100% existing functionality preserved
- [ ] **Performance**: ≤5% performance impact, ≥20% improvement for specialized cases
- [ ] **Adoption**: ≥80% user satisfaction in alpha testing
- [ ] **Documentation**: Complete migration guides and examples
- [ ] **Test Coverage**: ≥95% code coverage maintained across all implementations

---

**Status**: 🚧 **UNDER ACTIVE DEVELOPMENT** 🚧

This modularization is currently in progress. The project architecture already provides excellent separation between hypergraph-specific connectivity and generic neuromorphic computation, making this transition highly feasible.