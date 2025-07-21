//! Asynchronous spike processing and event-driven computation
//!
//! This module provides async/await based spike processing capabilities
//! for high-performance neuromorphic computation.

use crate::{
    error::{Result, SHNNError},
    spike::{NeuronId, Spike, TimedSpike, SpikeTrain},
    time::{Time, Duration},
    hypergraph::{HypergraphNetwork, SpikeRoute},
    neuron::{Neuron, NeuronPool, LIFNeuron},
    memory::{SpikeQueue, SpikeBuffer},
};

#[cfg(feature = "std")]
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::Instant,
};

use core::{
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
    fmt,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Async spike processor trait
pub trait AsyncSpikeProcessor {
    /// Process a single spike asynchronously
    fn process_spike(&mut self, spike: Spike) -> Pin<Box<dyn Future<Output = Result<Vec<Spike>>> + Send + '_>>;
    
    /// Process a batch of spikes
    fn process_batch(&mut self, spikes: Vec<Spike>) -> Pin<Box<dyn Future<Output = Result<Vec<Spike>>> + Send + '_>>;
    
    /// Update processor state
    fn update(&mut self, current_time: Time) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
}

/// Async event for spike processing
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SpikeEvent {
    /// Incoming spike event
    Spike(TimedSpike),
    /// Neuron update event
    Update {
        neuron_id: NeuronId,
        time: Time,
    },
    /// Network synchronization event
    Sync(Time),
    /// Processing complete event
    Complete {
        processed_spikes: usize,
        elapsed_time: Duration,
    },
}

impl fmt::Display for SpikeEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spike(spike) => write!(f, "SpikeEvent({})", spike.spike),
            Self::Update { neuron_id, time } => write!(f, "UpdateEvent({} @ {})", neuron_id, time),
            Self::Sync(time) => write!(f, "SyncEvent({})", time),
            Self::Complete { processed_spikes, elapsed_time } => {
                write!(f, "CompleteEvent({} spikes, {})", processed_spikes, elapsed_time)
            }
        }
    }
}

/// Async spike channel for event communication
pub struct SpikeChannel {
    /// Pending events
    events: VecDeque<SpikeEvent>,
    /// Waiting tasks
    wakers: Vec<Waker>,
    /// Channel capacity
    capacity: usize,
    /// Whether channel is closed
    closed: bool,
}

impl SpikeChannel {
    /// Create a new spike channel
    pub fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity),
            wakers: Vec::new(),
            capacity,
            closed: false,
        }
    }
    
    /// Send an event to the channel
    pub fn send(&mut self, event: SpikeEvent) -> Result<()> {
        if self.closed {
            return Err(SHNNError::async_error("Channel is closed"));
        }
        
        if self.events.len() >= self.capacity {
            return Err(SHNNError::async_error("Channel is full"));
        }
        
        self.events.push_back(event);
        
        // Wake up waiting tasks
        for waker in self.wakers.drain(..) {
            waker.wake();
        }
        
        Ok(())
    }
    
    /// Try to receive an event (non-blocking)
    pub fn try_recv(&mut self) -> Option<SpikeEvent> {
        self.events.pop_front()
    }
    
    /// Receive an event asynchronously
    pub fn recv(&mut self) -> SpikeChannelReceiver<'_> {
        SpikeChannelReceiver { channel: self }
    }
    
    /// Close the channel
    pub fn close(&mut self) {
        self.closed = true;
        
        // Wake up all waiting tasks
        for waker in self.wakers.drain(..) {
            waker.wake();
        }
    }
    
    /// Check if channel is closed
    pub fn is_closed(&self) -> bool {
        self.closed
    }
    
    /// Get number of pending events
    pub fn len(&self) -> usize {
        self.events.len()
    }
    
    /// Check if channel is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl Default for SpikeChannel {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Future for receiving events from SpikeChannel
pub struct SpikeChannelReceiver<'a> {
    channel: &'a mut SpikeChannel,
}

impl<'a> Future for SpikeChannelReceiver<'a> {
    type Output = Option<SpikeEvent>;
    
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(event) = self.channel.try_recv() {
            Poll::Ready(Some(event))
        } else if self.channel.closed {
            Poll::Ready(None)
        } else {
            // Register waker for when new events arrive
            self.channel.wakers.push(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// Async neural network processor
pub struct AsyncNeuralNetwork {
    /// Neuron pool
    neurons: NeuronPool,
    /// Hypergraph connectivity
    hypergraph: HypergraphNetwork,
    /// Event channel
    event_channel: SpikeChannel,
    /// Spike queue for delayed events
    spike_queue: SpikeQueue,
    /// Current simulation time
    current_time: Time,
    /// Processing statistics
    stats: ProcessingStats,
    /// Buffer for spike history
    spike_buffer: SpikeBuffer,
}

impl AsyncNeuralNetwork {
    /// Create a new async neural network
    pub fn new() -> Self {
        Self {
            neurons: NeuronPool::new(),
            hypergraph: HypergraphNetwork::new(),
            event_channel: SpikeChannel::default(),
            spike_queue: SpikeQueue::new(),
            current_time: Time::ZERO,
            stats: ProcessingStats::default(),
            spike_buffer: SpikeBuffer::default(),
        }
    }
    
    /// Add a neuron to the network
    pub fn add_neuron(&mut self, neuron: LIFNeuron) -> Result<()> {
        self.neurons.add_lif_neuron(neuron)
    }
    
    /// Get mutable access to hypergraph
    pub fn hypergraph_mut(&mut self) -> &mut HypergraphNetwork {
        &mut self.hypergraph
    }
    
    /// Process spikes asynchronously
    pub async fn process_spike(&mut self, spike: Spike) -> Result<Vec<Spike>> {
        self.stats.spikes_processed += 1;
        let start_time = self.current_time;
        
        // Route spike through hypergraph
        let routes = self.hypergraph.route_spike(&spike, self.current_time);
        let mut output_spikes = Vec::new();
        
        // Process each route
        for route in routes {
            for &target_id in &route.targets {
                if let Some(neuron) = self.neurons.get_lif_neuron_mut(target_id) {
                    if let Some(output) = neuron.process_spike(&spike, route.delivery_time) {
                        output_spikes.push(output);
                        
                        // Add to spike buffer
                        let timed_spike = TimedSpike::new(output.clone(), route.delivery_time);
                        self.spike_buffer.push(timed_spike.clone());
                        
                        // Schedule delayed delivery if needed
                        if route.delivery_time > self.current_time {
                            self.spike_queue.push(timed_spike);
                        }
                    }
                }
            }
        }
        
        // Send completion event
        let elapsed = self.current_time - start_time;
        let event = SpikeEvent::Complete {
            processed_spikes: 1,
            elapsed_time: elapsed,
        };
        
        self.event_channel.send(event)?;
        
        Ok(output_spikes)
    }
    
    /// Process a batch of spikes
    pub async fn process_batch(&mut self, spikes: Vec<Spike>) -> Result<Vec<Spike>> {
        let start_time = self.current_time;
        let mut all_outputs = Vec::new();
        
        for spike in spikes {
            let outputs = self.process_spike(spike).await?;
            all_outputs.extend(outputs);
        }
        
        let elapsed = self.current_time - start_time;
        self.stats.batches_processed += 1;
        self.stats.total_processing_time = self.stats.total_processing_time + elapsed;
        
        Ok(all_outputs)
    }
    
    /// Update network state
    pub async fn update(&mut self, new_time: Time) -> Result<()> {
        let dt = new_time - self.current_time;
        self.current_time = new_time;
        
        // Update all neurons
        self.neurons.update_all(new_time, dt);
        
        // Process any ready spikes from queue
        let ready_spikes = self.spike_queue.pop_ready(new_time);
        for timed_spike in ready_spikes {
            self.process_spike(timed_spike.spike).await?;
        }
        
        // Send sync event
        let event = SpikeEvent::Sync(new_time);
        self.event_channel.send(event)?;
        
        Ok(())
    }
    
    /// Run network simulation for a duration
    pub async fn run_simulation(&mut self, duration: Duration, dt: Duration) -> Result<ProcessingStats> {
        let start_time = self.current_time;
        let end_time = start_time + duration;
        
        while self.current_time < end_time {
            self.update(self.current_time + dt).await?;
            
            // Yield control to allow other tasks to run
            yield_now().await;
        }
        
        Ok(self.stats.clone())
    }
    
    /// Get processing statistics
    pub fn stats(&self) -> &ProcessingStats {
        &self.stats
    }
    
    /// Get event channel
    pub fn event_channel(&mut self) -> &mut SpikeChannel {
        &mut self.event_channel
    }
    
    /// Get spike history
    pub fn spike_history(&self) -> Vec<TimedSpike> {
        self.spike_buffer.to_vec()
    }
    
    /// Reset network state
    pub fn reset(&mut self) {
        self.neurons.reset_all();
        self.spike_queue.clear();
        self.spike_buffer.clear();
        self.current_time = Time::ZERO;
        self.stats = ProcessingStats::default();
    }
}

impl Default for AsyncNeuralNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncSpikeProcessor for AsyncNeuralNetwork {
    fn process_spike(&mut self, spike: Spike) -> Pin<Box<dyn Future<Output = Result<Vec<Spike>>> + Send + '_>> {
        Box::pin(self.process_spike(spike))
    }
    
    fn process_batch(&mut self, spikes: Vec<Spike>) -> Pin<Box<dyn Future<Output = Result<Vec<Spike>>> + Send + '_>> {
        Box::pin(self.process_batch(spikes))
    }
    
    fn update(&mut self, current_time: Time) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(self.update(current_time))
    }
}

/// Processing statistics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProcessingStats {
    /// Total spikes processed
    pub spikes_processed: u64,
    /// Total batches processed
    pub batches_processed: u64,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Average processing time per spike
    pub avg_spike_time: Duration,
    /// Peak memory usage (estimated)
    pub peak_memory_usage: usize,
}

impl ProcessingStats {
    /// Update average spike processing time
    pub fn update_avg_spike_time(&mut self) {
        if self.spikes_processed > 0 {
            self.avg_spike_time = Duration::from_nanos(
                self.total_processing_time.as_nanos() / self.spikes_processed
            );
        }
    }
    
    /// Get processing rate (spikes per second)
    pub fn processing_rate(&self) -> f64 {
        if self.total_processing_time.as_secs_f64() > 0.0 {
            self.spikes_processed as f64 / self.total_processing_time.as_secs_f64()
        } else {
            0.0
        }
    }
}

impl fmt::Display for ProcessingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Processing Stats: {} spikes, {} batches, {:.1} spikes/sec, avg_time={}",
            self.spikes_processed,
            self.batches_processed,
            self.processing_rate(),
            self.avg_spike_time
        )
    }
}

/// Async spike stream processor
pub struct SpikeStreamProcessor {
    /// Input stream buffer
    input_buffer: VecDeque<SpikeTrain>,
    /// Output stream buffer
    output_buffer: VecDeque<SpikeTrain>,
    /// Processing function
    processor: Box<dyn Fn(&SpikeTrain) -> Result<SpikeTrain> + Send + Sync>,
    /// Buffer capacity
    capacity: usize,
    /// Processing statistics
    stats: StreamStats,
}

impl SpikeStreamProcessor {
    /// Create a new stream processor
    pub fn new<F>(capacity: usize, processor: F) -> Self
    where
        F: Fn(&SpikeTrain) -> Result<SpikeTrain> + Send + Sync + 'static,
    {
        Self {
            input_buffer: VecDeque::with_capacity(capacity),
            output_buffer: VecDeque::with_capacity(capacity),
            processor: Box::new(processor),
            capacity,
            stats: StreamStats::default(),
        }
    }
    
    /// Add spike train to input stream
    pub async fn send(&mut self, train: SpikeTrain) -> Result<()> {
        if self.input_buffer.len() >= self.capacity {
            return Err(SHNNError::async_error("Input buffer full"));
        }
        
        self.input_buffer.push_back(train);
        self.stats.trains_received += 1;
        
        // Process if we have capacity
        if self.output_buffer.len() < self.capacity {
            self.process_next().await?;
        }
        
        Ok(())
    }
    
    /// Receive processed spike train
    pub async fn recv(&mut self) -> Option<SpikeTrain> {
        // Try to process more if output buffer is empty
        if self.output_buffer.is_empty() && !self.input_buffer.is_empty() {
            let _ = self.process_next().await;
        }
        
        let result = self.output_buffer.pop_front();
        if result.is_some() {
            self.stats.trains_sent += 1;
        }
        result
    }
    
    /// Process next spike train in queue
    async fn process_next(&mut self) -> Result<()> {
        if let Some(input_train) = self.input_buffer.pop_front() {
            let _start_time = Time::from_nanos(0); // Would use actual timer in real implementation
            
            match (self.processor)(&input_train) {
                Ok(output_train) => {
                    self.output_buffer.push_back(output_train);
                    self.stats.trains_processed += 1;
                }
                Err(e) => {
                    self.stats.processing_errors += 1;
                    return Err(e);
                }
            }
            
            // Yield to allow other tasks to run
            yield_now().await;
        }
        
        Ok(())
    }
    
    /// Get stream statistics
    pub fn stats(&self) -> &StreamStats {
        &self.stats
    }
    
    /// Check if processor is idle
    pub fn is_idle(&self) -> bool {
        self.input_buffer.is_empty() && self.output_buffer.is_empty()
    }
    
    /// Get buffer utilization
    pub fn input_utilization(&self) -> f32 {
        self.input_buffer.len() as f32 / self.capacity as f32
    }
    
    pub fn output_utilization(&self) -> f32 {
        self.output_buffer.len() as f32 / self.capacity as f32
    }
}

/// Stream processing statistics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StreamStats {
    /// Trains received
    pub trains_received: u64,
    /// Trains processed
    pub trains_processed: u64,
    /// Trains sent
    pub trains_sent: u64,
    /// Processing errors
    pub processing_errors: u64,
}

impl fmt::Display for StreamStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Stream Stats: {}/{}/{} (recv/proc/sent), {} errors",
            self.trains_received,
            self.trains_processed,
            self.trains_sent,
            self.processing_errors
        )
    }
}

/// Simple yield function for cooperative multitasking
pub async fn yield_now() {
    YieldNow { yielded: false }.await
}

/// Future that yields once
struct YieldNow {
    yielded: bool,
}

impl Future for YieldNow {
    type Output = ();
    
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// Async task spawner (simplified)
pub struct AsyncTaskSpawner;

impl AsyncTaskSpawner {
    /// Spawn a new async task
    pub fn spawn<F, T>(_future: F) -> AsyncTaskHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        // In a real implementation, this would interact with an async runtime
        AsyncTaskHandle::new()
    }
}

/// Handle for async tasks
pub struct AsyncTaskHandle<T> {
    _phantom: core::marker::PhantomData<T>,
}

impl<T> AsyncTaskHandle<T> {
    fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
    
    /// Check if task is complete (simplified)
    pub fn is_ready(&self) -> bool {
        false // Placeholder implementation
    }
    
    /// Wait for task completion (simplified)
    pub async fn join(self) -> Result<T> {
        Err(SHNNError::async_error("Task joining not implemented"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::{LIFNeuron, LIFConfig};
    
    #[test]
    fn test_spike_channel() {
        let mut channel = SpikeChannel::new(2);
        
        let event = SpikeEvent::Sync(Time::from_millis(1));
        assert!(channel.send(event.clone()).is_ok());
        assert_eq!(channel.len(), 1);
        
        let received = channel.try_recv();
        assert!(received.is_some());
        assert_eq!(channel.len(), 0);
    }
    
    #[test]
    fn test_async_network() {
        let mut network = AsyncNeuralNetwork::new();
        
        let neuron = LIFNeuron::new(
            NeuronId::new(0),
            LIFConfig::default()
        ).unwrap();
        
        assert!(network.add_neuron(neuron).is_ok());
        assert_eq!(network.stats().spikes_processed, 0);
    }
    
    #[tokio::test]
    async fn test_spike_processing() {
        let mut network = AsyncNeuralNetwork::new();
        
        let neuron = LIFNeuron::new(
            NeuronId::new(0),
            LIFConfig::default()
        ).unwrap();
        
        network.add_neuron(neuron).unwrap();
        
        let spike = Spike::binary(NeuronId::new(1), Time::from_millis(1)).unwrap();
        let _outputs = network.process_spike(spike).await.unwrap();
        
        assert_eq!(network.stats().spikes_processed, 1);
    }
    
    #[tokio::test]
    async fn test_stream_processor() {
        let mut processor = SpikeStreamProcessor::new(10, |train| {
            // Simple identity processor
            Ok(train.clone())
        });
        
        let train = SpikeTrain::binary(
            NeuronId::new(0),
            vec![Time::from_millis(1), Time::from_millis(2)]
        ).unwrap();
        
        assert!(processor.send(train).await.is_ok());
        assert_eq!(processor.stats().trains_received, 1);
        
        let output = processor.recv().await;
        assert!(output.is_some());
        assert_eq!(processor.stats().trains_sent, 1);
    }
    
    #[tokio::test]
    async fn test_yield_now() {
        // Just make sure yield_now doesn't block forever
        yield_now().await;
        assert!(true);
    }
}