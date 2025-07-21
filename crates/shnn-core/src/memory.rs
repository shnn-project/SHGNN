//! Memory management utilities for neuromorphic computing
//!
//! This module provides specialized memory management patterns optimized
//! for spike-based neural computation, including memory pools, circular
//! buffers, and cache-friendly data structures.

use crate::{
    error::{Result, SHNNError},
    spike::{NeuronId, Spike, TimedSpike},
    time::Time,
};
use core::{fmt, mem, ptr};

#[cfg(feature = "std")]
use std::collections::VecDeque;

#[cfg(not(feature = "std"))]
use heapless::{
    pool::{Pool, Node},
    Vec as HeaplessVec,
    Deque,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Memory pool for spike objects
#[cfg(feature = "std")]
pub struct SpikePool {
    /// Available spike objects
    available: VecDeque<Spike>,
    /// Maximum pool size
    max_size: usize,
    /// Current pool size
    current_size: usize,
    /// Pool statistics
    stats: PoolStats,
}

#[cfg(feature = "std")]
impl SpikePool {
    /// Create a new spike pool
    pub fn new(initial_size: usize, max_size: usize) -> Self {
        let mut pool = Self {
            available: VecDeque::with_capacity(initial_size),
            max_size,
            current_size: 0,
            stats: PoolStats::default(),
        };
        
        // Pre-allocate spike objects
        for _ in 0..initial_size {
            if let Ok(spike) = Spike::binary(NeuronId::new(0), Time::ZERO) {
                pool.available.push_back(spike);
                pool.current_size += 1;
            }
        }
        
        pool
    }
    
    /// Get a spike from the pool
    pub fn get(&mut self) -> Option<Spike> {
        self.stats.requests += 1;
        
        if let Some(spike) = self.available.pop_front() {
            self.stats.hits += 1;
            Some(spike)
        } else {
            self.stats.misses += 1;
            // Pool is empty, create new spike if under limit
            if self.current_size < self.max_size {
                self.current_size += 1;
                Spike::binary(NeuronId::new(0), Time::ZERO).ok()
            } else {
                None
            }
        }
    }
    
    /// Return a spike to the pool
    pub fn put(&mut self, mut spike: Spike) {
        if self.available.len() < self.max_size {
            // Reset spike to default state
            spike.source = NeuronId::new(0);
            spike.timestamp = Time::ZERO;
            spike.amplitude = 1.0;
            
            self.available.push_back(spike);
            self.stats.returns += 1;
        } else {
            // Pool is full, let spike be dropped
            self.current_size = self.current_size.saturating_sub(1);
        }
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }
    
    /// Reset pool statistics
    pub fn reset_stats(&mut self) {
        self.stats = PoolStats::default();
    }
    
    /// Get current pool utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        if self.max_size == 0 {
            0.0
        } else {
            self.available.len() as f32 / self.max_size as f32
        }
    }
}

#[cfg(feature = "std")]
impl Default for SpikePool {
    fn default() -> Self {
        Self::new(100, 1000)
    }
}

/// No-std spike pool using heapless
#[cfg(not(feature = "std"))]
pub struct SpikePool<const N: usize> {
    /// Pool nodes
    memory: [Node<Spike>; N],
    /// Pool manager
    pool: Pool<Spike>,
    /// Statistics
    stats: PoolStats,
}

#[cfg(not(feature = "std"))]
impl<const N: usize> SpikePool<N> {
    /// Create a new spike pool
    pub fn new() -> Self {
        let memory: [Node<Spike>; N] = unsafe {
            mem::MaybeUninit::uninit().assume_init()
        };
        let pool = Pool::new();
        
        Self {
            memory,
            pool,
            stats: PoolStats::default(),
        }
    }
    
    /// Initialize the pool (must be called before use)
    pub fn init(&mut self) {
        // Initialize pool with memory nodes
        for node in &mut self.memory {
            // This is a simplified initialization - real implementation
            // would properly manage the pool lifecycle
            unsafe {
                ptr::write(node as *mut Node<Spike>, Node::new());
            }
        }
    }
    
    /// Get statistics
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }
}

/// Pool statistics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PoolStats {
    /// Total allocation requests
    pub requests: u64,
    /// Cache hits (object found in pool)
    pub hits: u64,
    /// Cache misses (new allocation required)
    pub misses: u64,
    /// Objects returned to pool
    pub returns: u64,
}

impl PoolStats {
    /// Calculate hit ratio
    pub fn hit_ratio(&self) -> f32 {
        if self.requests == 0 {
            0.0
        } else {
            self.hits as f32 / self.requests as f32
        }
    }
    
    /// Calculate miss ratio
    pub fn miss_ratio(&self) -> f32 {
        1.0 - self.hit_ratio()
    }
}

impl fmt::Display for PoolStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pool Stats: {}/{} requests (hit ratio: {:.2}%), {} returns",
            self.hits,
            self.requests,
            self.hit_ratio() * 100.0,
            self.returns
        )
    }
}

/// Circular buffer for spike history
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpikeBuffer {
    /// Buffer storage
    buffer: Vec<TimedSpike>,
    /// Current write position
    write_pos: usize,
    /// Current buffer size
    size: usize,
    /// Maximum capacity
    capacity: usize,
    /// Whether buffer has wrapped around
    wrapped: bool,
}

impl SpikeBuffer {
    /// Create a new spike buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![
                TimedSpike::new(
                    Spike::binary(NeuronId::new(0), Time::ZERO).unwrap(),
                    Time::ZERO
                );
                capacity
            ],
            write_pos: 0,
            size: 0,
            capacity,
            wrapped: false,
        }
    }
    
    /// Add a spike to the buffer
    pub fn push(&mut self, spike: TimedSpike) {
        self.buffer[self.write_pos] = spike;
        self.write_pos = (self.write_pos + 1) % self.capacity;
        
        if self.size < self.capacity {
            self.size += 1;
        } else {
            self.wrapped = true;
        }
    }
    
    /// Get the most recent spike
    pub fn last(&self) -> Option<&TimedSpike> {
        if self.size == 0 {
            None
        } else {
            let pos = if self.write_pos == 0 {
                self.capacity - 1
            } else {
                self.write_pos - 1
            };
            Some(&self.buffer[pos])
        }
    }
    
    /// Get spike at offset from most recent (0 = most recent)
    pub fn get(&self, offset: usize) -> Option<&TimedSpike> {
        if offset >= self.size {
            return None;
        }
        
        let pos = if self.write_pos >= offset + 1 {
            self.write_pos - offset - 1
        } else {
            self.capacity - (offset + 1 - self.write_pos)
        };
        
        Some(&self.buffer[pos])
    }
    
    /// Get all spikes in chronological order
    pub fn to_vec(&self) -> Vec<TimedSpike> {
        if self.size == 0 {
            return Vec::new();
        }
        
        let mut result = Vec::with_capacity(self.size);
        
        if self.wrapped {
            // Buffer has wrapped, start from write_pos
            for i in 0..self.capacity {
                let pos = (self.write_pos + i) % self.capacity;
                result.push(self.buffer[pos].clone());
            }
        } else {
            // Buffer hasn't wrapped, take from beginning
            for i in 0..self.size {
                result.push(self.buffer[i].clone());
            }
        }
        
        result
    }
    
    /// Get spikes within a time window
    pub fn get_spikes_in_window(&self, start_time: Time, end_time: Time) -> Vec<TimedSpike> {
        let mut result = Vec::new();
        
        for i in 0..self.size {
            if let Some(spike) = self.get(i) {
                if spike.delivery_time >= start_time && spike.delivery_time <= end_time {
                    result.push(spike.clone());
                }
            }
        }
        
        result.sort_by_key(|s| s.delivery_time);
        result
    }
    
    /// Clear the buffer
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.size = 0;
        self.wrapped = false;
    }
    
    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.size
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    
    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }
    
    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get buffer utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        if self.capacity == 0 {
            0.0
        } else {
            self.size as f32 / self.capacity as f32
        }
    }
}

impl Default for SpikeBuffer {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Memory-efficient spike queue for event scheduling
#[derive(Debug, Clone)]
pub struct SpikeQueue {
    /// Priority queue of spikes (min-heap by delivery time)
    queue: Vec<TimedSpike>,
    /// Queue statistics
    stats: QueueStats,
}

impl SpikeQueue {
    /// Create a new spike queue
    pub fn new() -> Self {
        Self {
            queue: Vec::new(),
            stats: QueueStats::default(),
        }
    }
    
    /// Create with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            queue: Vec::with_capacity(capacity),
            stats: QueueStats::default(),
        }
    }
    
    /// Add a spike to the queue
    pub fn push(&mut self, spike: TimedSpike) {
        self.queue.push(spike);
        self.bubble_up(self.queue.len() - 1);
        self.stats.insertions += 1;
    }
    
    /// Remove and return the earliest spike
    pub fn pop(&mut self) -> Option<TimedSpike> {
        if self.queue.is_empty() {
            return None;
        }
        
        let result = self.queue[0].clone();
        let last_idx = self.queue.len() - 1;
        
        if last_idx > 0 {
            self.queue.swap(0, last_idx);
            self.queue.pop();
            self.bubble_down(0);
        } else {
            self.queue.pop();
        }
        
        self.stats.extractions += 1;
        Some(result)
    }
    
    /// Peek at the earliest spike without removing it
    pub fn peek(&self) -> Option<&TimedSpike> {
        self.queue.first()
    }
    
    /// Get all spikes ready for delivery at or before the given time
    pub fn pop_ready(&mut self, current_time: Time) -> Vec<TimedSpike> {
        let mut ready = Vec::new();
        
        while let Some(spike) = self.peek() {
            if spike.delivery_time <= current_time {
                ready.push(self.pop().unwrap());
            } else {
                break;
            }
        }
        
        ready
    }
    
    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
    
    /// Get queue length
    pub fn len(&self) -> usize {
        self.queue.len()
    }
    
    /// Clear the queue
    pub fn clear(&mut self) {
        self.queue.clear();
        self.stats = QueueStats::default();
    }
    
    /// Get queue statistics
    pub fn stats(&self) -> &QueueStats {
        &self.stats
    }
    
    /// Bubble up operation for heap maintenance
    fn bubble_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent_idx = (idx - 1) / 2;
            
            if self.queue[idx].delivery_time >= self.queue[parent_idx].delivery_time {
                break;
            }
            
            self.queue.swap(idx, parent_idx);
            idx = parent_idx;
        }
    }
    
    /// Bubble down operation for heap maintenance
    fn bubble_down(&mut self, mut idx: usize) {
        loop {
            let left_child = 2 * idx + 1;
            let right_child = 2 * idx + 2;
            let mut smallest = idx;
            
            if left_child < self.queue.len() && 
               self.queue[left_child].delivery_time < self.queue[smallest].delivery_time {
                smallest = left_child;
            }
            
            if right_child < self.queue.len() && 
               self.queue[right_child].delivery_time < self.queue[smallest].delivery_time {
                smallest = right_child;
            }
            
            if smallest == idx {
                break;
            }
            
            self.queue.swap(idx, smallest);
            idx = smallest;
        }
    }
}

impl Default for SpikeQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Queue statistics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QueueStats {
    /// Total insertions
    pub insertions: u64,
    /// Total extractions
    pub extractions: u64,
}

impl fmt::Display for QueueStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Queue Stats: {} insertions, {} extractions",
            self.insertions, self.extractions
        )
    }
}

/// Memory arena for bulk allocations
#[derive(Debug)]
pub struct MemoryArena {
    /// Memory blocks
    blocks: Vec<Vec<u8>>,
    /// Current block index
    current_block: usize,
    /// Current position in block
    current_pos: usize,
    /// Block size
    block_size: usize,
    /// Total allocated bytes
    total_allocated: usize,
}

impl MemoryArena {
    /// Create a new memory arena
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            current_block: 0,
            current_pos: 0,
            block_size,
            total_allocated: 0,
        }
    }
    
    /// Allocate memory for a type
    pub fn alloc<T>(&mut self) -> Result<*mut T> {
        self.alloc_bytes(mem::size_of::<T>(), mem::align_of::<T>())
            .map(|ptr| ptr as *mut T)
    }
    
    /// Allocate raw bytes with alignment
    pub fn alloc_bytes(&mut self, size: usize, align: usize) -> Result<*mut u8> {
        // Ensure we have a block
        if self.blocks.is_empty() {
            self.add_block()?;
        }
        
        // Align current position
        let aligned_pos = (self.current_pos + align - 1) & !(align - 1);
        
        // Check if we need a new block
        if aligned_pos + size > self.block_size {
            self.add_block()?;
            self.current_pos = 0;
            self.current_block = self.blocks.len() - 1;
            
            // Align again in new block
            self.current_pos = (self.current_pos + align - 1) & !(align - 1);
        }
        
        // Allocate from current block
        let ptr = unsafe {
            self.blocks[self.current_block]
                .as_mut_ptr()
                .add(self.current_pos)
        };
        
        self.current_pos += size;
        self.total_allocated += size;
        
        Ok(ptr)
    }
    
    /// Add a new memory block
    fn add_block(&mut self) -> Result<()> {
        self.blocks.push(vec![0u8; self.block_size]);
        Ok(())
    }
    
    /// Reset the arena (keeps allocated blocks but resets positions)
    pub fn reset(&mut self) {
        self.current_block = 0;
        self.current_pos = 0;
        // total_allocated is cumulative, don't reset
    }
    
    /// Clear all memory (deallocates blocks)
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.current_block = 0;
        self.current_pos = 0;
        self.total_allocated = 0;
    }
    
    /// Get total allocated bytes
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }
    
    /// Get number of blocks
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

impl Default for MemoryArena {
    fn default() -> Self {
        Self::new(64 * 1024) // 64KB blocks
    }
}

/// Cache-friendly neuron data layout
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeuronSoA {
    /// Neuron IDs
    pub ids: Vec<u32>,
    /// Membrane potentials
    pub voltages: Vec<f32>,
    /// Last spike times
    pub last_spikes: Vec<Option<Time>>,
    /// Active flags
    pub active: Vec<bool>,
    /// Number of neurons
    pub count: usize,
}

impl NeuronSoA {
    /// Create a new structure-of-arrays layout
    pub fn new(capacity: usize) -> Self {
        Self {
            ids: Vec::with_capacity(capacity),
            voltages: Vec::with_capacity(capacity),
            last_spikes: Vec::with_capacity(capacity),
            active: Vec::with_capacity(capacity),
            count: 0,
        }
    }
    
    /// Add a neuron
    pub fn add_neuron(&mut self, id: u32, voltage: f32) -> Result<usize> {
        let index = self.count;
        
        self.ids.push(id);
        self.voltages.push(voltage);
        self.last_spikes.push(None);
        self.active.push(true);
        self.count += 1;
        
        Ok(index)
    }
    
    /// Update voltage for multiple neurons (vectorizable)
    pub fn update_voltages<F>(&mut self, update_fn: F)
    where
        F: Fn(f32) -> f32,
    {
        for voltage in &mut self.voltages {
            *voltage = update_fn(*voltage);
        }
    }
    
    /// Apply threshold to all neurons (vectorizable)
    pub fn apply_threshold(&mut self, threshold: f32, current_time: Time) -> Vec<usize> {
        let mut fired = Vec::new();
        
        for i in 0..self.count {
            if self.active[i] && self.voltages[i] >= threshold {
                self.last_spikes[i] = Some(current_time);
                self.voltages[i] = -70.0; // Reset voltage
                fired.push(i);
            }
        }
        
        fired
    }
    
    /// Get neuron count
    pub fn len(&self) -> usize {
        self.count
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl Default for NeuronSoA {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[cfg(feature = "std")]
    #[test]
    fn test_spike_pool() {
        let mut pool = SpikePool::new(2, 5);
        
        // Get spikes from pool
        let spike1 = pool.get().unwrap();
        let spike2 = pool.get().unwrap();
        let spike3 = pool.get().unwrap(); // Should create new one
        
        assert_eq!(pool.stats().hits, 2);
        assert_eq!(pool.stats().misses, 1);
        
        // Return spikes to pool
        pool.put(spike1);
        pool.put(spike2);
        
        assert_eq!(pool.stats().returns, 2);
        
        // Get again should hit cache
        let _spike4 = pool.get().unwrap();
        assert_eq!(pool.stats().hits, 3);
    }
    
    #[test]
    fn test_spike_buffer() {
        let mut buffer = SpikeBuffer::new(3);
        
        let spike1 = TimedSpike::new(
            Spike::binary(NeuronId::new(1), Time::from_millis(1)).unwrap(),
            Time::from_millis(1)
        );
        let spike2 = TimedSpike::new(
            Spike::binary(NeuronId::new(2), Time::from_millis(2)).unwrap(),
            Time::from_millis(2)
        );
        let spike3 = TimedSpike::new(
            Spike::binary(NeuronId::new(3), Time::from_millis(3)).unwrap(),
            Time::from_millis(3)
        );
        let spike4 = TimedSpike::new(
            Spike::binary(NeuronId::new(4), Time::from_millis(4)).unwrap(),
            Time::from_millis(4)
        );
        
        buffer.push(spike1);
        buffer.push(spike2);
        buffer.push(spike3);
        assert_eq!(buffer.len(), 3);
        assert!(buffer.is_full());
        
        // Should wrap around
        buffer.push(spike4);
        assert_eq!(buffer.len(), 3);
        
        // Most recent should be spike4
        assert_eq!(buffer.last().unwrap().spike.source, NeuronId::new(4));
    }
    
    #[test]
    fn test_spike_queue() {
        let mut queue = SpikeQueue::new();
        
        let spike1 = TimedSpike::new(
            Spike::binary(NeuronId::new(1), Time::from_millis(3)).unwrap(),
            Time::from_millis(3)
        );
        let spike2 = TimedSpike::new(
            Spike::binary(NeuronId::new(2), Time::from_millis(1)).unwrap(),
            Time::from_millis(1)
        );
        let spike3 = TimedSpike::new(
            Spike::binary(NeuronId::new(3), Time::from_millis(2)).unwrap(),
            Time::from_millis(2)
        );
        
        queue.push(spike1);
        queue.push(spike2);
        queue.push(spike3);
        
        // Should pop in time order
        assert_eq!(queue.pop().unwrap().delivery_time, Time::from_millis(1));
        assert_eq!(queue.pop().unwrap().delivery_time, Time::from_millis(2));
        assert_eq!(queue.pop().unwrap().delivery_time, Time::from_millis(3));
        assert!(queue.is_empty());
    }
    
    #[test]
    fn test_memory_arena() {
        let mut arena = MemoryArena::new(1024);
        
        // Allocate some memory
        let _ptr1: *mut i32 = arena.alloc().unwrap();
        let _ptr2: *mut f64 = arena.alloc().unwrap();
        
        assert!(arena.total_allocated() >= 12); // At least size of i32 + f64
        
        arena.reset();
        // Should be able to allocate again
        let _ptr3: *mut i32 = arena.alloc().unwrap();
    }
    
    #[test]
    fn test_neuron_soa() {
        let mut soa = NeuronSoA::new(10);
        
        soa.add_neuron(0, -70.0).unwrap();
        soa.add_neuron(1, -65.0).unwrap();
        soa.add_neuron(2, -60.0).unwrap();
        
        assert_eq!(soa.len(), 3);
        
        // Apply threshold
        let fired = soa.apply_threshold(-62.0, Time::from_millis(1));
        assert_eq!(fired.len(), 2); // Neurons 1 and 2 should fire
        
        // Update voltages
        soa.update_voltages(|v| v * 0.95);
        
        // All voltages should be scaled
        for &voltage in &soa.voltages[..soa.count] {
            assert!(voltage < -60.0);
        }
    }
    
    #[test]
    fn test_spike_queue_pop_ready() {
        let mut queue = SpikeQueue::new();
        
        let spike1 = TimedSpike::new(
            Spike::binary(NeuronId::new(1), Time::from_millis(1)).unwrap(),
            Time::from_millis(5)
        );
        let spike2 = TimedSpike::new(
            Spike::binary(NeuronId::new(2), Time::from_millis(2)).unwrap(),
            Time::from_millis(10)
        );
        let spike3 = TimedSpike::new(
            Spike::binary(NeuronId::new(3), Time::from_millis(3)).unwrap(),
            Time::from_millis(15)
        );
        
        queue.push(spike1);
        queue.push(spike2);
        queue.push(spike3);
        
        // At time 7, only first spike should be ready
        let ready = queue.pop_ready(Time::from_millis(7));
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].spike.source, NeuronId::new(1));
        assert_eq!(queue.len(), 2);
        
        // At time 12, second spike should be ready
        let ready = queue.pop_ready(Time::from_millis(12));
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].spike.source, NeuronId::new(2));
        assert_eq!(queue.len(), 1);
    }
}