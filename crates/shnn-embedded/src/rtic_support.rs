//! RTIC (Real-Time Interrupt-driven Concurrency) support for neuromorphic systems
//!
//! This module provides integration with the RTIC framework for building
//! deterministic real-time neuromorphic applications on ARM Cortex-M processors.

use crate::{
    error::{EmbeddedError, EmbeddedResult},
    fixed_point::{FixedPoint, Q16_16, FixedSpike},
    embedded_neuron::{EmbeddedNeuron, EmbeddedLIFNeuron},
    embedded_network::{EmbeddedSNN, EmbeddedTopology, NetworkStatistics},
    embedded_memory::EmbeddedSpikeBuffer,
};
use heapless::{Vec, spsc::{Queue, Producer, Consumer}};
use core::{marker::PhantomData, sync::atomic::{AtomicBool, AtomicU32, Ordering}};

/// Maximum size for RTIC spike queues
pub const RTIC_SPIKE_QUEUE_SIZE: usize = 64;

/// Maximum number of concurrent RTIC tasks
pub const MAX_RTIC_TASKS: usize = 8;

/// RTIC-based neuromorphic scheduler
pub struct RTICScheduler<T: FixedPoint> {
    /// Network instance
    network: Option<EmbeddedSNN<T>>,
    /// Input spike queue producer
    input_producer: Option<Producer<'static, FixedSpike<T>, RTIC_SPIKE_QUEUE_SIZE>>,
    /// Output spike queue consumer
    output_consumer: Option<Consumer<'static, FixedSpike<T>, RTIC_SPIKE_QUEUE_SIZE>>,
    /// Simulation time step
    time_step: T,
    /// Current simulation time
    current_time: T,
    /// Processing statistics
    statistics: RTICStatistics,
    /// Task priorities
    task_priorities: [u8; MAX_RTIC_TASKS],
    /// Interrupt configuration
    interrupt_config: InterruptConfig,
}

/// RTIC task configuration and management
#[derive(Debug, Clone)]
pub struct RTICTaskConfig {
    /// Task ID
    pub task_id: u8,
    /// Task priority (higher number = higher priority)
    pub priority: u8,
    /// Task type
    pub task_type: RTICTaskType,
    /// Execution period in microseconds
    pub period_us: u32,
    /// Maximum execution time in microseconds
    pub max_execution_time_us: u32,
    /// Enable task monitoring
    pub enable_monitoring: bool,
}

/// Types of RTIC tasks in neuromorphic systems
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RTICTaskType {
    /// Network update task (highest priority)
    NetworkUpdate,
    /// Spike processing task
    SpikeProcessing,
    /// Input handling task
    InputHandling,
    /// Output generation task
    OutputGeneration,
    /// Plasticity update task
    PlasticityUpdate,
    /// Monitoring and diagnostics task
    Monitoring,
    /// Communication task
    Communication,
    /// System maintenance task
    SystemMaintenance,
}

/// Interrupt configuration for RTIC
#[derive(Debug, Clone)]
pub struct InterruptConfig {
    /// Timer interrupt for network updates
    pub timer_interrupt: Option<TimerInterruptConfig>,
    /// External input interrupts
    pub input_interrupts: Vec<InputInterruptConfig, 8>,
    /// DMA completion interrupts
    pub dma_interrupts: Vec<DMAInterruptConfig, 4>,
    /// System interrupts (watchdog, etc.)
    pub system_interrupts: Vec<SystemInterruptConfig, 4>,
}

/// Timer interrupt configuration
#[derive(Debug, Clone)]
pub struct TimerInterruptConfig {
    /// Timer ID
    pub timer_id: u8,
    /// Interrupt priority
    pub priority: u8,
    /// Timer frequency in Hz
    pub frequency: u32,
    /// Enable auto-reload
    pub auto_reload: bool,
}

/// Input interrupt configuration
#[derive(Debug, Clone)]
pub struct InputInterruptConfig {
    /// GPIO pin number
    pub pin: u8,
    /// Interrupt priority
    pub priority: u8,
    /// Trigger type
    pub trigger: InterruptTrigger,
    /// Debounce time in microseconds
    pub debounce_us: u32,
}

/// DMA interrupt configuration
#[derive(Debug, Clone)]
pub struct DMAInterruptConfig {
    /// DMA channel
    pub channel: u8,
    /// Interrupt priority
    pub priority: u8,
    /// Transfer type
    pub transfer_type: DMATransferType,
}

/// System interrupt configuration
#[derive(Debug, Clone)]
pub struct SystemInterruptConfig {
    /// Interrupt type
    pub interrupt_type: SystemInterruptType,
    /// Priority
    pub priority: u8,
    /// Enable flag
    pub enabled: bool,
}

/// Interrupt trigger types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterruptTrigger {
    RisingEdge,
    FallingEdge,
    BothEdges,
    LowLevel,
    HighLevel,
}

/// DMA transfer types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DMATransferType {
    MemoryToMemory,
    MemoryToPeripheral,
    PeripheralToMemory,
    PeripheralToPeripheral,
}

/// System interrupt types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SystemInterruptType {
    Watchdog,
    BrownOut,
    PowerManagement,
    ClockFailure,
    MemoryError,
}

/// RTIC execution statistics
#[derive(Debug, Clone)]
pub struct RTICStatistics {
    /// Task execution counts
    pub task_executions: [AtomicU32; MAX_RTIC_TASKS],
    /// Task deadline misses
    pub deadline_misses: [AtomicU32; MAX_RTIC_TASKS],
    /// Maximum execution times per task
    pub max_execution_times: [u32; MAX_RTIC_TASKS],
    /// Average execution times per task
    pub avg_execution_times: [u32; MAX_RTIC_TASKS],
    /// System load percentage
    pub system_load: AtomicU32,
    /// Stack usage high watermark
    pub max_stack_usage: AtomicU32,
    /// Interrupt counts
    pub interrupt_counts: [AtomicU32; 16],
    /// Context switches
    pub context_switches: AtomicU32,
}

/// Real-time constraints for neuromorphic processing
#[derive(Debug, Clone)]
pub struct RTConstraints {
    /// Maximum allowed jitter in microseconds
    pub max_jitter_us: u32,
    /// Minimum inter-spike interval in microseconds
    pub min_spike_interval_us: u32,
    /// Maximum processing latency in microseconds
    pub max_latency_us: u32,
    /// Required throughput in spikes per second
    pub required_throughput: u32,
    /// Memory constraints
    pub memory_constraints: MemoryConstraints,
}

/// Memory constraints for real-time operation
#[derive(Debug, Clone)]
pub struct MemoryConstraints {
    /// Maximum heap usage in bytes
    pub max_heap_bytes: usize,
    /// Maximum stack usage in bytes
    pub max_stack_bytes: usize,
    /// DMA buffer size in bytes
    pub dma_buffer_size: usize,
    /// Spike buffer size
    pub spike_buffer_size: usize,
}

impl<T: FixedPoint> RTICScheduler<T> {
    /// Create a new RTIC scheduler
    pub fn new(time_step: T) -> Self {
        Self {
            network: None,
            input_producer: None,
            output_consumer: None,
            time_step,
            current_time: T::zero(),
            statistics: RTICStatistics::new(),
            task_priorities: [0; MAX_RTIC_TASKS],
            interrupt_config: InterruptConfig::default(),
        }
    }
    
    /// Initialize the scheduler with a network
    pub fn init_network(&mut self, network: EmbeddedSNN<T>) -> EmbeddedResult<()> {
        self.network = Some(network);
        Ok(())
    }
    
    /// Configure RTIC tasks
    pub fn configure_tasks(&mut self, tasks: &[RTICTaskConfig]) -> EmbeddedResult<()> {
        for task in tasks {
            if (task.task_id as usize) < MAX_RTIC_TASKS {
                self.task_priorities[task.task_id as usize] = task.priority;
            }
        }
        Ok(())
    }
    
    /// Configure interrupts
    pub fn configure_interrupts(&mut self, config: InterruptConfig) -> EmbeddedResult<()> {
        self.interrupt_config = config;
        Ok(())
    }
    
    /// Execute network update task
    pub fn execute_network_update(&mut self, inputs: &[T]) -> EmbeddedResult<Vec<FixedSpike<T>, 32>> {
        let start_time = self.get_timestamp();
        
        let spikes = if let Some(ref mut network) = self.network {
            network.update(inputs)?
        } else {
            Vec::new()
        };
        
        let execution_time = self.get_timestamp() - start_time;
        self.update_task_statistics(RTICTaskType::NetworkUpdate as usize, execution_time);
        
        self.current_time = self.current_time + self.time_step;
        Ok(spikes)
    }
    
    /// Process input spikes
    pub fn process_input_spikes(&mut self, spikes: &[FixedSpike<T>]) -> EmbeddedResult<()> {
        let start_time = self.get_timestamp();
        
        // Process incoming spikes through input queue
        for spike in spikes {
            if let Some(ref mut producer) = self.input_producer {
                if producer.enqueue(*spike).is_err() {
                    // Queue full, handle overflow
                    self.statistics.deadline_misses[RTICTaskType::SpikeProcessing as usize]
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        
        let execution_time = self.get_timestamp() - start_time;
        self.update_task_statistics(RTICTaskType::SpikeProcessing as usize, execution_time);
        
        Ok(())
    }
    
    /// Handle timer interrupt (typically called from RTIC interrupt handler)
    pub fn handle_timer_interrupt(&mut self) -> EmbeddedResult<()> {
        self.statistics.interrupt_counts[0].fetch_add(1, Ordering::Relaxed);
        
        // Trigger network update task
        // In a real RTIC application, this would spawn the network update task
        
        Ok(())
    }
    
    /// Handle input interrupt (GPIO spike input)
    pub fn handle_input_interrupt(&mut self, pin: u8, timestamp: T) -> EmbeddedResult<()> {
        let interrupt_idx = (pin % 16) as usize;
        self.statistics.interrupt_counts[interrupt_idx].fetch_add(1, Ordering::Relaxed);
        
        // Create spike from GPIO input
        let spike = FixedSpike::new(pin as u16, timestamp, T::one());
        
        // Queue spike for processing
        if let Some(ref mut producer) = self.input_producer {
            let _ = producer.enqueue(spike);
        }
        
        Ok(())
    }
    
    /// Handle DMA completion interrupt
    pub fn handle_dma_interrupt(&mut self, channel: u8) -> EmbeddedResult<()> {
        let interrupt_idx = (8 + channel % 8) as usize;
        self.statistics.interrupt_counts[interrupt_idx].fetch_add(1, Ordering::Relaxed);
        
        // Process DMA completion
        // This could be used for large spike transfers or weight updates
        
        Ok(())
    }
    
    /// Validate real-time constraints
    pub fn validate_constraints(&self, constraints: &RTConstraints) -> EmbeddedResult<bool> {
        // Check jitter constraints
        let max_jitter = self.calculate_max_jitter();
        if max_jitter > constraints.max_jitter_us {
            return Ok(false);
        }
        
        // Check latency constraints
        let max_latency = self.calculate_max_latency();
        if max_latency > constraints.max_latency_us {
            return Ok(false);
        }
        
        // Check throughput constraints
        let current_throughput = self.calculate_throughput();
        if current_throughput < constraints.required_throughput {
            return Ok(false);
        }
        
        // Check memory constraints
        if !self.validate_memory_constraints(&constraints.memory_constraints) {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Get current scheduler statistics
    pub fn get_statistics(&self) -> RTICSchedulerStats {
        RTICSchedulerStats {
            total_executions: self.statistics.task_executions.iter()
                .map(|count| count.load(Ordering::Relaxed))
                .sum(),
            total_deadline_misses: self.statistics.deadline_misses.iter()
                .map(|count| count.load(Ordering::Relaxed))
                .sum(),
            system_load: self.statistics.system_load.load(Ordering::Relaxed),
            max_stack_usage: self.statistics.max_stack_usage.load(Ordering::Relaxed),
            context_switches: self.statistics.context_switches.load(Ordering::Relaxed),
        }
    }
    
    /// Update task execution statistics
    fn update_task_statistics(&mut self, task_id: usize, execution_time: u32) {
        if task_id < MAX_RTIC_TASKS {
            self.statistics.task_executions[task_id].fetch_add(1, Ordering::Relaxed);
            
            if execution_time > self.statistics.max_execution_times[task_id] {
                self.statistics.max_execution_times[task_id] = execution_time;
            }
            
            // Update average execution time (simple moving average)
            let current_avg = self.statistics.avg_execution_times[task_id];
            let executions = self.statistics.task_executions[task_id].load(Ordering::Relaxed);
            let new_avg = (current_avg * (executions - 1) + execution_time) / executions;
            self.statistics.avg_execution_times[task_id] = new_avg;
        }
    }
    
    /// Get high-resolution timestamp (platform-specific)
    fn get_timestamp(&self) -> u32 {
        // In a real implementation, this would use platform-specific high-resolution timer
        // For now, return a placeholder
        0
    }
    
    /// Calculate maximum jitter
    fn calculate_max_jitter(&self) -> u32 {
        // Calculate jitter based on execution time variations
        let mut max_jitter = 0;
        for i in 0..MAX_RTIC_TASKS {
            let max_time = self.statistics.max_execution_times[i];
            let avg_time = self.statistics.avg_execution_times[i];
            let jitter = if max_time > avg_time { max_time - avg_time } else { 0 };
            if jitter > max_jitter {
                max_jitter = jitter;
            }
        }
        max_jitter
    }
    
    /// Calculate maximum processing latency
    fn calculate_max_latency(&self) -> u32 {
        // Sum of maximum execution times for critical path
        self.statistics.max_execution_times[RTICTaskType::NetworkUpdate as usize] +
        self.statistics.max_execution_times[RTICTaskType::SpikeProcessing as usize] +
        self.statistics.max_execution_times[RTICTaskType::OutputGeneration as usize]
    }
    
    /// Calculate current throughput
    fn calculate_throughput(&self) -> u32 {
        // Calculate spikes processed per second
        let spike_processing_executions = 
            self.statistics.task_executions[RTICTaskType::SpikeProcessing as usize]
                .load(Ordering::Relaxed);
        
        // Assuming each execution processes one spike on average
        spike_processing_executions // This would be scaled by actual time in a real implementation
    }
    
    /// Validate memory constraints
    fn validate_memory_constraints(&self, constraints: &MemoryConstraints) -> bool {
        let current_stack = self.statistics.max_stack_usage.load(Ordering::Relaxed) as usize;
        current_stack <= constraints.max_stack_bytes
    }
}

impl RTICStatistics {
    /// Create new statistics instance
    pub fn new() -> Self {
        Self {
            task_executions: Default::default(),
            deadline_misses: Default::default(),
            max_execution_times: [0; MAX_RTIC_TASKS],
            avg_execution_times: [0; MAX_RTIC_TASKS],
            system_load: AtomicU32::new(0),
            max_stack_usage: AtomicU32::new(0),
            interrupt_counts: Default::default(),
            context_switches: AtomicU32::new(0),
        }
    }
    
    /// Reset all statistics
    pub fn reset(&mut self) {
        for i in 0..MAX_RTIC_TASKS {
            self.task_executions[i].store(0, Ordering::Relaxed);
            self.deadline_misses[i].store(0, Ordering::Relaxed);
            self.max_execution_times[i] = 0;
            self.avg_execution_times[i] = 0;
        }
        
        for i in 0..16 {
            self.interrupt_counts[i].store(0, Ordering::Relaxed);
        }
        
        self.system_load.store(0, Ordering::Relaxed);
        self.max_stack_usage.store(0, Ordering::Relaxed);
        self.context_switches.store(0, Ordering::Relaxed);
    }
}

impl Default for InterruptConfig {
    fn default() -> Self {
        Self {
            timer_interrupt: None,
            input_interrupts: Vec::new(),
            dma_interrupts: Vec::new(),
            system_interrupts: Vec::new(),
        }
    }
}

/// Simplified scheduler statistics for external reporting
#[derive(Debug, Clone)]
pub struct RTICSchedulerStats {
    /// Total task executions
    pub total_executions: u32,
    /// Total deadline misses
    pub total_deadline_misses: u32,
    /// System load percentage (0-100)
    pub system_load: u32,
    /// Maximum stack usage in bytes
    pub max_stack_usage: u32,
    /// Total context switches
    pub context_switches: u32,
}

/// RTIC configuration builder
pub struct RTICConfigBuilder<T: FixedPoint> {
    tasks: Vec<RTICTaskConfig, MAX_RTIC_TASKS>,
    constraints: Option<RTConstraints>,
    interrupt_config: InterruptConfig,
    _phantom: PhantomData<T>,
}

impl<T: FixedPoint> RTICConfigBuilder<T> {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            constraints: None,
            interrupt_config: InterruptConfig::default(),
            _phantom: PhantomData,
        }
    }
    
    /// Add a task configuration
    pub fn add_task(mut self, config: RTICTaskConfig) -> EmbeddedResult<Self> {
        self.tasks.push(config)
            .map_err(|_| EmbeddedError::BufferFull)?;
        Ok(self)
    }
    
    /// Set real-time constraints
    pub fn set_constraints(mut self, constraints: RTConstraints) -> Self {
        self.constraints = Some(constraints);
        self
    }
    
    /// Add timer interrupt
    pub fn add_timer_interrupt(mut self, config: TimerInterruptConfig) -> Self {
        self.interrupt_config.timer_interrupt = Some(config);
        self
    }
    
    /// Add input interrupt
    pub fn add_input_interrupt(mut self, config: InputInterruptConfig) -> EmbeddedResult<Self> {
        self.interrupt_config.input_interrupts.push(config)
            .map_err(|_| EmbeddedError::BufferFull)?;
        Ok(self)
    }
    
    /// Build the configuration
    pub fn build(self) -> RTICConfig<T> {
        RTICConfig {
            tasks: self.tasks,
            constraints: self.constraints,
            interrupt_config: self.interrupt_config,
            _phantom: PhantomData,
        }
    }
}

/// Complete RTIC configuration
pub struct RTICConfig<T: FixedPoint> {
    /// Task configurations
    pub tasks: Vec<RTICTaskConfig, MAX_RTIC_TASKS>,
    /// Real-time constraints
    pub constraints: Option<RTConstraints>,
    /// Interrupt configuration
    pub interrupt_config: InterruptConfig,
    /// Phantom data
    _phantom: PhantomData<T>,
}

impl<T: FixedPoint> Default for RTICConfigBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rtic_scheduler_creation() {
        let scheduler = RTICScheduler::<Q16_16>::new(Q16_16::from_float(0.001));
        assert_eq!(scheduler.current_time, Q16_16::zero());
        assert_eq!(scheduler.time_step, Q16_16::from_float(0.001));
    }
    
    #[test]
    fn test_task_configuration() {
        let task_config = RTICTaskConfig {
            task_id: 0,
            priority: 10,
            task_type: RTICTaskType::NetworkUpdate,
            period_us: 1000,
            max_execution_time_us: 100,
            enable_monitoring: true,
        };
        
        assert_eq!(task_config.task_type, RTICTaskType::NetworkUpdate);
        assert_eq!(task_config.priority, 10);
    }
    
    #[test]
    fn test_interrupt_configuration() {
        let timer_config = TimerInterruptConfig {
            timer_id: 0,
            priority: 15,
            frequency: 1000,
            auto_reload: true,
        };
        
        assert_eq!(timer_config.frequency, 1000);
        assert!(timer_config.auto_reload);
    }
    
    #[test]
    fn test_constraints_validation() {
        let constraints = RTConstraints {
            max_jitter_us: 100,
            min_spike_interval_us: 1,
            max_latency_us: 1000,
            required_throughput: 1000,
            memory_constraints: MemoryConstraints {
                max_heap_bytes: 1024,
                max_stack_bytes: 512,
                dma_buffer_size: 256,
                spike_buffer_size: 64,
            },
        };
        
        assert_eq!(constraints.max_jitter_us, 100);
        assert_eq!(constraints.memory_constraints.max_heap_bytes, 1024);
    }
    
    #[test]
    fn test_rtic_config_builder() {
        let config = RTICConfigBuilder::<Q16_16>::new()
            .add_task(RTICTaskConfig {
                task_id: 0,
                priority: 10,
                task_type: RTICTaskType::NetworkUpdate,
                period_us: 1000,
                max_execution_time_us: 100,
                enable_monitoring: true,
            }).unwrap()
            .build();
        
        assert_eq!(config.tasks.len(), 1);
    }
}