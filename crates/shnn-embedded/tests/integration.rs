//! Integration tests for shnn-embedded crate
//! 
//! These tests verify no-std compatibility, embedded system support,
//! fixed-point arithmetic, and real-time deterministic behavior.

#![no_std]
#![no_main]

extern crate alloc;
use alloc::vec::Vec;
use alloc::boxed::Box;

use shnn_embedded::prelude::*;
use shnn_embedded::error::EmbeddedShnnError;
use shnn_embedded::neuron::{EmbeddedLIFNeuron, EmbeddedNeuronId, EmbeddedNeuronState};
use shnn_embedded::network::{EmbeddedHypergraphNetwork, EmbeddedNetworkConfig};
use shnn_embedded::spike::{EmbeddedSpike, EmbeddedSpikeBuffer};
use shnn_embedded::time::{EmbeddedTimeStep, FixedPoint};
use shnn_embedded::memory::{StaticAllocator, MemoryPool, EMBEDDED_HEAP_SIZE};
use shnn_embedded::hal::{EmbeddedTimer, EmbeddedGpio, EmbeddedAdc};

// Custom test harness for no-std environment
use panic_halt as _;

#[cfg(test)]
mod tests {
    use super::*;
    use cortex_m_rt::entry;
    
    #[test]
    fn test_fixed_point_arithmetic() {
        let a = FixedPoint::from_f64(3.14159);
        let b = FixedPoint::from_f64(2.71828);
        
        // Test arithmetic operations
        let sum = a + b;
        let diff = a - b;
        let product = a * b;
        let quotient = a / b;
        
        // Verify precision
        assert!((sum.to_f64() - 5.85987).abs() < 0.001);
        assert!((diff.to_f64() - 0.42331).abs() < 0.001);
        assert!((product.to_f64() - 8.53974).abs() < 0.001);
        assert!((quotient.to_f64() - 1.15573).abs() < 0.001);
        
        // Test edge cases
        let zero = FixedPoint::ZERO;
        let one = FixedPoint::ONE;
        let max_val = FixedPoint::MAX;
        
        assert_eq!(zero.to_f64(), 0.0);
        assert_eq!(one.to_f64(), 1.0);
        assert!(max_val.to_f64() > 1000.0);
        
        // Test overflow protection
        let result = max_val + one;
        assert_eq!(result, max_val); // Should saturate
    }
    
    #[test]
    fn test_embedded_time_operations() {
        let time1 = EmbeddedTimeStep::from_ms(FixedPoint::from_f64(10.5));
        let time2 = EmbeddedTimeStep::from_ms(FixedPoint::from_f64(5.25));
        
        // Test arithmetic
        let sum = time1 + time2;
        let diff = time1 - time2;
        
        assert!((sum.as_ms().to_f64() - 15.75).abs() < 0.01);
        assert!((diff.as_ms().to_f64() - 5.25).abs() < 0.01);
        
        // Test comparisons
        assert!(time1 > time2);
        assert!(time2 < time1);
        assert_eq!(time1, time1);
        
        // Test microsecond precision
        let us_time = EmbeddedTimeStep::from_us(FixedPoint::from_f64(1500.0));
        assert!((us_time.as_ms().to_f64() - 1.5).abs() < 0.001);
    }
    
    #[test]
    fn test_embedded_lif_neuron() {
        let config = EmbeddedNeuronConfig {
            tau_membrane: FixedPoint::from_f64(20.0),
            threshold: FixedPoint::from_f64(-55.0),
            reset_potential: FixedPoint::from_f64(-70.0),
            resting_potential: FixedPoint::from_f64(-65.0),
            refractory_period: FixedPoint::from_f64(2.0),
        };
        
        let mut neuron = EmbeddedLIFNeuron::new(EmbeddedNeuronId(0), config);
        
        // Test initial state
        assert_eq!(neuron.id(), EmbeddedNeuronId(0));
        assert!((neuron.membrane_potential().to_f64() + 65.0).abs() < 0.1);
        assert!(!neuron.is_refractory());
        
        // Test integration
        let dt = EmbeddedTimeStep::from_ms(FixedPoint::from_f64(0.1));
        let input_current = FixedPoint::from_f64(1.0);
        
        neuron.integrate(input_current, dt);
        assert!(neuron.membrane_potential() > FixedPoint::from_f64(-65.0));
        
        // Test spike generation
        neuron.set_membrane_potential(FixedPoint::from_f64(-50.0));
        let spike_result = neuron.update(dt);
        
        assert!(spike_result.is_some());
        assert!((neuron.membrane_potential().to_f64() + 70.0).abs() < 0.1); // Reset
        assert!(neuron.is_refractory());
    }
    
    #[test]
    fn test_embedded_network_creation() {
        let config = EmbeddedNetworkConfig {
            max_neurons: 100,
            max_hyperedges: 200,
            max_spikes_per_step: 50,
            use_static_allocation: true,
        };
        
        let mut network = EmbeddedHypergraphNetwork::new(config);
        assert!(network.is_ok());
        
        let mut network = network.unwrap();
        
        // Test capacity limits
        assert_eq!(network.max_neurons(), 100);
        assert_eq!(network.max_hyperedges(), 200);
        assert_eq!(network.neuron_count(), 0);
        assert_eq!(network.hyperedge_count(), 0);
    }
    
    #[test]
    fn test_embedded_network_operations() {
        let config = EmbeddedNetworkConfig::default();
        let mut network = EmbeddedHypergraphNetwork::new(config).unwrap();
        
        // Add neurons
        let neuron_config = EmbeddedNeuronConfig::default();
        let neuron1 = network.add_neuron(neuron_config.clone()).unwrap();
        let neuron2 = network.add_neuron(neuron_config).unwrap();
        
        assert_eq!(network.neuron_count(), 2);
        
        // Connect neurons
        let weight = FixedPoint::from_f64(0.5);
        let edge = network.add_hyperedge(
            &[neuron1],
            &[neuron2],
            weight
        ).unwrap();
        
        assert_eq!(network.hyperedge_count(), 1);
        
        // Process spike
        let spike = EmbeddedSpike::new(neuron1, EmbeddedTimeStep::zero());
        let result = network.process_spike(spike);
        assert!(result.is_ok());
        
        // Check propagation
        let neuron2_state = network.get_neuron_state(neuron2).unwrap();
        assert!(neuron2_state.membrane_potential() > FixedPoint::from_f64(-65.0));
    }
    
    #[test]
    fn test_static_memory_allocation() {
        let allocator = StaticAllocator::new();
        
        // Test allocation within limits
        let ptr1 = allocator.allocate(1024);
        assert!(ptr1.is_some());
        
        let ptr2 = allocator.allocate(2048);
        assert!(ptr2.is_some());
        
        // Test deallocation
        allocator.deallocate(ptr1.unwrap(), 1024);
        allocator.deallocate(ptr2.unwrap(), 2048);
        
        // Test allocation after deallocation
        let ptr3 = allocator.allocate(4096);
        assert!(ptr3.is_some());
        
        allocator.deallocate(ptr3.unwrap(), 4096);
    }
    
    #[test]
    fn test_memory_pool_management() {
        let mut pool = MemoryPool::new(8192); // 8KB pool
        
        // Test block allocation
        let block1 = pool.allocate_block(1024);
        assert!(block1.is_some());
        
        let block2 = pool.allocate_block(2048);
        assert!(block2.is_some());
        
        // Test pool statistics
        assert!(pool.used_memory() >= 3072);
        assert!(pool.free_memory() <= 5120);
        assert!(pool.fragmentation_ratio() >= 0.0);
        
        // Test block deallocation
        pool.deallocate_block(block1.unwrap()).unwrap();
        pool.deallocate_block(block2.unwrap()).unwrap();
        
        // Test pool compaction
        pool.compact().unwrap();
        assert_eq!(pool.used_memory(), 0);
    }
    
    #[test]
    fn test_spike_buffer_operations() {
        let mut buffer = EmbeddedSpikeBuffer::new(100); // 100 spike capacity
        
        // Test spike addition
        for i in 0..50 {
            let spike = EmbeddedSpike::new(
                EmbeddedNeuronId(i),
                EmbeddedTimeStep::from_ms(FixedPoint::from_f64(i as f64))
            );
            assert!(buffer.add_spike(spike).is_ok());
        }
        
        assert_eq!(buffer.len(), 50);
        assert!(!buffer.is_full());
        
        // Test spike retrieval
        let spikes = buffer.get_spikes_in_window(
            EmbeddedTimeStep::from_ms(FixedPoint::from_f64(10.0)),
            EmbeddedTimeStep::from_ms(FixedPoint::from_f64(20.0))
        );
        
        assert_eq!(spikes.len(), 11); // Spikes 10-20 inclusive
        
        // Test buffer overflow
        for i in 50..150 {
            let spike = EmbeddedSpike::new(
                EmbeddedNeuronId(i),
                EmbeddedTimeStep::from_ms(FixedPoint::from_f64(i as f64))
            );
            let result = buffer.add_spike(spike);
            
            if i >= 100 {
                assert!(result.is_err()); // Should fail when full
            }
        }
        
        assert!(buffer.is_full());
        
        // Test buffer clearing
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(!buffer.is_full());
    }
    
    #[test]
    fn test_deterministic_behavior() {
        let config = EmbeddedNetworkConfig::default();
        let mut network1 = EmbeddedHypergraphNetwork::new(config.clone()).unwrap();
        let mut network2 = EmbeddedHypergraphNetwork::new(config).unwrap();
        
        // Create identical networks
        let neuron_config = EmbeddedNeuronConfig::default();
        for _ in 0..10 {
            network1.add_neuron(neuron_config.clone()).unwrap();
            network2.add_neuron(neuron_config.clone()).unwrap();
        }
        
        // Create identical connections
        for i in 0..9 {
            let neuron_i = EmbeddedNeuronId(i);
            let neuron_next = EmbeddedNeuronId(i + 1);
            let weight = FixedPoint::from_f64(0.5);
            
            network1.add_hyperedge(&[neuron_i], &[neuron_next], weight).unwrap();
            network2.add_hyperedge(&[neuron_i], &[neuron_next], weight).unwrap();
        }
        
        // Process identical spike sequences
        let dt = EmbeddedTimeStep::from_ms(FixedPoint::from_f64(0.1));
        
        for step in 0..100 {
            // Inject identical input
            if step % 10 == 0 {
                let spike = EmbeddedSpike::new(
                    EmbeddedNeuronId(0),
                    EmbeddedTimeStep::from_ms(FixedPoint::from_f64(step as f64 * 0.1))
                );
                network1.process_spike(spike.clone()).unwrap();
                network2.process_spike(spike).unwrap();
            }
            
            // Update both networks
            network1.update(dt).unwrap();
            network2.update(dt).unwrap();
        }
        
        // Compare final states - should be identical
        for i in 0..10 {
            let neuron_id = EmbeddedNeuronId(i);
            let state1 = network1.get_neuron_state(neuron_id).unwrap();
            let state2 = network2.get_neuron_state(neuron_id).unwrap();
            
            // States should be exactly equal due to deterministic fixed-point arithmetic
            assert_eq!(state1.membrane_potential(), state2.membrane_potential());
            assert_eq!(state1.refractory_timer(), state2.refractory_timer());
        }
    }
    
    #[test]
    fn test_real_time_constraints() {
        let config = EmbeddedNetworkConfig {
            max_neurons: 50,
            max_hyperedges: 100,
            max_spikes_per_step: 25,
            use_static_allocation: true,
        };
        
        let mut network = EmbeddedHypergraphNetwork::new(config).unwrap();
        
        // Create network
        let neuron_config = EmbeddedNeuronConfig::default();
        let mut neurons = Vec::new();
        for i in 0..50 {
            let neuron = network.add_neuron(neuron_config.clone()).unwrap();
            neurons.push(neuron);
        }
        
        // Create connections
        for i in 0..49 {
            network.add_hyperedge(
                &[neurons[i]],
                &[neurons[i + 1]],
                FixedPoint::from_f64(0.5)
            ).unwrap();
        }
        
        // Measure processing time
        let timer = EmbeddedTimer::new();
        let start_time = timer.current_time();
        
        // Process spikes for 1000 time steps
        let dt = EmbeddedTimeStep::from_ms(FixedPoint::from_f64(0.1));
        
        for step in 0..1000 {
            // Inject periodic input
            if step % 50 == 0 {
                let spike = EmbeddedSpike::new(
                    neurons[0],
                    EmbeddedTimeStep::from_ms(FixedPoint::from_f64(step as f64 * 0.1))
                );
                network.process_spike(spike).unwrap();
            }
            
            // Update network
            network.update(dt).unwrap();
        }
        
        let end_time = timer.current_time();
        let processing_time = end_time - start_time;
        
        // Should complete in real-time (100ms simulation in <100ms real time)
        assert!(processing_time.as_ms() < FixedPoint::from_f64(100.0));
        
        // Calculate performance metrics
        let steps_per_ms = FixedPoint::from_f64(1000.0) / processing_time.as_ms();
        assert!(steps_per_ms > FixedPoint::from_f64(10.0)); // At least 10 steps per ms
    }
    
    #[test]
    fn test_hardware_abstraction_layer() {
        // Test GPIO operations
        let mut gpio = EmbeddedGpio::new();
        
        gpio.set_pin_mode(0, PinMode::Output).unwrap();
        gpio.set_pin_mode(1, PinMode::Input).unwrap();
        
        gpio.write_pin(0, PinState::High).unwrap();
        let pin_state = gpio.read_pin(1).unwrap();
        assert!(matches!(pin_state, PinState::High | PinState::Low));
        
        // Test ADC operations
        let mut adc = EmbeddedAdc::new();
        
        adc.configure_channel(0, AdcResolution::Bits12).unwrap();
        let reading = adc.read_channel(0).unwrap();
        assert!(reading <= 4095); // 12-bit max value
        
        // Test timer operations
        let timer = EmbeddedTimer::new();
        let start = timer.current_time();
        
        // Simulate some delay
        for _ in 0..1000 {
            core::hint::spin_loop();
        }
        
        let end = timer.current_time();
        assert!(end > start);
    }
    
    #[test]
    fn test_interrupt_safe_operations() {
        let config = EmbeddedNetworkConfig::default();
        let mut network = EmbeddedHypergraphNetwork::new(config).unwrap();
        
        // Test that network operations are interrupt-safe
        cortex_m::interrupt::free(|_cs| {
            let neuron_config = EmbeddedNeuronConfig::default();
            let neuron = network.add_neuron(neuron_config).unwrap();
            
            let spike = EmbeddedSpike::new(neuron, EmbeddedTimeStep::zero());
            network.process_spike(spike).unwrap();
            
            let dt = EmbeddedTimeStep::from_ms(FixedPoint::from_f64(0.1));
            network.update(dt).unwrap();
        });
        
        // Network should remain in valid state
        assert_eq!(network.neuron_count(), 1);
    }
    
    #[test]
    fn test_power_consumption_estimation() {
        let mut network = EmbeddedHypergraphNetwork::new(EmbeddedNetworkConfig::default()).unwrap();
        
        // Enable power monitoring
        network.enable_power_monitoring().unwrap();
        
        // Create small network
        let neuron_config = EmbeddedNeuronConfig::default();
        let neuron1 = network.add_neuron(neuron_config.clone()).unwrap();
        let neuron2 = network.add_neuron(neuron_config).unwrap();
        
        network.add_hyperedge(
            &[neuron1],
            &[neuron2],
            FixedPoint::from_f64(1.0)
        ).unwrap();
        
        // Simulate activity
        let dt = EmbeddedTimeStep::from_ms(FixedPoint::from_f64(1.0));
        
        for i in 0..100 {
            if i % 10 == 0 {
                let spike = EmbeddedSpike::new(neuron1, EmbeddedTimeStep::from_ms(FixedPoint::from_f64(i as f64)));
                network.process_spike(spike).unwrap();
            }
            
            network.update(dt).unwrap();
        }
        
        // Get power consumption estimate
        let power_stats = network.get_power_statistics().unwrap();
        
        assert!(power_stats.average_power_mw > FixedPoint::ZERO);
        assert!(power_stats.peak_power_mw >= power_stats.average_power_mw);
        assert!(power_stats.total_energy_uj > FixedPoint::ZERO);
        
        // Power consumption should be reasonable for embedded system
        assert!(power_stats.average_power_mw < FixedPoint::from_f64(10.0)); // <10mW
    }
    
    #[test]
    fn test_error_recovery() {
        let config = EmbeddedNetworkConfig::default();
        let mut network = EmbeddedHypergraphNetwork::new(config).unwrap();
        
        // Test invalid neuron ID
        let invalid_spike = EmbeddedSpike::new(
            EmbeddedNeuronId(999),
            EmbeddedTimeStep::zero()
        );
        
        let result = network.process_spike(invalid_spike);
        assert!(matches!(result, Err(EmbeddedShnnError::InvalidNeuronId(_))));
        
        // Network should recover and continue working
        let neuron_config = EmbeddedNeuronConfig::default();
        let neuron = network.add_neuron(neuron_config).unwrap();
        
        let valid_spike = EmbeddedSpike::new(neuron, EmbeddedTimeStep::zero());
        let result = network.process_spike(valid_spike);
        assert!(result.is_ok());
    }
}

// Test entry point for embedded target
#[cfg(not(test))]
#[entry]
fn main() -> ! {
    // Initialize hardware
    let config = EmbeddedNetworkConfig::default();
    let mut network = EmbeddedHypergraphNetwork::new(config).unwrap();
    
    // Create simple network
    let neuron_config = EmbeddedNeuronConfig::default();
    let neuron = network.add_neuron(neuron_config).unwrap();
    
    // Main loop
    let dt = EmbeddedTimeStep::from_ms(FixedPoint::from_f64(1.0));
    let mut step_counter = 0u32;
    
    loop {
        // Inject periodic stimulation
        if step_counter % 1000 == 0 {
            let spike = EmbeddedSpike::new(
                neuron,
                EmbeddedTimeStep::from_ms(FixedPoint::from_f64(step_counter as f64))
            );
            let _ = network.process_spike(spike);
        }
        
        // Update network
        let _ = network.update(dt);
        
        step_counter = step_counter.wrapping_add(1);
        
        // Brief delay to prevent watchdog timeout
        for _ in 0..1000 {
            cortex_m::asm::nop();
        }
    }
}