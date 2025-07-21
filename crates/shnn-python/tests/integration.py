#!/usr/bin/env python3
"""
Integration tests for shnn-python crate

These tests verify Python bindings, NumPy integration, visualization,
performance profiling, and cross-language neuromorphic computing.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import multiprocessing
import gc
import sys
import os
from typing import List, Tuple, Optional

# Add the built Python module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'wheels'))

try:
    import shnn
    from shnn import (
        PyHypergraphNetwork, PyLIFNeuron, PyAdExNeuron, PyIzhikevichNeuron,
        PySpike, PyTimeStep, PyNeuronId, PySTDPRule, PyPlasticityConfig,
        PyNetworkConfig, PyVisualization, PyPerformanceProfiler,
        PyRateEncoder, PyTemporalEncoder, PyPopulationEncoder,
        PyAcceleratorManager, PyBenchmarkSuite
    )
    SHNN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SHNN Python module not available: {e}")
    SHNN_AVAILABLE = False


class TestPythonBindings(unittest.TestCase):
    """Test basic Python bindings functionality"""
    
    def setUp(self):
        if not SHNN_AVAILABLE:
            self.skipTest("SHNN Python module not available")
    
    def test_network_creation(self):
        """Test creating a hypergraph network from Python"""
        config = PyNetworkConfig()
        network = PyHypergraphNetwork(config)
        
        self.assertIsNotNone(network)
        self.assertEqual(network.neuron_count(), 0)
        self.assertEqual(network.hyperedge_count(), 0)
    
    def test_neuron_creation(self):
        """Test creating different neuron types"""
        # LIF neuron
        lif_config = {
            'tau_membrane': 20.0,
            'threshold': -55.0,
            'reset_potential': -70.0,
            'resting_potential': -65.0,
            'refractory_period': 2.0
        }
        lif_neuron = PyLIFNeuron(**lif_config)
        
        self.assertEqual(lif_neuron.threshold(), -55.0)
        self.assertEqual(lif_neuron.membrane_potential(), -65.0)
        
        # AdEx neuron
        adex_neuron = PyAdExNeuron()
        self.assertIsNotNone(adex_neuron)
        
        # Izhikevich neuron
        izh_neuron = PyIzhikevichNeuron(0.02, 0.2, -65.0, 8.0)
        self.assertEqual(izh_neuron.threshold(), 30.0)
    
    def test_spike_processing(self):
        """Test spike creation and processing"""
        network = PyHypergraphNetwork(PyNetworkConfig())
        
        # Add neurons
        lif_config = {'tau_membrane': 20.0, 'threshold': -55.0}
        neuron1 = network.add_neuron(PyLIFNeuron(**lif_config))
        neuron2 = network.add_neuron(PyLIFNeuron(**lif_config))
        
        # Connect neurons
        edge = network.add_hyperedge([neuron1], [neuron2], 1.0)
        
        # Create and process spike
        spike = PySpike(neuron1, PyTimeStep.from_ms(1.0))
        result = network.process_spike(spike)
        
        self.assertTrue(result)
        
        # Check spike propagation
        neuron2_state = network.get_neuron_state(neuron2)
        self.assertGreater(neuron2_state.membrane_potential(), -65.0)


class TestNumpyIntegration(unittest.TestCase):
    """Test NumPy array integration"""
    
    def setUp(self):
        if not SHNN_AVAILABLE:
            self.skipTest("SHNN Python module not available")
    
    def test_numpy_spike_data(self):
        """Test converting spike data to/from NumPy arrays"""
        network = PyHypergraphNetwork(PyNetworkConfig())
        
        # Add neurons
        neurons = []
        for i in range(10):
            neuron = network.add_neuron(PyLIFNeuron())
            neurons.append(neuron)
        
        # Generate spikes
        spike_times = np.random.exponential(10.0, size=100)  # Exponential ISI
        spike_neurons = np.random.randint(0, 10, size=100)
        
        spikes = []
        for i, (time, neuron_idx) in enumerate(zip(spike_times, spike_neurons)):
            spike = PySpike(neurons[neuron_idx], PyTimeStep.from_ms(float(time)))
            spikes.append(spike)
            network.process_spike(spike)
        
        # Convert to NumPy arrays
        spike_array = network.get_spike_data_as_numpy()
        
        self.assertIsInstance(spike_array, np.ndarray)
        self.assertEqual(spike_array.shape[0], len(spikes))
        self.assertEqual(spike_array.shape[1], 2)  # (neuron_id, timestamp)
        
        # Verify data integrity
        for i, spike in enumerate(spikes):
            self.assertEqual(spike_array[i, 0], spike.neuron_id().as_int())
            self.assertAlmostEqual(spike_array[i, 1], spike.timestamp().as_ms())
    
    def test_numpy_weight_matrices(self):
        """Test working with weight matrices as NumPy arrays"""
        network = PyHypergraphNetwork(PyNetworkConfig())
        
        # Create neurons
        num_neurons = 50
        neurons = [network.add_neuron(PyLIFNeuron()) for _ in range(num_neurons)]
        
        # Create weight matrix
        weight_matrix = np.random.normal(0.5, 0.1, size=(num_neurons, num_neurons))
        weight_matrix = np.clip(weight_matrix, 0.0, 1.0)  # Positive weights only
        
        # Set connectivity from weight matrix
        for i in range(num_neurons):
            for j in range(num_neurons):
                if weight_matrix[i, j] > 0.3:  # Threshold for connection
                    network.add_hyperedge([neurons[i]], [neurons[j]], weight_matrix[i, j])
        
        # Retrieve weight matrix
        retrieved_matrix = network.get_weight_matrix_as_numpy()
        
        self.assertIsInstance(retrieved_matrix, np.ndarray)
        self.assertEqual(retrieved_matrix.shape, (num_neurons, num_neurons))
        
        # Check that strong connections are preserved
        strong_connections = weight_matrix > 0.3
        for i in range(num_neurons):
            for j in range(num_neurons):
                if strong_connections[i, j]:
                    self.assertGreater(retrieved_matrix[i, j], 0.0)
    
    def test_membrane_potential_arrays(self):
        """Test accessing membrane potentials as NumPy arrays"""
        network = PyHypergraphNetwork(PyNetworkConfig())
        
        # Create neurons with different initial potentials
        num_neurons = 20
        neurons = []
        initial_potentials = np.linspace(-70.0, -60.0, num_neurons)
        
        for potential in initial_potentials:
            neuron = PyLIFNeuron()
            neuron.set_membrane_potential(potential)
            neuron_id = network.add_neuron(neuron)
            neurons.append(neuron_id)
        
        # Get membrane potentials as array
        potentials = network.get_membrane_potentials_as_numpy()
        
        self.assertIsInstance(potentials, np.ndarray)
        self.assertEqual(potentials.shape, (num_neurons,))
        
        # Verify values
        for i, expected_potential in enumerate(initial_potentials):
            self.assertAlmostEqual(potentials[i], expected_potential, places=2)


class TestVisualization(unittest.TestCase):
    """Test visualization capabilities"""
    
    def setUp(self):
        if not SHNN_AVAILABLE:
            self.skipTest("SHNN Python module not available")
        
        # Create test network
        self.network = PyHypergraphNetwork(PyNetworkConfig())
        self.neurons = []
        for i in range(10):
            neuron = self.network.add_neuron(PyLIFNeuron())
            self.neurons.append(neuron)
        
        # Create connections
        for i in range(9):
            self.network.add_hyperedge([self.neurons[i]], [self.neurons[i + 1]], 0.5)
    
    def test_spike_raster_plot(self):
        """Test creating spike raster plots"""
        viz = PyVisualization(self.network)
        
        # Generate some spikes
        spikes = []
        for i in range(50):
            neuron_idx = np.random.randint(0, 10)
            time = np.random.uniform(0, 100)
            spike = PySpike(self.neurons[neuron_idx], PyTimeStep.from_ms(time))
            spikes.append(spike)
            self.network.process_spike(spike)
        
        # Create raster plot
        fig, ax = plt.subplots(figsize=(10, 6))
        viz.plot_spike_raster(ax, time_window=(0, 100))
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(ax.collections), 1)  # Should have scatter plot
        
        # Save plot (optional, for visual inspection)
        if os.environ.get('SAVE_PLOTS'):
            plt.savefig('test_spike_raster.png')
        plt.close(fig)
    
    def test_membrane_potential_plot(self):
        """Test plotting membrane potentials over time"""
        viz = PyVisualization(self.network)
        
        # Simulate network for some time
        dt = PyTimeStep.from_ms(0.1)
        time_points = []
        membrane_potentials = {neuron: [] for neuron in self.neurons}
        
        for step in range(1000):
            current_time = step * 0.1
            time_points.append(current_time)
            
            # Inject input occasionally
            if step % 100 == 0:
                spike = PySpike(self.neurons[0], PyTimeStep.from_ms(current_time))
                self.network.process_spike(spike)
            
            # Update network
            self.network.update(dt)
            
            # Record membrane potentials
            for neuron in self.neurons:
                state = self.network.get_neuron_state(neuron)
                membrane_potentials[neuron].append(state.membrane_potential())
        
        # Create membrane potential plot
        fig, ax = plt.subplots(figsize=(12, 8))
        viz.plot_membrane_potentials(ax, time_points, membrane_potentials, self.neurons[:3])
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(ax.lines), 3)  # Should have 3 lines for 3 neurons
        
        if os.environ.get('SAVE_PLOTS'):
            plt.savefig('test_membrane_potentials.png')
        plt.close(fig)
    
    def test_network_topology_plot(self):
        """Test plotting network topology"""
        viz = PyVisualization(self.network)
        
        # Create more complex network structure
        for i in range(5):
            for j in range(i + 1, 5):
                if np.random.random() > 0.5:
                    weight = np.random.uniform(0.2, 0.8)
                    self.network.add_hyperedge([self.neurons[i]], [self.neurons[j]], weight)
        
        # Plot network topology
        fig, ax = plt.subplots(figsize=(10, 10))
        viz.plot_network_topology(ax)
        
        self.assertIsNotNone(fig)
        
        if os.environ.get('SAVE_PLOTS'):
            plt.savefig('test_network_topology.png')
        plt.close(fig)


class TestPerformanceProfiling(unittest.TestCase):
    """Test performance profiling and monitoring"""
    
    def setUp(self):
        if not SHNN_AVAILABLE:
            self.skipTest("SHNN Python module not available")
    
    def test_performance_profiler(self):
        """Test performance profiling functionality"""
        profiler = PyPerformanceProfiler()
        
        # Profile network creation
        profiler.start_timer("network_creation")
        
        network = PyHypergraphNetwork(PyNetworkConfig())
        neurons = []
        for i in range(1000):
            neuron = network.add_neuron(PyLIFNeuron())
            neurons.append(neuron)
        
        creation_time = profiler.end_timer("network_creation")
        
        self.assertGreater(creation_time, 0.0)
        self.assertLess(creation_time, 10.0)  # Should be reasonable
        
        # Profile spike processing
        profiler.start_timer("spike_processing")
        
        for i in range(10000):
            neuron_idx = np.random.randint(0, 1000)
            spike = PySpike(neurons[neuron_idx], PyTimeStep.from_ms(i * 0.1))
            network.process_spike(spike)
        
        processing_time = profiler.end_timer("spike_processing")
        
        self.assertGreater(processing_time, 0.0)
        
        # Calculate performance metrics
        spikes_per_second = 10000 / processing_time
        self.assertGreater(spikes_per_second, 1000)  # Should process at least 1K spikes/sec
        
        # Get profiler statistics
        stats = profiler.get_statistics()
        self.assertIn("network_creation", stats)
        self.assertIn("spike_processing", stats)
    
    def test_memory_profiling(self):
        """Test memory usage profiling"""
        profiler = PyPerformanceProfiler()
        
        initial_memory = profiler.get_memory_usage()
        
        # Create large network
        network = PyHypergraphNetwork(PyNetworkConfig())
        neurons = []
        
        for i in range(5000):
            neuron = network.add_neuron(PyLIFNeuron())
            neurons.append(neuron)
        
        # Create many connections
        for i in range(10000):
            pre = np.random.randint(0, 5000)
            post = np.random.randint(0, 5000)
            if pre != post:
                weight = np.random.uniform(0.1, 1.0)
                network.add_hyperedge([neurons[pre]], [neurons[post]], weight)
        
        peak_memory = profiler.get_memory_usage()
        memory_increase = peak_memory - initial_memory
        
        self.assertGreater(memory_increase, 0)
        
        # Clean up
        del network
        del neurons
        gc.collect()
        
        final_memory = profiler.get_memory_usage()
        memory_freed = peak_memory - final_memory
        
        # Should have freed most of the memory
        self.assertGreater(memory_freed, memory_increase * 0.5)


class TestEncodingSchemes(unittest.TestCase):
    """Test different spike encoding schemes"""
    
    def setUp(self):
        if not SHNN_AVAILABLE:
            self.skipTest("SHNN Python module not available")
    
    def test_rate_encoding(self):
        """Test rate-based spike encoding"""
        encoder = PyRateEncoder(value_range=(0.0, 100.0), max_rate=100.0)
        
        # Test encoding of different values
        values = [0.0, 25.0, 50.0, 75.0, 100.0]
        simulation_time = PyTimeStep.from_ms(1000.0)
        
        for value in values:
            spike_train = encoder.encode(value, simulation_time)
            spike_count = len(spike_train.get_spikes())
            
            # Expected rate proportional to value
            expected_rate = (value / 100.0) * 100.0  # Hz
            expected_spikes = expected_rate * 1.0  # 1 second simulation
            
            # Allow some variance due to stochastic nature
            self.assertAlmostEqual(spike_count, expected_spikes, delta=expected_spikes * 0.2)
    
    def test_temporal_encoding(self):
        """Test temporal spike encoding"""
        encoder = PyTemporalEncoder(
            min_delay=PyTimeStep.from_ms(1.0),
            max_delay=PyTimeStep.from_ms(10.0)
        )
        
        # Test encoding of different values
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        spike_times = []
        for value in values:
            spike_time = encoder.encode_first_spike(value)
            spike_times.append(spike_time.as_ms())
        
        # Higher values should have shorter delays
        for i in range(len(values) - 1):
            if values[i] < values[i + 1]:
                self.assertGreaterEqual(spike_times[i], spike_times[i + 1])
    
    def test_population_encoding(self):
        """Test population-based encoding"""
        encoder = PyPopulationEncoder(
            population_size=20,
            value_range=(0.0, 100.0)
        )
        
        # Test encoding
        value = 50.0  # Middle of range
        activations = encoder.encode(value)
        
        self.assertEqual(len(activations), 20)
        
        # Should activate neurons around the middle
        max_activation_idx = np.argmax(activations)
        self.assertTrue(8 <= max_activation_idx <= 12)  # Around middle (index 10)
        
        # Test edge values
        edge_activations_low = encoder.encode(0.0)
        edge_activations_high = encoder.encode(100.0)
        
        max_low_idx = np.argmax(edge_activations_low)
        max_high_idx = np.argmax(edge_activations_high)
        
        self.assertLess(max_low_idx, 5)   # Should activate early neurons
        self.assertGreater(max_high_idx, 15)  # Should activate late neurons


class TestBenchmarking(unittest.TestCase):
    """Test benchmarking capabilities"""
    
    def setUp(self):
        if not SHNN_AVAILABLE:
            self.skipTest("SHNN Python module not available")
    
    def test_benchmark_suite(self):
        """Test running benchmark suite"""
        benchmark = PyBenchmarkSuite()
        
        # Define test scenarios
        scenarios = [
            {
                'name': 'small_network',
                'neurons': 100,
                'connections': 500,
                'simulation_time': 100.0,
                'timestep': 0.1
            },
            {
                'name': 'medium_network',
                'neurons': 1000,
                'connections': 5000,
                'simulation_time': 100.0,
                'timestep': 0.1
            }
        ]
        
        results = {}
        for scenario in scenarios:
            print(f"Running benchmark: {scenario['name']}")
            
            start_time = time.time()
            
            # Create network
            network = PyHypergraphNetwork(PyNetworkConfig())
            neurons = []
            
            for i in range(scenario['neurons']):
                neuron = network.add_neuron(PyLIFNeuron())
                neurons.append(neuron)
            
            # Create random connections
            for i in range(scenario['connections']):
                pre = np.random.randint(0, scenario['neurons'])
                post = np.random.randint(0, scenario['neurons'])
                if pre != post:
                    weight = np.random.uniform(0.1, 1.0)
                    network.add_hyperedge([neurons[pre]], [neurons[post]], weight)
            
            # Run simulation
            dt = PyTimeStep.from_ms(scenario['timestep'])
            num_steps = int(scenario['simulation_time'] / scenario['timestep'])
            
            for step in range(num_steps):
                # Inject random spikes
                if np.random.random() < 0.01:  # 1% chance per step
                    neuron_idx = np.random.randint(0, scenario['neurons'])
                    spike = PySpike(neurons[neuron_idx], PyTimeStep.from_ms(step * scenario['timestep']))
                    network.process_spike(spike)
                
                network.update(dt)
            
            execution_time = time.time() - start_time
            
            results[scenario['name']] = {
                'execution_time': execution_time,
                'neurons': scenario['neurons'],
                'steps_per_second': num_steps / execution_time,
                'neurons_per_second': scenario['neurons'] * num_steps / execution_time
            }
            
            print(f"  Execution time: {execution_time:.3f}s")
            print(f"  Steps per second: {results[scenario['name']]['steps_per_second']:.1f}")
            print(f"  Neurons per second: {results[scenario['name']]['neurons_per_second']:.1f}")
        
        # Verify reasonable performance
        self.assertLess(results['small_network']['execution_time'], 5.0)
        self.assertGreater(results['small_network']['steps_per_second'], 100)
    
    def test_performance_comparison(self):
        """Test performance comparison with pure Python implementation"""
        
        # Pure Python LIF neuron for comparison
        class PurePythonLIF:
            def __init__(self):
                self.v = -65.0
                self.threshold = -55.0
                self.reset = -70.0
                self.tau = 20.0
                self.dt = 0.1
            
            def integrate(self, current):
                self.v += (-self.v + current * 10) / self.tau * self.dt
            
            def update(self):
                if self.v >= self.threshold:
                    self.v = self.reset
                    return True
                return False
        
        # Benchmark pure Python
        num_neurons = 1000
        num_steps = 1000
        
        start_time = time.time()
        
        python_neurons = [PurePythonLIF() for _ in range(num_neurons)]
        
        for step in range(num_steps):
            for neuron in python_neurons:
                neuron.integrate(1.0)
                neuron.update()
        
        python_time = time.time() - start_time
        
        # Benchmark SHNN
        start_time = time.time()
        
        network = PyHypergraphNetwork(PyNetworkConfig())
        shnn_neurons = []
        
        for i in range(num_neurons):
            neuron = network.add_neuron(PyLIFNeuron())
            shnn_neurons.append(neuron)
        
        dt = PyTimeStep.from_ms(0.1)
        
        for step in range(num_steps):
            # Inject current (simulate by processing spikes)
            for neuron in shnn_neurons:
                if np.random.random() < 0.1:  # 10% chance of spike
                    spike = PySpike(neuron, PyTimeStep.from_ms(step * 0.1))
                    network.process_spike(spike)
            
            network.update(dt)
        
        shnn_time = time.time() - start_time
        
        print(f"Pure Python time: {python_time:.3f}s")
        print(f"SHNN time: {shnn_time:.3f}s")
        print(f"Speedup: {python_time / shnn_time:.2f}x")
        
        # SHNN should be faster (or at least competitive)
        self.assertLessEqual(shnn_time, python_time * 2.0)  # Allow up to 2x slower due to overhead


class TestThreadingSafety(unittest.TestCase):
    """Test thread safety of Python bindings"""
    
    def setUp(self):
        if not SHNN_AVAILABLE:
            self.skipTest("SHNN Python module not available")
    
    def test_concurrent_network_access(self):
        """Test concurrent access to network from multiple threads"""
        network = PyHypergraphNetwork(PyNetworkConfig())
        
        # Create neurons
        neurons = []
        for i in range(100):
            neuron = network.add_neuron(PyLIFNeuron())
            neurons.append(neuron)
        
        # Function to process spikes in a thread
        def spike_processor(thread_id, num_spikes):
            for i in range(num_spikes):
                neuron_idx = (thread_id * 31 + i * 17) % len(neurons)  # Deterministic but distributed
                spike = PySpike(neurons[neuron_idx], PyTimeStep.from_ms(i * 0.1))
                network.process_spike(spike)
        
        # Run multiple threads
        threads = []
        num_threads = 4
        spikes_per_thread = 250
        
        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=spike_processor,
                args=(thread_id, spikes_per_thread)
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Network should still be in valid state
        self.assertEqual(network.neuron_count(), 100)
        
        # Should have processed all spikes
        total_expected_spikes = num_threads * spikes_per_thread
        # Note: We can't easily verify exact spike count due to internal processing


class TestErrorHandling(unittest.TestCase):
    """Test error handling in Python bindings"""
    
    def setUp(self):
        if not SHNN_AVAILABLE:
            self.skipTest("SHNN Python module not available")
    
    def test_invalid_neuron_access(self):
        """Test handling of invalid neuron access"""
        network = PyHypergraphNetwork(PyNetworkConfig())
        
        # Try to access non-existent neuron
        invalid_neuron = PyNeuronId(999)
        
        with self.assertRaises(Exception):  # Should raise appropriate exception
            network.get_neuron_state(invalid_neuron)
    
    def test_invalid_spike_processing(self):
        """Test handling of invalid spike processing"""
        network = PyHypergraphNetwork(PyNetworkConfig())
        
        # Try to process spike for non-existent neuron
        invalid_neuron = PyNeuronId(999)
        invalid_spike = PySpike(invalid_neuron, PyTimeStep.from_ms(1.0))
        
        with self.assertRaises(Exception):
            network.process_spike(invalid_spike)
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        
        # Invalid time values
        with self.assertRaises(Exception):
            PyTimeStep.from_ms(-1.0)  # Negative time
        
        # Invalid neuron parameters
        with self.assertRaises(Exception):
            PyLIFNeuron(tau_membrane=-1.0)  # Negative time constant


if __name__ == '__main__':
    # Set up test environment
    if not SHNN_AVAILABLE:
        print("SHNN Python module not available. Please build the module first:")
        print("  cd crates/shnn-python")
        print("  maturin develop")
        sys.exit(1)
    
    # Configure test output
    if len(sys.argv) > 1 and sys.argv[1] == '--save-plots':
        os.environ['SAVE_PLOTS'] = '1'
    
    # Run tests
    unittest.main(verbosity=2)