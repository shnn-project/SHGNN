//! Integration tests for shnn-wasm crate
//! 
//! These tests verify WebAssembly compilation, JavaScript bindings,
//! browser compatibility, and web-based neuromorphic computing.

use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;
use shnn_wasm::prelude::*;
use shnn_wasm::network::WasmHypergraphNetwork;
use shnn_wasm::neuron::{WasmLIFNeuron, WasmNeuronId};
use shnn_wasm::spike::{WasmSpike, WasmSpikeId};
use shnn_wasm::visualization::{WasmNetworkRenderer, WasmSpikeRenderer};
use shnn_wasm::bindings::{JSNeuronConfig, JSNetworkConfig, JSVisualizationConfig};
use shnn_wasm::utils::{WasmTimeStep, WasmLogger};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_wasm_network_creation() {
    let config = JSNetworkConfig::default();
    let network = WasmHypergraphNetwork::new(config);
    
    assert!(network.is_ok());
    let network = network.unwrap();
    
    assert_eq!(network.neuron_count(), 0);
    assert_eq!(network.hyperedge_count(), 0);
    assert!(network.is_initialized());
}

#[wasm_bindgen_test]
fn test_wasm_neuron_creation() {
    let config = JSNeuronConfig {
        neuron_type: "LIF".to_string(),
        tau_membrane: 20.0,
        threshold: -55.0,
        reset_potential: -70.0,
        resting_potential: -65.0,
        refractory_period: 2.0,
    };
    
    let neuron = WasmLIFNeuron::new(config);
    assert!(neuron.is_ok());
    
    let neuron = neuron.unwrap();
    assert_eq!(neuron.threshold(), -55.0);
    assert_eq!(neuron.membrane_potential(), -65.0);
}

#[wasm_bindgen_test]
fn test_wasm_spike_processing() {
    let mut network = WasmHypergraphNetwork::new(JSNetworkConfig::default()).unwrap();
    
    // Add neurons
    let neuron1_config = JSNeuronConfig::default();
    let neuron2_config = JSNeuronConfig::default();
    
    let neuron1 = network.add_neuron(neuron1_config).unwrap();
    let neuron2 = network.add_neuron(neuron2_config).unwrap();
    
    // Connect neurons
    let edge = network.add_hyperedge(
        vec![neuron1],
        vec![neuron2],
        1.0
    ).unwrap();
    
    // Create and process spike
    let spike = WasmSpike::new(neuron1, WasmTimeStep::from_ms(1.0));
    let result = network.process_spike(spike);
    
    assert!(result.is_ok());
    
    // Check that spike was propagated
    let neuron2_state = network.get_neuron_state(neuron2).unwrap();
    assert!(neuron2_state.membrane_potential() > -65.0);
}

#[wasm_bindgen_test]
fn test_wasm_time_operations() {
    let time1 = WasmTimeStep::from_ms(10.0);
    let time2 = WasmTimeStep::from_ms(5.0);
    
    // Test arithmetic
    let sum = time1.add(&time2);
    assert_eq!(sum.as_ms(), 15.0);
    
    let diff = time1.subtract(&time2);
    assert_eq!(diff.as_ms(), 5.0);
    
    // Test comparisons
    assert!(time1.greater_than(&time2));
    assert!(time2.less_than(&time1));
    
    // Test conversions
    assert_eq!(time1.as_us(), 10000.0);
    assert_eq!(time1.as_s(), 0.01);
}

#[wasm_bindgen_test]
fn test_wasm_network_simulation() {
    let mut network = WasmHypergraphNetwork::new(JSNetworkConfig::default()).unwrap();
    
    // Create a small network
    let mut neurons = vec![];
    for i in 0..5 {
        let config = JSNeuronConfig::default();
        let neuron = network.add_neuron(config).unwrap();
        neurons.push(neuron);
    }
    
    // Create connections
    for i in 0..4 {
        network.add_hyperedge(
            vec![neurons[i]],
            vec![neurons[i + 1]],
            0.8
        ).unwrap();
    }
    
    // Run simulation for multiple time steps
    let dt = WasmTimeStep::from_ms(0.1);
    let mut spike_count = 0;
    
    for step in 0..1000 {
        // Inject input to first neuron occasionally
        if step % 100 == 0 {
            let input_spike = WasmSpike::new(neurons[0], WasmTimeStep::from_ms(step as f64 * 0.1));
            network.process_spike(input_spike).unwrap();
        }
        
        // Update network
        let spikes = network.update(dt).unwrap();
        spike_count += spikes.len();
        
        // Verify network state remains valid
        assert_eq!(network.neuron_count(), 5);
        assert_eq!(network.hyperedge_count(), 4);
    }
    
    // Should have generated some spikes
    assert!(spike_count > 0);
}

#[wasm_bindgen_test]
fn test_wasm_network_renderer() {
    let config = JSVisualizationConfig {
        canvas_width: 800,
        canvas_height: 600,
        neuron_radius: 5.0,
        edge_width: 2.0,
        background_color: "#ffffff".to_string(),
        neuron_color: "#0066cc".to_string(),
        edge_color: "#cccccc".to_string(),
        spike_color: "#ff0000".to_string(),
    };
    
    let renderer = WasmNetworkRenderer::new(config);
    assert!(renderer.is_ok());
    
    let mut renderer = renderer.unwrap();
    
    // Create test network
    let mut network = WasmHypergraphNetwork::new(JSNetworkConfig::default()).unwrap();
    let neuron1 = network.add_neuron(JSNeuronConfig::default()).unwrap();
    let neuron2 = network.add_neuron(JSNeuronConfig::default()).unwrap();
    network.add_hyperedge(vec![neuron1], vec![neuron2], 1.0).unwrap();
    
    // Render network
    let result = renderer.render_network(&network);
    assert!(result.is_ok());
    
    // Verify rendering properties
    assert_eq!(renderer.canvas_width(), 800);
    assert_eq!(renderer.canvas_height(), 600);
}

#[wasm_bindgen_test]
fn test_wasm_spike_renderer() {
    let config = JSVisualizationConfig::default();
    let mut renderer = WasmSpikeRenderer::new(config).unwrap();
    
    // Create test spikes
    let spikes = vec![
        WasmSpike::new(WasmNeuronId::new(0), WasmTimeStep::from_ms(1.0)),
        WasmSpike::new(WasmNeuronId::new(1), WasmTimeStep::from_ms(2.0)),
        WasmSpike::new(WasmNeuronId::new(0), WasmTimeStep::from_ms(3.0)),
    ];
    
    // Render spike raster
    let result = renderer.render_spike_raster(&spikes, WasmTimeStep::from_ms(5.0));
    assert!(result.is_ok());
    
    // Render spike histogram
    let result = renderer.render_spike_histogram(&spikes, 10);
    assert!(result.is_ok());
}

#[wasm_bindgen_test]
fn test_wasm_memory_management() {
    // Test that WASM memory is properly managed
    let initial_memory = wasm_bindgen::memory().buffer().byte_length();
    
    // Create and destroy many networks
    for _ in 0..100 {
        let mut network = WasmHypergraphNetwork::new(JSNetworkConfig::default()).unwrap();
        
        // Add neurons and connections
        for i in 0..50 {
            let config = JSNeuronConfig::default();
            network.add_neuron(config).unwrap();
        }
        
        // Let network go out of scope to trigger cleanup
        drop(network);
    }
    
    // Memory should not have grown excessively
    let final_memory = wasm_bindgen::memory().buffer().byte_length();
    let memory_growth = final_memory - initial_memory;
    
    // Allow some growth but not excessive
    assert!(memory_growth < 10 * 1024 * 1024); // Less than 10MB growth
}

#[wasm_bindgen_test]
fn test_wasm_error_handling() {
    let mut network = WasmHypergraphNetwork::new(JSNetworkConfig::default()).unwrap();
    
    // Test invalid neuron access
    let invalid_neuron = WasmNeuronId::new(999);
    let result = network.get_neuron_state(invalid_neuron);
    assert!(result.is_err());
    
    // Test invalid spike processing
    let invalid_spike = WasmSpike::new(invalid_neuron, WasmTimeStep::from_ms(1.0));
    let result = network.process_spike(invalid_spike);
    assert!(result.is_err());
    
    // Network should remain functional after errors
    let config = JSNeuronConfig::default();
    let neuron = network.add_neuron(config);
    assert!(neuron.is_ok());
}

#[wasm_bindgen_test]
fn test_wasm_javascript_interop() {
    // Test JavaScript value conversions
    let js_config = js_sys::Object::new();
    js_sys::Reflect::set(&js_config, &"neuron_type".into(), &"LIF".into()).unwrap();
    js_sys::Reflect::set(&js_config, &"threshold".into(), &(-55.0).into()).unwrap();
    
    let wasm_config = JSNeuronConfig::from_js_value(&js_config.into());
    assert!(wasm_config.is_ok());
    
    let config = wasm_config.unwrap();
    assert_eq!(config.neuron_type, "LIF");
    assert_eq!(config.threshold, -55.0);
}

#[wasm_bindgen_test]
fn test_wasm_performance_monitoring() {
    let mut network = WasmHypergraphNetwork::new(JSNetworkConfig::default()).unwrap();
    
    // Enable performance monitoring
    network.enable_performance_monitoring().unwrap();
    
    // Add neurons and process spikes
    let neuron = network.add_neuron(JSNeuronConfig::default()).unwrap();
    
    for i in 0..100 {
        let spike = WasmSpike::new(neuron, WasmTimeStep::from_ms(i as f64));
        network.process_spike(spike).unwrap();
    }
    
    // Get performance metrics
    let metrics = network.get_performance_metrics().unwrap();
    
    assert_eq!(metrics.total_spikes_processed(), 100);
    assert!(metrics.average_processing_time() > 0.0);
    assert!(metrics.peak_memory_usage() > 0);
}

#[wasm_bindgen_test]
fn test_wasm_real_time_visualization() {
    let vis_config = JSVisualizationConfig::default();
    let mut renderer = WasmNetworkRenderer::new(vis_config).unwrap();
    
    let mut network = WasmHypergraphNetwork::new(JSNetworkConfig::default()).unwrap();
    
    // Create network with multiple neurons
    let mut neurons = vec![];
    for _ in 0..10 {
        let neuron = network.add_neuron(JSNeuronConfig::default()).unwrap();
        neurons.push(neuron);
    }
    
    // Create connections
    for i in 0..9 {
        network.add_hyperedge(vec![neurons[i]], vec![neurons[i + 1]], 1.0).unwrap();
    }
    
    // Simulate and render in real-time
    let dt = WasmTimeStep::from_ms(1.0);
    let mut frame_count = 0;
    
    for frame in 0..60 { // 60 frames
        // Inject stimulus
        if frame % 10 == 0 {
            let spike = WasmSpike::new(neurons[0], WasmTimeStep::from_ms(frame as f64));
            network.process_spike(spike).unwrap();
        }
        
        // Update network
        network.update(dt).unwrap();
        
        // Render frame
        let render_result = renderer.render_network(&network);
        if render_result.is_ok() {
            frame_count += 1;
        }
    }
    
    // Should have successfully rendered most frames
    assert!(frame_count >= 50);
}

#[wasm_bindgen_test]
fn test_wasm_logger_functionality() {
    // Test WASM logging capabilities
    let logger = WasmLogger::new();
    
    logger.log("Test info message");
    logger.warn("Test warning message");
    logger.error("Test error message");
    
    // Logger should track message counts
    assert!(logger.message_count() >= 3);
    
    // Test log level filtering
    logger.set_log_level("error");
    logger.info("This should be filtered");
    logger.error("This should appear");
    
    let filtered_count = logger.message_count();
    assert!(filtered_count >= 4); // Previous 3 + 1 error
}

#[wasm_bindgen_test]
fn test_wasm_browser_compatibility() {
    // Test features that depend on browser APIs
    let window = web_sys::window().expect("should have window");
    let document = window.document().expect("should have document");
    
    // Create canvas element for rendering
    let canvas = document
        .create_element("canvas")
        .expect("should create canvas")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("should be canvas");
    
    canvas.set_width(800);
    canvas.set_height(600);
    
    // Test WebGL context creation
    let gl_context = canvas
        .get_context("webgl")
        .expect("should get context")
        .and_then(|ctx| ctx.dyn_into::<web_sys::WebGlRenderingContext>().ok());
    
    // WebGL should be available in test environment
    assert!(gl_context.is_some());
    
    // Test animation frame scheduling
    let closure = wasm_bindgen::closure::Closure::wrap(Box::new(move |_time: f64| {
        // Animation frame callback
    }) as Box<dyn FnMut(f64)>);
    
    let result = window.request_animation_frame(closure.as_ref().unchecked_ref());
    assert!(result.is_ok());
    
    // Clean up
    closure.forget();
}

#[wasm_bindgen_test]
fn test_wasm_multithreading_simulation() {
    // Test shared memory and web workers simulation
    let config = JSNetworkConfig {
        enable_multithreading: true,
        worker_count: 2,
        ..Default::default()
    };
    
    let mut network = WasmHypergraphNetwork::new(config).unwrap();
    
    // Create larger network for parallel processing
    let mut neurons = vec![];
    for _ in 0..100 {
        let neuron = network.add_neuron(JSNeuronConfig::default()).unwrap();
        neurons.push(neuron);
    }
    
    // Create many connections
    for i in 0..99 {
        network.add_hyperedge(vec![neurons[i]], vec![neurons[i + 1]], 0.5).unwrap();
    }
    
    // Process many spikes concurrently
    let start_time = js_sys::Date::now();
    
    for i in 0..1000 {
        let neuron = neurons[i % neurons.len()];
        let spike = WasmSpike::new(neuron, WasmTimeStep::from_ms(i as f64 * 0.1));
        network.process_spike(spike).unwrap();
    }
    
    let end_time = js_sys::Date::now();
    let processing_time = end_time - start_time;
    
    // Should complete in reasonable time (less than 1 second)
    assert!(processing_time < 1000.0);
    
    // Verify network state
    assert_eq!(network.neuron_count(), 100);
    assert_eq!(network.hyperedge_count(), 99);
}