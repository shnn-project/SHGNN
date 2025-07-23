//! Comprehensive tests for SHNN serialization refactoring
//!
//! This test suite validates all the zero-dependency serialization components
//! that replaced serde functionality.

use shnn_serialize::{
    Serialize, Deserialize, BinaryEncoder, BinaryDecoder,
    Buffer, BufferMut, ZeroCopyBuffer,
    endian::{LittleEndian, BigEndian, NetworkEndian},
    utils::{calculate_size, align_to, padding_needed},
    neural::{SpikeEvent, WeightMatrix, LayerState, NeuralSerializer},
};
use std::{collections::HashMap, mem};

const EPSILON: f32 = 1e-6;

/// Test basic serialization and deserialization
#[test]
fn test_basic_serialize_deserialize() {
    let original_u32 = 0x12345678u32;
    let original_f32 = 3.14159f32;
    let original_bool = true;
    
    // Test u32
    let mut buffer = Vec::new();
    let mut encoder = BinaryEncoder::new(&mut buffer);
    original_u32.serialize(&mut encoder).unwrap();
    
    let mut decoder = BinaryDecoder::new(&buffer);
    let decoded_u32 = u32::deserialize(&mut decoder).unwrap();
    assert_eq!(original_u32, decoded_u32);
    
    // Test f32
    buffer.clear();
    encoder = BinaryEncoder::new(&mut buffer);
    original_f32.serialize(&mut encoder).unwrap();
    
    decoder = BinaryDecoder::new(&buffer);
    let decoded_f32 = f32::deserialize(&mut decoder).unwrap();
    assert!((original_f32 - decoded_f32).abs() < EPSILON);
    
    // Test bool
    buffer.clear();
    encoder = BinaryEncoder::new(&mut buffer);
    original_bool.serialize(&mut encoder).unwrap();
    
    decoder = BinaryDecoder::new(&buffer);
    let decoded_bool = bool::deserialize(&mut decoder).unwrap();
    assert_eq!(original_bool, decoded_bool);
}

/// Test vector serialization
#[test]
fn test_vector_serialization() {
    let original_vec = vec![1u32, 2, 3, 4, 5];
    
    let mut buffer = Vec::new();
    let mut encoder = BinaryEncoder::new(&mut buffer);
    original_vec.serialize(&mut encoder).unwrap();
    
    let mut decoder = BinaryDecoder::new(&buffer);
    let decoded_vec = Vec::<u32>::deserialize(&mut decoder).unwrap();
    
    assert_eq!(original_vec, decoded_vec);
}

/// Test string serialization
#[test]
fn test_string_serialization() {
    let original_string = "Hello, SHNN World! 🧠".to_string();
    
    let mut buffer = Vec::new();
    let mut encoder = BinaryEncoder::new(&mut buffer);
    original_string.serialize(&mut encoder).unwrap();
    
    let mut decoder = BinaryDecoder::new(&buffer);
    let decoded_string = String::deserialize(&mut decoder).unwrap();
    
    assert_eq!(original_string, decoded_string);
}

/// Test struct serialization with derive
#[derive(Debug, PartialEq)]
struct TestStruct {
    id: u32,
    name: String,
    values: Vec<f32>,
    active: bool,
}

impl Serialize for TestStruct {
    fn serialize<W: BufferMut>(&self, encoder: &mut BinaryEncoder<W>) -> Result<(), Box<dyn std::error::Error>> {
        self.id.serialize(encoder)?;
        self.name.serialize(encoder)?;
        self.values.serialize(encoder)?;
        self.active.serialize(encoder)?;
        Ok(())
    }
}

impl Deserialize for TestStruct {
    fn deserialize<R: Buffer>(decoder: &mut BinaryDecoder<R>) -> Result<Self, Box<dyn std::error::Error>> {
        let id = u32::deserialize(decoder)?;
        let name = String::deserialize(decoder)?;
        let values = Vec::<f32>::deserialize(decoder)?;
        let active = bool::deserialize(decoder)?;
        
        Ok(TestStruct { id, name, values, active })
    }
}

#[test]
fn test_struct_serialization() {
    let original_struct = TestStruct {
        id: 42,
        name: "Test Neuron".to_string(),
        values: vec![1.0, 2.0, 3.0, 4.0],
        active: true,
    };
    
    let mut buffer = Vec::new();
    let mut encoder = BinaryEncoder::new(&mut buffer);
    original_struct.serialize(&mut encoder).unwrap();
    
    let mut decoder = BinaryDecoder::new(&buffer);
    let decoded_struct = TestStruct::deserialize(&mut decoder).unwrap();
    
    assert_eq!(original_struct, decoded_struct);
}

/// Test endianness handling
#[test]
fn test_endianness() {
    let value = 0x12345678u32;
    
    // Test little endian
    let le_bytes = LittleEndian::to_bytes(value);
    assert_eq!(le_bytes, [0x78, 0x56, 0x34, 0x12]);
    assert_eq!(LittleEndian::from_bytes(le_bytes), value);
    
    // Test big endian
    let be_bytes = BigEndian::to_bytes(value);
    assert_eq!(be_bytes, [0x12, 0x34, 0x56, 0x78]);
    assert_eq!(BigEndian::from_bytes(be_bytes), value);
    
    // Test network endian (should be same as big endian)
    let ne_bytes = NetworkEndian::to_bytes(value);
    assert_eq!(ne_bytes, be_bytes);
    assert_eq!(NetworkEndian::from_bytes(ne_bytes), value);
}

/// Test zero-copy buffer operations
#[test]
fn test_zero_copy_buffer() {
    let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let buffer = ZeroCopyBuffer::new(&data);
    
    assert_eq!(buffer.len(), 8);
    assert_eq!(buffer.remaining(), 8);
    assert!(!buffer.is_empty());
    
    // Test reading
    let bytes = buffer.read_bytes(4).unwrap();
    assert_eq!(bytes, &[1, 2, 3, 4]);
    assert_eq!(buffer.remaining(), 4);
    
    let more_bytes = buffer.read_bytes(4).unwrap();
    assert_eq!(more_bytes, &[5, 6, 7, 8]);
    assert_eq!(buffer.remaining(), 0);
    assert!(buffer.is_empty());
    
    // Test reading beyond end
    assert!(buffer.read_bytes(1).is_err());
}

/// Test spike event serialization
#[test]
fn test_spike_event_serialization() {
    let spike = SpikeEvent {
        neuron_id: 42,
        timestamp: 123.456,
        amplitude: 0.8,
    };
    
    let mut buffer = Vec::new();
    let mut encoder = BinaryEncoder::new(&mut buffer);
    spike.serialize(&mut encoder).unwrap();
    
    let mut decoder = BinaryDecoder::new(&buffer);
    let decoded_spike = SpikeEvent::deserialize(&mut decoder).unwrap();
    
    assert_eq!(spike.neuron_id, decoded_spike.neuron_id);
    assert!((spike.timestamp - decoded_spike.timestamp).abs() < EPSILON);
    assert!((spike.amplitude - decoded_spike.amplitude).abs() < EPSILON);
}

/// Test weight matrix serialization
#[test]
fn test_weight_matrix_serialization() {
    let weights = WeightMatrix {
        rows: 3,
        cols: 4,
        data: vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ],
    };
    
    let mut buffer = Vec::new();
    let mut encoder = BinaryEncoder::new(&mut buffer);
    weights.serialize(&mut encoder).unwrap();
    
    let mut decoder = BinaryDecoder::new(&buffer);
    let decoded_weights = WeightMatrix::deserialize(&mut decoder).unwrap();
    
    assert_eq!(weights.rows, decoded_weights.rows);
    assert_eq!(weights.cols, decoded_weights.cols);
    assert_eq!(weights.data.len(), decoded_weights.data.len());
    
    for (orig, decoded) in weights.data.iter().zip(decoded_weights.data.iter()) {
        assert!((orig - decoded).abs() < EPSILON);
    }
}

/// Test layer state serialization
#[test]
fn test_layer_state_serialization() {
    let mut neurons = HashMap::new();
    neurons.insert(0, vec![1.0, 2.0, 3.0]);
    neurons.insert(1, vec![4.0, 5.0, 6.0]);
    neurons.insert(2, vec![7.0, 8.0, 9.0]);
    
    let layer_state = LayerState {
        layer_id: 5,
        neurons,
        timestamp: 987.654,
    };
    
    let mut buffer = Vec::new();
    let mut encoder = BinaryEncoder::new(&mut buffer);
    layer_state.serialize(&mut encoder).unwrap();
    
    let mut decoder = BinaryDecoder::new(&buffer);
    let decoded_state = LayerState::deserialize(&mut decoder).unwrap();
    
    assert_eq!(layer_state.layer_id, decoded_state.layer_id);
    assert!((layer_state.timestamp - decoded_state.timestamp).abs() < EPSILON);
    assert_eq!(layer_state.neurons.len(), decoded_state.neurons.len());
    
    for (key, orig_values) in &layer_state.neurons {
        let decoded_values = decoded_state.neurons.get(key).unwrap();
        assert_eq!(orig_values.len(), decoded_values.len());
        
        for (orig, decoded) in orig_values.iter().zip(decoded_values.iter()) {
            assert!((orig - decoded).abs() < EPSILON);
        }
    }
}

/// Test neural serializer
#[test]
fn test_neural_serializer() {
    let serializer = NeuralSerializer::new();
    
    // Create test data
    let spike_events = vec![
        SpikeEvent { neuron_id: 1, timestamp: 10.0, amplitude: 0.5 },
        SpikeEvent { neuron_id: 2, timestamp: 15.0, amplitude: 0.8 },
        SpikeEvent { neuron_id: 3, timestamp: 20.0, amplitude: 0.3 },
    ];
    
    let weights = WeightMatrix {
        rows: 2,
        cols: 3,
        data: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    };
    
    let mut neurons = HashMap::new();
    neurons.insert(0, vec![1.0, 2.0]);
    neurons.insert(1, vec![3.0, 4.0]);
    
    let layer_state = LayerState {
        layer_id: 1,
        neurons,
        timestamp: 100.0,
    };
    
    // Serialize network state
    let serialized = serializer.serialize_network_state(&spike_events, &weights, &layer_state).unwrap();
    
    // Deserialize network state
    let (decoded_spikes, decoded_weights, decoded_state) = serializer.deserialize_network_state(&serialized).unwrap();
    
    // Verify spikes
    assert_eq!(spike_events.len(), decoded_spikes.len());
    for (orig, decoded) in spike_events.iter().zip(decoded_spikes.iter()) {
        assert_eq!(orig.neuron_id, decoded.neuron_id);
        assert!((orig.timestamp - decoded.timestamp).abs() < EPSILON);
        assert!((orig.amplitude - decoded.amplitude).abs() < EPSILON);
    }
    
    // Verify weights
    assert_eq!(weights.rows, decoded_weights.rows);
    assert_eq!(weights.cols, decoded_weights.cols);
    for (orig, decoded) in weights.data.iter().zip(decoded_weights.data.iter()) {
        assert!((orig - decoded).abs() < EPSILON);
    }
    
    // Verify layer state
    assert_eq!(layer_state.layer_id, decoded_state.layer_id);
    assert!((layer_state.timestamp - decoded_state.timestamp).abs() < EPSILON);
    assert_eq!(layer_state.neurons.len(), decoded_state.neurons.len());
}

/// Test utility functions
#[test]
fn test_utility_functions() {
    // Test calculate_size
    assert_eq!(calculate_size::<u32>(), 4);
    assert_eq!(calculate_size::<f64>(), 8);
    assert_eq!(calculate_size::<bool>(), 1);
    
    // Test align_to
    assert_eq!(align_to(5, 4), 8);
    assert_eq!(align_to(8, 4), 8);
    assert_eq!(align_to(9, 4), 12);
    assert_eq!(align_to(0, 4), 0);
    
    // Test padding_needed
    assert_eq!(padding_needed(5, 4), 3);
    assert_eq!(padding_needed(8, 4), 0);
    assert_eq!(padding_needed(9, 4), 3);
    assert_eq!(padding_needed(0, 4), 0);
}

/// Test error handling
#[test]
fn test_error_handling() {
    // Test buffer underflow
    let small_buffer = vec![1u8, 2];
    let mut decoder = BinaryDecoder::new(&small_buffer);
    
    // Try to read more data than available
    let result = u64::deserialize(&mut decoder);
    assert!(result.is_err());
    
    // Test invalid UTF-8 in string
    let invalid_utf8 = vec![
        4u8, 0, 0, 0,  // length = 4
        0xFF, 0xFF, 0xFF, 0xFF,  // invalid UTF-8
    ];
    
    let mut decoder = BinaryDecoder::new(&invalid_utf8);
    let result = String::deserialize(&mut decoder);
    assert!(result.is_err());
}

/// Test deterministic serialization
#[test]
fn test_deterministic_serialization() {
    let test_data = TestStruct {
        id: 123,
        name: "Deterministic Test".to_string(),
        values: vec![1.1, 2.2, 3.3],
        active: false,
    };
    
    // Serialize multiple times
    let mut buffer1 = Vec::new();
    let mut encoder1 = BinaryEncoder::new(&mut buffer1);
    test_data.serialize(&mut encoder1).unwrap();
    
    let mut buffer2 = Vec::new();
    let mut encoder2 = BinaryEncoder::new(&mut buffer2);
    test_data.serialize(&mut encoder2).unwrap();
    
    // Results should be identical
    assert_eq!(buffer1, buffer2);
}

/// Test large data serialization
#[test]
fn test_large_data_serialization() {
    let large_vec: Vec<u32> = (0..100_000).collect();
    
    let start = std::time::Instant::now();
    
    let mut buffer = Vec::new();
    let mut encoder = BinaryEncoder::new(&mut buffer);
    large_vec.serialize(&mut encoder).unwrap();
    
    let serialize_time = start.elapsed();
    
    let start = std::time::Instant::now();
    
    let mut decoder = BinaryDecoder::new(&buffer);
    let decoded_vec = Vec::<u32>::deserialize(&mut decoder).unwrap();
    
    let deserialize_time = start.elapsed();
    
    assert_eq!(large_vec, decoded_vec);
    
    println!("Serialized {} items in {:?}", large_vec.len(), serialize_time);
    println!("Deserialized {} items in {:?}", decoded_vec.len(), deserialize_time);
    
    // Should be reasonably fast
    assert!(serialize_time < std::time::Duration::from_millis(100));
    assert!(deserialize_time < std::time::Duration::from_millis(100));
}

/// Test nested data structures
#[test]
fn test_nested_structures() {
    let nested_data = vec![
        vec![1u32, 2, 3],
        vec![4, 5],
        vec![6, 7, 8, 9],
        vec![],
        vec![10],
    ];
    
    let mut buffer = Vec::new();
    let mut encoder = BinaryEncoder::new(&mut buffer);
    nested_data.serialize(&mut encoder).unwrap();
    
    let mut decoder = BinaryDecoder::new(&buffer);
    let decoded_nested = Vec::<Vec<u32>>::deserialize(&mut decoder).unwrap();
    
    assert_eq!(nested_data, decoded_nested);
}

/// Test concurrent serialization safety
#[test]
fn test_concurrent_serialization() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let test_data = Arc::new(vec![
        TestStruct {
            id: 1,
            name: "Thread 1".to_string(),
            values: vec![1.0, 2.0],
            active: true,
        },
        TestStruct {
            id: 2,
            name: "Thread 2".to_string(),
            values: vec![3.0, 4.0],
            active: false,
        },
    ]);
    
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();
    
    for i in 0..4 {
        let data_clone = test_data.clone();
        let results_clone = results.clone();
        
        let handle = thread::spawn(move || {
            let mut buffer = Vec::new();
            let mut encoder = BinaryEncoder::new(&mut buffer);
            data_clone[i % 2].serialize(&mut encoder).unwrap();
            
            results_clone.lock().unwrap().push(buffer);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let results = results.lock().unwrap();
    assert_eq!(results.len(), 4);
    
    // Verify all serializations are valid
    for buffer in results.iter() {
        let mut decoder = BinaryDecoder::new(buffer);
        let _decoded = TestStruct::deserialize(&mut decoder).unwrap();
    }
}

/// Test memory alignment and padding
#[test]
fn test_memory_alignment() {
    #[repr(C)]
    struct AlignedStruct {
        a: u8,    // 1 byte
        b: u32,   // 4 bytes (requires 4-byte alignment)
        c: u8,    // 1 byte
        d: u64,   // 8 bytes (requires 8-byte alignment)
    }
    
    impl Serialize for AlignedStruct {
        fn serialize<W: BufferMut>(&self, encoder: &mut BinaryEncoder<W>) -> Result<(), Box<dyn std::error::Error>> {
            self.a.serialize(encoder)?;
            // Add padding for b's alignment
            encoder.write_padding(padding_needed(1, 4))?;
            self.b.serialize(encoder)?;
            self.c.serialize(encoder)?;
            // Add padding for d's alignment
            encoder.write_padding(padding_needed(1 + 4 + 1, 8))?;
            self.d.serialize(encoder)?;
            Ok(())
        }
    }
    
    impl Deserialize for AlignedStruct {
        fn deserialize<R: Buffer>(decoder: &mut BinaryDecoder<R>) -> Result<Self, Box<dyn std::error::Error>> {
            let a = u8::deserialize(decoder)?;
            decoder.skip_padding(padding_needed(1, 4))?;
            let b = u32::deserialize(decoder)?;
            let c = u8::deserialize(decoder)?;
            decoder.skip_padding(padding_needed(1 + 4 + 1, 8))?;
            let d = u64::deserialize(decoder)?;
            
            Ok(AlignedStruct { a, b, c, d })
        }
    }
    
    let aligned = AlignedStruct {
        a: 0x12,
        b: 0x34567890,
        c: 0xAB,
        d: 0xCDEF123456789ABC,
    };
    
    let mut buffer = Vec::new();
    let mut encoder = BinaryEncoder::new(&mut buffer);
    aligned.serialize(&mut encoder).unwrap();
    
    let mut decoder = BinaryDecoder::new(&buffer);
    let decoded = AlignedStruct::deserialize(&mut decoder).unwrap();
    
    assert_eq!(aligned.a, decoded.a);
    assert_eq!(aligned.b, decoded.b);
    assert_eq!(aligned.c, decoded.c);
    assert_eq!(aligned.d, decoded.d);
}

/// Test version compatibility
#[test]
fn test_version_compatibility() {
    // Simulate older version data
    let v1_data = vec![
        1u8, 0, 0, 0,  // version = 1
        42u8, 0, 0, 0,  // id = 42
        5u8, 0, 0, 0,   // name length = 5
        b'H', b'e', b'l', b'l', b'o',  // name = "Hello"
    ];
    
    // Try to parse with version handling
    let mut decoder = BinaryDecoder::new(&v1_data);
    let version = u32::deserialize(&mut decoder).unwrap();
    
    assert_eq!(version, 1);
    
    if version == 1 {
        let id = u32::deserialize(&mut decoder).unwrap();
        let name = String::deserialize(&mut decoder).unwrap();
        
        assert_eq!(id, 42);
        assert_eq!(name, "Hello");
    }
}