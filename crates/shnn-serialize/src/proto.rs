//! Protocol buffer implementation
//! 
//! Provides a lightweight protocol buffer implementation for structured
//! serialization with schema evolution support and efficient encoding.

use crate::{Buffer, BufferMut, Result, SerializeError};
#[cfg(feature = "std")]
use std::{vec::Vec, string::{String, ToString}};

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::{String, ToString}};

/// Wire types for protocol buffer encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireType {
    /// Variable-length integers
    Varint = 0,
    /// 64-bit fixed-length values
    Fixed64 = 1,
    /// Length-delimited values (strings, bytes, embedded messages)
    LengthDelimited = 2,
    /// 32-bit fixed-length values
    Fixed32 = 5,
}

impl WireType {
    /// Convert wire type from u8
    pub fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(WireType::Varint),
            1 => Ok(WireType::Fixed64),
            2 => Ok(WireType::LengthDelimited),
            5 => Ok(WireType::Fixed32),
            _ => Err(SerializeError::InvalidWireType),
        }
    }

    /// Convert wire type to u8
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

/// Protocol buffer field tag
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldTag {
    /// Field number (1-536870911)
    pub field_number: u32,
    /// Wire type
    pub wire_type: WireType,
}

impl FieldTag {
    /// Create a new field tag
    pub fn new(field_number: u32, wire_type: WireType) -> Result<Self> {
        if field_number == 0 || field_number > 536870911 {
            return Err(SerializeError::InvalidFieldNumber);
        }
        Ok(Self { field_number, wire_type })
    }

    /// Encode field tag as varint
    pub fn encode(&self) -> u32 {
        (self.field_number << 3) | (self.wire_type.to_u8() as u32)
    }

    /// Decode field tag from varint
    pub fn decode(tag: u32) -> Result<Self> {
        let wire_type = WireType::from_u8((tag & 0x7) as u8)?;
        let field_number = tag >> 3;
        Self::new(field_number, wire_type)
    }
}

/// Protocol buffer encoder
#[derive(Debug)]
pub struct ProtoEncoder<'a> {
    buffer: BufferMut<'a>,
}

impl<'a> ProtoEncoder<'a> {
    /// Create new protocol buffer encoder
    pub fn new(buffer: BufferMut<'a>) -> Self {
        Self { buffer }
    }

    /// Encode varint (variable-length integer)
    pub fn encode_varint(&mut self, mut value: u64) -> Result<()> {
        while value >= 0x80 {
            self.buffer.write_u8((value & 0x7F | 0x80) as u8)?;
            value >>= 7;
        }
        self.buffer.write_u8(value as u8)
    }

    /// Encode signed varint using zigzag encoding
    pub fn encode_sint_varint(&mut self, value: i64) -> Result<()> {
        let zigzag = if value >= 0 {
            (value as u64) << 1
        } else {
            ((-value - 1) as u64) << 1 | 1
        };
        self.encode_varint(zigzag)
    }

    /// Encode field tag
    pub fn encode_tag(&mut self, tag: FieldTag) -> Result<()> {
        self.encode_varint(tag.encode() as u64)
    }

    /// Encode 32-bit fixed value
    pub fn encode_fixed32(&mut self, value: u32) -> Result<()> {
        self.buffer.write_u32(value)
    }

    /// Encode 64-bit fixed value
    pub fn encode_fixed64(&mut self, value: u64) -> Result<()> {
        let bytes = value.to_le_bytes();
        self.buffer.write_bytes(&bytes)
    }

    /// Encode length-delimited value
    pub fn encode_length_delimited(&mut self, data: &[u8]) -> Result<()> {
        self.encode_varint(data.len() as u64)?;
        self.buffer.write_bytes(data)
    }

    /// Encode string field
    pub fn encode_string_field(&mut self, field_number: u32, value: &str) -> Result<()> {
        let tag = FieldTag::new(field_number, WireType::LengthDelimited)?;
        self.encode_tag(tag)?;
        self.encode_length_delimited(value.as_bytes())
    }

    /// Encode bytes field
    pub fn encode_bytes_field(&mut self, field_number: u32, value: &[u8]) -> Result<()> {
        let tag = FieldTag::new(field_number, WireType::LengthDelimited)?;
        self.encode_tag(tag)?;
        self.encode_length_delimited(value)
    }

    /// Encode u32 field
    pub fn encode_u32_field(&mut self, field_number: u32, value: u32) -> Result<()> {
        let tag = FieldTag::new(field_number, WireType::Varint)?;
        self.encode_tag(tag)?;
        self.encode_varint(value as u64)
    }

    /// Encode u64 field
    pub fn encode_u64_field(&mut self, field_number: u32, value: u64) -> Result<()> {
        let tag = FieldTag::new(field_number, WireType::Varint)?;
        self.encode_tag(tag)?;
        self.encode_varint(value)
    }

    /// Encode i32 field
    pub fn encode_i32_field(&mut self, field_number: u32, value: i32) -> Result<()> {
        let tag = FieldTag::new(field_number, WireType::Varint)?;
        self.encode_tag(tag)?;
        self.encode_sint_varint(value as i64)
    }

    /// Encode i64 field
    pub fn encode_i64_field(&mut self, field_number: u32, value: i64) -> Result<()> {
        let tag = FieldTag::new(field_number, WireType::Varint)?;
        self.encode_tag(tag)?;
        self.encode_sint_varint(value)
    }

    /// Encode f32 field
    pub fn encode_f32_field(&mut self, field_number: u32, value: f32) -> Result<()> {
        let tag = FieldTag::new(field_number, WireType::Fixed32)?;
        self.encode_tag(tag)?;
        self.encode_fixed32(value.to_bits())
    }

    /// Encode f64 field
    pub fn encode_f64_field(&mut self, field_number: u32, value: f64) -> Result<()> {
        let tag = FieldTag::new(field_number, WireType::Fixed64)?;
        self.encode_tag(tag)?;
        self.encode_fixed64(value.to_bits())
    }

    /// Encode bool field
    pub fn encode_bool_field(&mut self, field_number: u32, value: bool) -> Result<()> {
        let tag = FieldTag::new(field_number, WireType::Varint)?;
        self.encode_tag(tag)?;
        self.encode_varint(if value { 1 } else { 0 })
    }

    /// Get current position
    pub fn position(&self) -> usize {
        self.buffer.position()
    }

    /// Get written data
    pub fn written(&self) -> &[u8] {
        self.buffer.written()
    }
}

/// Protocol buffer decoder
#[derive(Debug)]
pub struct ProtoDecoder<'a> {
    buffer: Buffer<'a>,
}

impl<'a> ProtoDecoder<'a> {
    /// Create new protocol buffer decoder
    pub fn new(buffer: Buffer<'a>) -> Self {
        Self { buffer }
    }

    /// Decode varint
    pub fn decode_varint(&mut self) -> Result<u64> {
        let mut result = 0u64;
        let mut shift = 0;

        loop {
            if shift >= 64 {
                return Err(SerializeError::VarintTooLong);
            }

            let byte = self.buffer.read_u8()?;
            result |= ((byte & 0x7F) as u64) << shift;

            if byte & 0x80 == 0 {
                break;
            }

            shift += 7;
        }

        Ok(result)
    }

    /// Decode signed varint using zigzag decoding
    pub fn decode_sint_varint(&mut self) -> Result<i64> {
        let zigzag = self.decode_varint()?;
        let result = if zigzag & 1 == 0 {
            (zigzag >> 1) as i64
        } else {
            -((zigzag >> 1) as i64) - 1
        };
        Ok(result)
    }

    /// Decode field tag
    pub fn decode_tag(&mut self) -> Result<FieldTag> {
        let tag = self.decode_varint()? as u32;
        FieldTag::decode(tag)
    }

    /// Decode 32-bit fixed value
    pub fn decode_fixed32(&mut self) -> Result<u32> {
        self.buffer.read_u32()
    }

    /// Decode 64-bit fixed value
    pub fn decode_fixed64(&mut self) -> Result<u64> {
        let bytes = self.buffer.read_bytes(8)?;
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(array))
    }

    /// Decode length-delimited value
    pub fn decode_length_delimited(&mut self) -> Result<&'a [u8]> {
        let length = self.decode_varint()? as usize;
        self.buffer.read_bytes(length)
    }

    /// Decode string
    pub fn decode_string(&mut self) -> Result<&'a str> {
        let bytes = self.decode_length_delimited()?;
        core::str::from_utf8(bytes).map_err(|_| SerializeError::InvalidUtf8)
    }

    /// Skip field based on wire type
    pub fn skip_field(&mut self, wire_type: WireType) -> Result<()> {
        match wire_type {
            WireType::Varint => {
                self.decode_varint()?;
            }
            WireType::Fixed32 => {
                self.buffer.read_bytes(4)?;
            }
            WireType::Fixed64 => {
                self.buffer.read_bytes(8)?;
            }
            WireType::LengthDelimited => {
                self.decode_length_delimited()?;
            }
        }
        Ok(())
    }

    /// Get remaining bytes
    pub fn remaining(&self) -> usize {
        self.buffer.remaining()
    }

    /// Check if at end
    pub fn is_at_end(&self) -> bool {
        self.buffer.remaining() == 0
    }
}

/// Protocol buffer message trait
pub trait ProtoMessage: Sized {
    /// Encode message to buffer
    fn encode_to(&self, encoder: &mut ProtoEncoder) -> Result<()>;
    
    /// Decode message from buffer
    fn decode_from(decoder: &mut ProtoDecoder) -> Result<Self>;
    
    /// Get encoded size
    fn encoded_size(&self) -> usize;
}

/// Simple message implementation for neural data
#[derive(Debug, Clone, PartialEq)]
pub struct NeuralMessage {
    pub neuron_id: u32,
    pub timestamp: f64,
    pub spike_data: Vec<u8>,
    pub layer_name: String,
}

impl ProtoMessage for NeuralMessage {
    fn encode_to(&self, encoder: &mut ProtoEncoder) -> Result<()> {
        encoder.encode_u32_field(1, self.neuron_id)?;
        encoder.encode_f64_field(2, self.timestamp)?;
        encoder.encode_bytes_field(3, &self.spike_data)?;
        encoder.encode_string_field(4, &self.layer_name)?;
        Ok(())
    }

    fn decode_from(decoder: &mut ProtoDecoder) -> Result<Self> {
        let mut neuron_id = 0;
        let mut timestamp = 0.0;
        let mut spike_data = Vec::new();
        let mut layer_name = String::new();

        while !decoder.is_at_end() {
            let tag = decoder.decode_tag()?;
            
            match tag.field_number {
                1 => {
                    if tag.wire_type != WireType::Varint {
                        return Err(SerializeError::WireTypeMismatch);
                    }
                    neuron_id = decoder.decode_varint()? as u32;
                }
                2 => {
                    if tag.wire_type != WireType::Fixed64 {
                        return Err(SerializeError::WireTypeMismatch);
                    }
                    let bits = decoder.decode_fixed64()?;
                    timestamp = f64::from_bits(bits);
                }
                3 => {
                    if tag.wire_type != WireType::LengthDelimited {
                        return Err(SerializeError::WireTypeMismatch);
                    }
                    let bytes = decoder.decode_length_delimited()?;
                    spike_data = bytes.to_vec();
                }
                4 => {
                    if tag.wire_type != WireType::LengthDelimited {
                        return Err(SerializeError::WireTypeMismatch);
                    }
                    let s = decoder.decode_string()?;
                    layer_name = s.to_string();
                }
                _ => {
                    decoder.skip_field(tag.wire_type)?;
                }
            }
        }

        Ok(Self {
            neuron_id,
            timestamp,
            spike_data,
            layer_name,
        })
    }

    fn encoded_size(&self) -> usize {
        // Simplified size calculation
        1 + varint_size(self.neuron_id as u64) +   // field 1
        1 + 8 +                                     // field 2 (f64)
        1 + varint_size(self.spike_data.len() as u64) + self.spike_data.len() + // field 3
        1 + varint_size(self.layer_name.len() as u64) + self.layer_name.len()   // field 4
    }
}

/// Calculate size of varint encoding
pub fn varint_size(mut value: u64) -> usize {
    if value == 0 {
        return 1;
    }
    
    let mut size = 0;
    while value > 0 {
        size += 1;
        value >>= 7;
    }
    size
}

/// Protocol buffer utilities
pub mod utils {
    use super::*;

    /// Encode message to bytes
    pub fn encode_message<T: ProtoMessage>(message: &T) -> Result<Vec<u8>> {
        let size = message.encoded_size();
        let mut buffer = vec![0u8; size];
        let position = {
            let mut encoder = ProtoEncoder::new(BufferMut::new(&mut buffer));
            message.encode_to(&mut encoder)?;
            encoder.position()
        };
        Ok(buffer[..position].to_vec())
    }

    /// Decode message from bytes
    pub fn decode_message<T: ProtoMessage>(bytes: &[u8]) -> Result<T> {
        let mut decoder = ProtoDecoder::new(Buffer::new(bytes));
        T::decode_from(&mut decoder)
    }

    /// Validate protobuf format
    pub fn validate_format(bytes: &[u8]) -> Result<()> {
        let mut decoder = ProtoDecoder::new(Buffer::new(bytes));
        
        while !decoder.is_at_end() {
            let tag = decoder.decode_tag()?;
            decoder.skip_field(tag.wire_type)?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_encoding() {
        let mut buffer = [0u8; 10];
        let mut encoder = ProtoEncoder::new(BufferMut::new(&mut buffer));
        
        encoder.encode_varint(150).unwrap();
        let written = encoder.written().to_vec();
        
        let mut decoder = ProtoDecoder::new(Buffer::new(&written));
        let decoded = decoder.decode_varint().unwrap();
        
        assert_eq!(decoded, 150);
    }

    #[test]
    fn test_zigzag_encoding() {
        let mut buffer = [0u8; 10];
        let mut encoder = ProtoEncoder::new(BufferMut::new(&mut buffer));
        
        encoder.encode_sint_varint(-1).unwrap();
        let written = encoder.written().to_vec();
        
        let mut decoder = ProtoDecoder::new(Buffer::new(&written));
        let decoded = decoder.decode_sint_varint().unwrap();
        
        assert_eq!(decoded, -1);
    }

    #[test]
    fn test_field_tag() {
        let tag = FieldTag::new(1, WireType::Varint).unwrap();
        let encoded = tag.encode();
        let decoded = FieldTag::decode(encoded).unwrap();
        
        assert_eq!(tag, decoded);
    }

    #[test]
    fn test_neural_message() {
        let message = NeuralMessage {
            neuron_id: 42,
            timestamp: 1.5,
            spike_data: vec![1, 2, 3, 4],
            layer_name: "input".to_string(),
        };

        let encoded = utils::encode_message(&message).unwrap();
        let decoded: NeuralMessage = utils::decode_message(&encoded).unwrap();
        
        assert_eq!(message, decoded);
    }

    #[test]
    fn test_string_field() {
        let mut buffer = [0u8; 100];
        let mut encoder = ProtoEncoder::new(BufferMut::new(&mut buffer));
        
        encoder.encode_string_field(1, "hello").unwrap();
        let written = encoder.written().to_vec();
        
        let mut decoder = ProtoDecoder::new(Buffer::new(&written));
        let tag = decoder.decode_tag().unwrap();
        let decoded_string = decoder.decode_string().unwrap();
        
        assert_eq!(tag.field_number, 1);
        assert_eq!(decoded_string, "hello");
    }

    #[test]
    fn test_varint_size_calculation() {
        assert_eq!(varint_size(0), 1);
        assert_eq!(varint_size(127), 1);
        assert_eq!(varint_size(128), 2);
        assert_eq!(varint_size(16383), 2);
        assert_eq!(varint_size(16384), 3);
    }
}