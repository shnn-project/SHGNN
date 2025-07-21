//! Fixed-point arithmetic for deterministic neuromorphic computation
//!
//! This module provides fixed-point number types that ensure deterministic
//! behavior across different hardware platforms and eliminate floating-point
//! dependencies in embedded systems.

use crate::error::{EmbeddedError, EmbeddedResult};
use core::{fmt, ops};

#[cfg(feature = "fixed-point")]
use fixed::{types::I32F32 as FixedI32F32, FixedI32, FixedU32};

/// Fixed-point number trait for neuromorphic computation
pub trait FixedPoint: 
    Copy + Clone + fmt::Debug + fmt::Display +
    ops::Add<Output = Self> + ops::Sub<Output = Self> +
    ops::Mul<Output = Self> + ops::Div<Output = Self> +
    PartialEq + PartialOrd
{
    /// Create from integer
    fn from_int(value: i32) -> Self;
    
    /// Create from float (for initialization only)
    fn from_float(value: f32) -> Self;
    
    /// Convert to float (for interfacing with external systems)
    fn to_float(self) -> f32;
    
    /// Get the zero value
    fn zero() -> Self;
    
    /// Get the one value
    fn one() -> Self;
    
    /// Saturating addition
    fn saturating_add(self, other: Self) -> Self;
    
    /// Saturating subtraction
    fn saturating_sub(self, other: Self) -> Self;
    
    /// Saturating multiplication
    fn saturating_mul(self, other: Self) -> Self;
    
    /// Checked operations
    fn checked_add(self, other: Self) -> Option<Self>;
    fn checked_sub(self, other: Self) -> Option<Self>;
    fn checked_mul(self, other: Self) -> Option<Self>;
    fn checked_div(self, other: Self) -> Option<Self>;
    
    /// Exponential function (approximated)
    fn exp(self) -> Self;
    
    /// Natural logarithm (approximated)
    fn ln(self) -> Self;
    
    /// Absolute value
    fn abs(self) -> Self;
    
    /// Maximum of two values
    fn max(self, other: Self) -> Self;
    
    /// Minimum of two values
    fn min(self, other: Self) -> Self;
}

/// Q16.16 fixed-point type (16 integer bits, 16 fractional bits)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Q16_16(i32);

impl Q16_16 {
    /// Fractional bits
    pub const FRAC_BITS: u32 = 16;
    /// Scale factor (2^16)
    pub const SCALE: i32 = 1 << Self::FRAC_BITS;
    /// Maximum value
    pub const MAX: Self = Self(i32::MAX);
    /// Minimum value
    pub const MIN: Self = Self(i32::MIN);
    
    /// Create from raw i32 value
    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }
    
    /// Get raw i32 value
    pub const fn to_raw(self) -> i32 {
        self.0
    }
    
    /// Fast exponential approximation using lookup table and interpolation
    pub fn fast_exp(self) -> Self {
        // Simplified exponential approximation
        // In a real implementation, this would use a lookup table
        // or polynomial approximation for better accuracy
        
        if self.0 < -Self::SCALE * 5 {
            return Self::zero();
        }
        if self.0 > Self::SCALE * 5 {
            return Self::from_float(148.413); // e^5
        }
        
        // Linear approximation for demonstration
        // Real implementation would use more sophisticated methods
        Self::one() + self
    }
    
    /// Sigmoid function using fixed-point arithmetic
    pub fn sigmoid(self) -> Self {
        // sigmoid(x) = 1 / (1 + exp(-x))
        let neg_x = Self::zero() - self;
        let exp_neg_x = neg_x.fast_exp();
        let denominator = Self::one() + exp_neg_x;
        Self::one() / denominator
    }
}

impl FixedPoint for Q16_16 {
    fn from_int(value: i32) -> Self {
        Self(value.saturating_mul(Self::SCALE))
    }
    
    fn from_float(value: f32) -> Self {
        Self((value * Self::SCALE as f32) as i32)
    }
    
    fn to_float(self) -> f32 {
        self.0 as f32 / Self::SCALE as f32
    }
    
    fn zero() -> Self {
        Self(0)
    }
    
    fn one() -> Self {
        Self(Self::SCALE)
    }
    
    fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }
    
    fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
    
    fn saturating_mul(self, other: Self) -> Self {
        let result = (self.0 as i64 * other.0 as i64) >> Self::FRAC_BITS;
        Self(result.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
    }
    
    fn checked_add(self, other: Self) -> Option<Self> {
        self.0.checked_add(other.0).map(Self)
    }
    
    fn checked_sub(self, other: Self) -> Option<Self> {
        self.0.checked_sub(other.0).map(Self)
    }
    
    fn checked_mul(self, other: Self) -> Option<Self> {
        let result = (self.0 as i64).checked_mul(other.0 as i64)?;
        let scaled = result >> Self::FRAC_BITS;
        if scaled >= i32::MIN as i64 && scaled <= i32::MAX as i64 {
            Some(Self(scaled as i32))
        } else {
            None
        }
    }
    
    fn checked_div(self, other: Self) -> Option<Self> {
        if other.0 == 0 {
            return None;
        }
        let result = ((self.0 as i64) << Self::FRAC_BITS) / (other.0 as i64);
        if result >= i32::MIN as i64 && result <= i32::MAX as i64 {
            Some(Self(result as i32))
        } else {
            None
        }
    }
    
    fn exp(self) -> Self {
        self.fast_exp()
    }
    
    fn ln(self) -> Self {
        // Simplified natural logarithm approximation
        // Real implementation would use more sophisticated methods
        if self.0 <= 0 {
            return Self::from_int(-10); // Approximate -infinity
        }
        
        // Linear approximation: ln(x) ≈ x - 1 for x near 1
        self - Self::one()
    }
    
    fn abs(self) -> Self {
        Self(self.0.abs())
    }
    
    fn max(self, other: Self) -> Self {
        if self.0 >= other.0 { self } else { other }
    }
    
    fn min(self, other: Self) -> Self {
        if self.0 <= other.0 { self } else { other }
    }
}

impl ops::Add for Q16_16 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl ops::Sub for Q16_16 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl ops::Mul for Q16_16 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let result = (self.0 as i64 * other.0 as i64) >> Self::FRAC_BITS;
        Self(result as i32)
    }
}

impl ops::Div for Q16_16 {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let result = ((self.0 as i64) << Self::FRAC_BITS) / (other.0 as i64);
        Self(result as i32)
    }
}

impl fmt::Display for Q16_16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.to_float())
    }
}

impl Default for Q16_16 {
    fn default() -> Self {
        Self::zero()
    }
}

/// Fixed-point spike representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedSpike<T: FixedPoint> {
    /// Source neuron ID
    pub source: u16,
    /// Timestamp in fixed-point
    pub timestamp: T,
    /// Amplitude in fixed-point
    pub amplitude: T,
}

impl<T: FixedPoint> FixedSpike<T> {
    /// Create a new fixed-point spike
    pub fn new(source: u16, timestamp: T, amplitude: T) -> Self {
        Self {
            source,
            timestamp,
            amplitude,
        }
    }
    
    /// Create a binary spike (amplitude = 1.0)
    pub fn binary(source: u16, timestamp: T) -> Self {
        Self::new(source, timestamp, T::one())
    }
}

/// Conversion utilities between floating-point and fixed-point
pub mod convert {
    use super::*;
    
    /// Convert floating-point neuron parameters to fixed-point
    pub fn float_to_fixed_config(
        tau_m: f32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
    ) -> (Q16_16, Q16_16, Q16_16, Q16_16) {
        (
            Q16_16::from_float(tau_m),
            Q16_16::from_float(v_rest),
            Q16_16::from_float(v_thresh),
            Q16_16::from_float(v_reset),
        )
    }
    
    /// Convert fixed-point results back to floating-point for output
    pub fn fixed_to_float_results(values: &[Q16_16]) -> heapless::Vec<f32, 32> {
        let mut result = heapless::Vec::new();
        for &value in values {
            let _ = result.push(value.to_float());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_q16_16_basic_ops() {
        let a = Q16_16::from_float(2.5);
        let b = Q16_16::from_float(1.5);
        
        let sum = a + b;
        assert!((sum.to_float() - 4.0).abs() < 0.001);
        
        let diff = a - b;
        assert!((diff.to_float() - 1.0).abs() < 0.001);
        
        let prod = a * b;
        assert!((prod.to_float() - 3.75).abs() < 0.001);
        
        let quot = a / b;
        assert!((quot.to_float() - (5.0/3.0)).abs() < 0.001);
    }
    
    #[test]
    fn test_q16_16_saturating_ops() {
        let max_val = Q16_16::from_raw(i32::MAX - 1000);
        let small_val = Q16_16::from_int(1);
        
        let result = max_val.saturating_add(small_val);
        assert_eq!(result.to_raw(), i32::MAX);
    }
    
    #[test]
    fn test_fixed_spike() {
        let spike = FixedSpike::binary(42, Q16_16::from_float(1.5));
        assert_eq!(spike.source, 42);
        assert!((spike.timestamp.to_float() - 1.5).abs() < 0.001);
        assert!((spike.amplitude.to_float() - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_sigmoid_approximation() {
        let x = Q16_16::from_float(0.0);
        let sig = x.sigmoid();
        // sigmoid(0) should be approximately 0.5
        assert!((sig.to_float() - 0.5).abs() < 0.1);
    }
    
    #[test]
    fn test_conversion_utilities() {
        let (tau_m, v_rest, v_thresh, v_reset) = convert::float_to_fixed_config(
            0.02, -70.0, -55.0, -75.0
        );
        
        assert!((tau_m.to_float() - 0.02).abs() < 0.001);
        assert!((v_rest.to_float() - (-70.0)).abs() < 0.001);
        assert!((v_thresh.to_float() - (-55.0)).abs() < 0.001);
        assert!((v_reset.to_float() - (-75.0)).abs() < 0.001);
    }
}