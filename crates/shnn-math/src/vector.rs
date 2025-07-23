//! Vector operations optimized for neuromorphic computations
//! 
//! This module provides SIMD-accelerated vector operations with zero external dependencies.
//! Designed specifically for spike train processing and neural state vectors.

use core::ops::{Add, Sub, Mul, Div, Index, IndexMut};
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::{Float, Result, MathError, math::FloatMath};

/// Dynamic vector with SIMD-optimized operations
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    data: Vec<Float>,
}

impl Vector {
    /// Create new vector with given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Create zero vector of given size
    pub fn zeros(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
        }
    }

    /// Create vector filled with given value
    pub fn filled(size: usize, value: Float) -> Self {
        Self {
            data: vec![value; size],
        }
    }

    /// Create vector filled with ones
    pub fn ones(size: usize) -> Self {
        Self::filled(size, 1.0)
    }

    /// Create vector from raw data
    pub fn from_slice(data: &[Float]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }

    /// Create vector from iterator
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Float>,
    {
        Self {
            data: iter.into_iter().collect(),
        }
    }

    /// Get vector length
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if vector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get raw data slice
    #[inline]
    pub fn as_slice(&self) -> &[Float] {
        &self.data
    }

    /// Get mutable raw data slice
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Float] {
        &mut self.data
    }

    /// Push element to vector
    pub fn push(&mut self, value: Float) {
        self.data.push(value);
    }

    /// Resize vector
    pub fn resize(&mut self, new_len: usize, value: Float) {
        self.data.resize(new_len, value);
    }

    /// Clear vector
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get element at index with bounds checking
    pub fn get(&self, index: usize) -> Result<Float> {
        self.data.get(index).copied()
            .ok_or(MathError::IndexOutOfBounds { index, len: self.len() })
    }

    /// Set element at index with bounds checking  
    pub fn set(&mut self, index: usize, value: Float) -> Result<()> {
        if index >= self.len() {
            return Err(MathError::IndexOutOfBounds { index, len: self.len() });
        }
        self.data[index] = value;
        Ok(())
    }

    /// Compute dot product with another vector
    pub fn dot(&self, other: &Vector) -> Result<Float> {
        if self.len() != other.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }

        Ok(self.dot_unchecked(other))
    }

    /// Compute dot product without bounds checking (unsafe but fast)
    #[inline]
    pub fn dot_unchecked(&self, other: &Vector) -> Float {
        #[cfg(feature = "simd")]
        {
            self.dot_simd(&other.data)
        }
        #[cfg(not(feature = "simd"))]
        {
            self.data.iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .sum()
        }
    }

    /// SIMD-optimized dot product
    #[cfg(feature = "simd")]
    fn dot_simd(&self, other: &[Float]) -> Float {
        use core::simd::{f32x8, SimdFloat};
        
        let mut sum = 0.0;
        let chunks = self.data.len() / 8;
        
        // Process 8 elements at a time with SIMD
        for i in 0..chunks {
            let offset = i * 8;
            let a = f32x8::from_slice(&self.data[offset..offset + 8]);
            let b = f32x8::from_slice(&other[offset..offset + 8]);
            sum += (a * b).reduce_sum();
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..self.data.len() {
            sum += self.data[i] * other[i];
        }
        
        sum
    }

    /// Compute L2 norm (magnitude)
    pub fn norm(&self) -> Float {
        self.dot_unchecked(self).sqrt()
    }

    /// Compute squared L2 norm
    pub fn norm_squared(&self) -> Float {
        self.dot_unchecked(self)
    }

    /// Normalize vector to unit length
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > crate::constants::EPSILON {
            self.scale(1.0 / norm);
        }
    }

    /// Get normalized copy of vector
    pub fn normalized(&self) -> Vector {
        let mut result = self.clone();
        result.normalize();
        result
    }

    /// Scale vector by scalar
    pub fn scale(&mut self, scalar: Float) {
        #[cfg(feature = "simd")]
        {
            self.scale_simd(scalar);
        }
        #[cfg(not(feature = "simd"))]
        {
            for x in &mut self.data {
                *x *= scalar;
            }
        }
    }

    /// SIMD-optimized scaling
    #[cfg(feature = "simd")]
    fn scale_simd(&mut self, scalar: Float) {
        use core::simd::{f32x8, SimdFloat};
        
        let scalar_vec = f32x8::splat(scalar);
        let chunks = self.data.len() / 8;
        
        // Process 8 elements at a time
        for i in 0..chunks {
            let offset = i * 8;
            let mut chunk = f32x8::from_slice(&self.data[offset..offset + 8]);
            chunk *= scalar_vec;
            chunk.copy_to_slice(&mut self.data[offset..offset + 8]);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..self.data.len() {
            self.data[i] *= scalar;
        }
    }

    /// Add another vector in-place
    pub fn add_assign(&mut self, other: &Vector) -> Result<()> {
        if self.len() != other.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }

        self.add_assign_unchecked(other);
        Ok(())
    }

    /// Add another vector in-place without bounds checking
    #[inline]
    pub fn add_assign_unchecked(&mut self, other: &Vector) {
        #[cfg(feature = "simd")]
        {
            self.add_simd(&other.data);
        }
        #[cfg(not(feature = "simd"))]
        {
            for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
                *a += b;
            }
        }
    }

    /// SIMD-optimized addition
    #[cfg(feature = "simd")]
    fn add_simd(&mut self, other: &[Float]) {
        use core::simd::f32x8;
        
        let chunks = self.data.len() / 8;
        
        for i in 0..chunks {
            let offset = i * 8;
            let mut a = f32x8::from_slice(&self.data[offset..offset + 8]);
            let b = f32x8::from_slice(&other[offset..offset + 8]);
            a += b;
            a.copy_to_slice(&mut self.data[offset..offset + 8]);
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..self.data.len() {
            self.data[i] += other[i];
        }
    }

    /// Compute element-wise exponential (for softmax)
    pub fn exp(&mut self) {
        for x in &mut self.data {
            *x = x.exp();
        }
    }

    /// Apply activation function element-wise
    pub fn apply_activation<F>(&mut self, activation: F)
    where
        F: Fn(Float) -> Float,
    {
        for x in &mut self.data {
            *x = activation(*x);
        }
    }

    /// Sum all elements
    pub fn sum(&self) -> Float {
        #[cfg(feature = "simd")]
        {
            self.sum_simd()
        }
        #[cfg(not(feature = "simd"))]
        {
            self.data.iter().sum()
        }
    }

    /// SIMD-optimized sum
    #[cfg(feature = "simd")]
    fn sum_simd(&self) -> Float {
        use core::simd::{f32x8, SimdFloat};
        
        let mut sum = 0.0;
        let chunks = self.data.len() / 8;
        
        for i in 0..chunks {
            let offset = i * 8;
            let chunk = f32x8::from_slice(&self.data[offset..offset + 8]);
            sum += chunk.reduce_sum();
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..self.data.len() {
            sum += self.data[i];
        }
        
        sum
    }

    /// Find maximum element
    pub fn max(&self) -> Option<Float> {
        self.data.iter().copied().fold(None, |acc, x| {
            Some(match acc {
                Some(acc) => acc.max(x),
                None => x,
            })
        })
    }

    /// Find minimum element
    pub fn min(&self) -> Option<Float> {
        self.data.iter().copied().fold(None, |acc, x| {
            Some(match acc {
                Some(acc) => acc.min(x),
                None => x,
            })
        })
    }

    /// Apply thresholding for spike detection
    pub fn threshold(&mut self, threshold: Float) -> usize {
        let mut spike_count = 0;
        for x in &mut self.data {
            if *x >= threshold {
                spike_count += 1;
                *x = 0.0; // Reset after spike
            }
        }
        spike_count
    }

    /// Create spike train from membrane potentials
    pub fn to_spikes(&self, threshold: Float) -> Vec<bool> {
        self.data.iter().map(|&x| x >= threshold).collect()
    }
}

/// Vector operations trait for generic implementations
pub trait VectorOps<Rhs = Self> {
    type Output;

    fn add(self, rhs: Rhs) -> Self::Output;
    fn sub(self, rhs: Rhs) -> Self::Output;
    fn mul(self, rhs: Rhs) -> Self::Output;
    fn div(self, rhs: Rhs) -> Self::Output;
}

impl VectorOps for Vector {
    type Output = Result<Vector>;

    fn add(self, rhs: Vector) -> Self::Output {
        if self.len() != rhs.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.len(),
                got: rhs.len(),
            });
        }

        let data = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Vector { data })
    }

    fn sub(self, rhs: Vector) -> Self::Output {
        if self.len() != rhs.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.len(),
                got: rhs.len(),
            });
        }

        let data = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Ok(Vector { data })
    }

    fn mul(self, rhs: Vector) -> Self::Output {
        if self.len() != rhs.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.len(),
                got: rhs.len(),
            });
        }

        let data = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Ok(Vector { data })
    }

    fn div(self, rhs: Vector) -> Self::Output {
        if self.len() != rhs.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.len(),
                got: rhs.len(),
            });
        }

        let data: Result<Vec<Float>> = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| {
                if b.abs() < crate::constants::EPSILON {
                    Err(MathError::DivisionByZero)
                } else {
                    Ok(a / b)
                }
            })
            .collect();

        Ok(Vector { data: data? })
    }
}

// Standard trait implementations
impl Index<usize> for Vector {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Add<Float> for Vector {
    type Output = Vector;

    fn add(mut self, rhs: Float) -> Self::Output {
        for x in &mut self.data {
            *x += rhs;
        }
        self
    }
}

impl Mul<Float> for Vector {
    type Output = Vector;

    fn mul(mut self, rhs: Float) -> Self::Output {
        self.scale(rhs);
        self
    }
}

impl Add<&Vector> for &Vector {
    type Output = Vector;

    fn add(self, rhs: &Vector) -> Self::Output {
        assert_eq!(self.len(), rhs.len(), "Vector dimensions must match for addition");
        
        let data = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        
        Vector { data }
    }
}

impl From<Vec<Float>> for Vector {
    fn from(data: Vec<Float>) -> Self {
        Self { data }
    }
}

impl From<&[Float]> for Vector {
    fn from(data: &[Float]) -> Self {
        Self::from_slice(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let v = Vector::zeros(5);
        assert_eq!(v.len(), 5);
        assert!(v.as_slice().iter().all(|&x| x == 0.0));

        let v = Vector::filled(3, 2.5);
        assert_eq!(v.len(), 3);
        assert!(v.as_slice().iter().all(|&x| x == 2.5));
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let v2 = Vector::from_slice(&[4.0, 5.0, 6.0]);
        let result = v1.dot(&v2).unwrap();
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_norm() {
        let v = Vector::from_slice(&[3.0, 4.0]);
        assert_eq!(v.norm(), 5.0);
        assert_eq!(v.norm_squared(), 25.0);
    }

    #[test]
    fn test_vector_addition() {
        let v1 = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let v2 = Vector::from_slice(&[4.0, 5.0, 6.0]);
        let result = v1.add(v2).unwrap();
        assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_spike_detection() {
        let mut v = Vector::from_slice(&[-70.0, -45.0, -30.0, -60.0]);
        let spike_count = v.threshold(-50.0);
        assert_eq!(spike_count, 2); // -45.0 and -30.0 exceed threshold
    }

    #[test]
    fn test_dimension_mismatch() {
        let v1 = Vector::zeros(3);
        let v2 = Vector::zeros(5);
        assert!(matches!(v1.dot(&v2), Err(MathError::DimensionMismatch { .. })));
    }
}

// Standalone vector functions for compatibility
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let mag_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum.sqrt()
}