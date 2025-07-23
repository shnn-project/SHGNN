//! Matrix operations optimized for neural network computations
//! 
//! Provides SIMD-accelerated matrix operations with memory-efficient storage
//! designed specifically for weight matrices and connectivity patterns.

use core::ops::{Index, IndexMut, Mul};
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::{Float, Result, MathError, Vector, math::FloatMath};

/// Dense matrix with row-major storage
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    data: Vec<Float>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// Create new matrix with given dimensions
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Create matrix filled with given value
    pub fn filled(rows: usize, cols: usize, value: Float) -> Self {
        Self {
            data: vec![value; rows * cols],
            rows,
            cols,
        }
    }

    /// Create matrix filled with zeros
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::filled(rows, cols, 0.0)
    }

    /// Create matrix filled with ones
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self::filled(rows, cols, 1.0)
    }

    /// Create identity matrix
    pub fn identity(size: usize) -> Self {
        let mut matrix = Self::new(size, size);
        for i in 0..size {
            matrix[(i, i)] = 1.0;
        }
        matrix
    }

    /// Create matrix from 2D array data
    pub fn from_data(data: Vec<Vec<Float>>) -> Result<Self> {
        if data.is_empty() {
            return Err(MathError::InvalidInput { reason: "Empty matrix data" });
        }

        let rows = data.len();
        let cols = data[0].len();

        // Check all rows have same length
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(MathError::DimensionMismatch {
                    expected: cols,
                    got: row.len(),
                });
            }
        }

        let flat_data: Vec<Float> = data.into_iter().flatten().collect();
        
        Ok(Self {
            data: flat_data,
            rows,
            cols,
        })
    }

    /// Create matrix from flat data with dimensions
    pub fn from_flat(data: Vec<Float>, rows: usize, cols: usize) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(MathError::DimensionMismatch {
                expected: rows * cols,
                got: data.len(),
            });
        }

        Ok(Self { data, rows, cols })
    }

    /// Get matrix dimensions
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get number of rows
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get number of columns
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Check if matrix is square
    #[inline]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
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

    /// Convert linear index to (row, col)
    #[inline]
    fn to_coords(&self, index: usize) -> (usize, usize) {
        (index / self.cols, index % self.cols)
    }

    /// Convert (row, col) to linear index
    #[inline]
    fn to_index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    /// Get element at (row, col) with bounds checking
    pub fn get(&self, row: usize, col: usize) -> Result<Float> {
        if row >= self.rows || col >= self.cols {
            return Err(MathError::IndexOutOfBounds { 
                index: self.to_index(row, col), 
                len: self.data.len() 
            });
        }
        Ok(self.data[self.to_index(row, col)])
    }

    /// Set element at (row, col) with bounds checking
    pub fn set(&mut self, row: usize, col: usize, value: Float) -> Result<()> {
        if row >= self.rows || col >= self.cols {
            return Err(MathError::IndexOutOfBounds { 
                index: self.to_index(row, col), 
                len: self.data.len() 
            });
        }
        let index = self.to_index(row, col);
        self.data[index] = value;
        Ok(())
    }

    /// Get row as vector slice
    pub fn row(&self, row: usize) -> Result<&[Float]> {
        if row >= self.rows {
            return Err(MathError::IndexOutOfBounds { index: row, len: self.rows });
        }
        let start = row * self.cols;
        Ok(&self.data[start..start + self.cols])
    }

    /// Get row as vector slice (alias for compatibility)
    pub fn get_row(&self, row: usize) -> Option<&[Float]> {
        self.row(row).ok()
    }

    /// Get mutable row as vector slice
    pub fn row_mut(&mut self, row: usize) -> Result<&mut [Float]> {
        if row >= self.rows {
            return Err(MathError::IndexOutOfBounds { index: row, len: self.rows });
        }
        let start = row * self.cols;
        Ok(&mut self.data[start..start + self.cols])
    }

    /// Get column as vector (creates new allocation)
    pub fn col(&self, col: usize) -> Result<Vector> {
        if col >= self.cols {
            return Err(MathError::IndexOutOfBounds { index: col, len: self.cols });
        }

        let mut column = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            column.push(self.data[self.to_index(row, col)]);
        }
        
        Ok(Vector::from(column))
    }

    /// Set column from vector
    pub fn set_col(&mut self, col: usize, values: &Vector) -> Result<()> {
        if col >= self.cols {
            return Err(MathError::IndexOutOfBounds { index: col, len: self.cols });
        }
        if values.len() != self.rows {
            return Err(MathError::DimensionMismatch {
                expected: self.rows,
                got: values.len(),
            });
        }

        for row in 0..self.rows {
            let index = self.to_index(row, col);
            self.data[index] = values[row];
        }
        
        Ok(())
    }

    /// Matrix-vector multiplication
    pub fn mul_vec(&self, vec: &Vector) -> Result<Vector> {
        if self.cols != vec.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.cols,
                got: vec.len(),
            });
        }

        let mut result = Vector::zeros(self.rows);
        self.mul_vec_into(vec, &mut result)?;
        Ok(result)
    }

    /// Matrix-vector multiplication into existing vector (avoids allocation)
    pub fn mul_vec_into(&self, vec: &Vector, result: &mut Vector) -> Result<()> {
        if self.cols != vec.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.cols,
                got: vec.len(),
            });
        }
        if self.rows != result.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.rows,
                got: result.len(),
            });
        }

        // Ensure result is zeroed
        result.as_mut_slice().fill(0.0);

        #[cfg(feature = "simd")]
        {
            self.mul_vec_simd(vec.as_slice(), result.as_mut_slice());
        }
        #[cfg(not(feature = "simd"))]
        {
            for row in 0..self.rows {
                let row_data = &self.data[row * self.cols..(row + 1) * self.cols];
                let mut sum = 0.0;
                for (a, b) in row_data.iter().zip(vec.as_slice().iter()) {
                    sum += a * b;
                }
                result.as_mut_slice()[row] = sum;
            }
        }

        Ok(())
    }

    /// SIMD-optimized matrix-vector multiplication
    #[cfg(feature = "simd")]
    fn mul_vec_simd(&self, vec: &[Float], result: &mut [Float]) {
        use core::simd::{f32x8, SimdFloat};

        for row in 0..self.rows {
            let row_data = &self.data[row * self.cols..(row + 1) * self.cols];
            let mut sum = 0.0;
            
            let chunks = self.cols / 8;
            
            // Process 8 elements at a time
            for i in 0..chunks {
                let offset = i * 8;
                let a = f32x8::from_slice(&row_data[offset..offset + 8]);
                let b = f32x8::from_slice(&vec[offset..offset + 8]);
                sum += (a * b).reduce_sum();
            }
            
            // Handle remaining elements
            for i in (chunks * 8)..self.cols {
                sum += row_data[i] * vec[i];
            }
            
            result[row] = sum;
        }
    }

    /// Matrix multiplication
    pub fn mul_matrix(&self, other: &Matrix) -> Result<Matrix> {
        if self.cols != other.rows {
            return Err(MathError::DimensionMismatch {
                expected: self.cols,
                got: other.rows,
            });
        }

        let mut result = Matrix::new(self.rows, other.cols);
        self.mul_matrix_into(other, &mut result)?;
        Ok(result)
    }

    /// Matrix multiplication into existing matrix
    pub fn mul_matrix_into(&self, other: &Matrix, result: &mut Matrix) -> Result<()> {
        if self.cols != other.rows {
            return Err(MathError::DimensionMismatch {
                expected: self.cols,
                got: other.rows,
            });
        }
        if result.rows != self.rows || result.cols != other.cols {
            return Err(MathError::DimensionMismatch {
                expected: self.rows * other.cols,
                got: result.data.len(),
            });
        }

        result.data.fill(0.0);

        // Standard O(n³) matrix multiplication
        // TODO: Implement more efficient algorithms (Strassen, etc.) for large matrices
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self[(i, k)] * other[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }

        Ok(())
    }

    /// Transpose matrix
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(j, i)] = self[(i, j)];
            }
        }
        
        result
    }

    /// Transpose matrix in-place (only works for square matrices)
    pub fn transpose_inplace(&mut self) -> Result<()> {
        if !self.is_square() {
            return Err(MathError::InvalidInput { 
                reason: "In-place transpose only works for square matrices" 
            });
        }

        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                let temp = self[(i, j)];
                self[(i, j)] = self[(j, i)];
                self[(j, i)] = temp;
            }
        }

        Ok(())
    }

    /// Add scalar to all elements
    pub fn add_scalar(&mut self, scalar: Float) {
        for x in &mut self.data {
            *x += scalar;
        }
    }

    /// Multiply all elements by scalar
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

    /// Apply activation function element-wise
    pub fn apply_activation<F>(&mut self, activation: F)
    where
        F: Fn(Float) -> Float,
    {
        for x in &mut self.data {
            *x = activation(*x);
        }
    }

    /// Initialize with Xavier/Glorot initialization
    pub fn init_xavier(&mut self) {
        let limit = (6.0 / (self.rows + self.cols) as Float).sqrt();
        self.init_uniform(-limit, limit);
    }

    /// Initialize with He initialization (for ReLU networks)
    pub fn init_he(&mut self) {
        let std = (2.0 / self.rows as Float).sqrt();
        self.init_normal(0.0, std);
    }

    /// Initialize with uniform random values
    pub fn init_uniform(&mut self, min: Float, max: Float) {
        // Simple linear congruential generator for deterministic initialization
        let mut rng_state = 12345u64;
        for x in &mut self.data {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let uniform = (rng_state >> 16) as f32 / (1u32 << 16) as f32;
            *x = min + uniform * (max - min);
        }
    }

    /// Initialize with normal distribution (Box-Muller transform)
    pub fn init_normal(&mut self, mean: Float, std: Float) {
        let mut rng_state = 54321u64;
        let mut has_spare = false;
        let mut spare = 0.0;

        for x in &mut self.data {
            if has_spare {
                *x = mean + std * spare;
                has_spare = false;
            } else {
                // Box-Muller transform
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let u1 = (rng_state >> 16) as f32 / (1u32 << 16) as f32;
                
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let u2 = (rng_state >> 16) as f32 / (1u32 << 16) as f32;

                let mag = std * (-2.0 * u1.ln()).sqrt();
                *x = mean + mag * (2.0 * core::f32::consts::PI * u2).cos();
                spare = mag * (2.0 * core::f32::consts::PI * u2).sin();
                has_spare = true;
            }
        }
    }

    /// Sum all elements
    pub fn sum(&self) -> Float {
        self.data.iter().sum()
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

    /// Apply weight decay (L2 regularization)
    pub fn apply_weight_decay(&mut self, decay: Float) {
        let factor = 1.0 - decay;
        self.scale(factor);
    }

    /// Clip weights to prevent gradient explosion
    pub fn clip_weights(&mut self, max_value: Float) {
        for x in &mut self.data {
            *x = x.clamp(-max_value, max_value);
        }
    }
}

/// Matrix operations trait
pub trait MatrixOps<Rhs = Self> {
    type Output;

    fn add(self, rhs: Rhs) -> Self::Output;
    fn sub(self, rhs: Rhs) -> Self::Output;
    fn mul(self, rhs: Rhs) -> Self::Output;
}

impl MatrixOps for Matrix {
    type Output = Result<Matrix>;

    fn add(self, rhs: Matrix) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(MathError::DimensionMismatch {
                expected: self.data.len(),
                got: rhs.data.len(),
            });
        }

        let data = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }

    fn sub(self, rhs: Matrix) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(MathError::DimensionMismatch {
                expected: self.data.len(),
                got: rhs.data.len(),
            });
        }

        let data = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Ok(Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }

    fn mul(self, rhs: Matrix) -> Self::Output {
        self.mul_matrix(&rhs)
    }
}

// Standard trait implementations
impl Index<(usize, usize)> for Matrix {
    type Output = Float;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[self.to_index(row, col)]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        let index = self.to_index(row, col);
        &mut self.data[index]
    }
}

impl Mul<Vector> for Matrix {
    type Output = Result<Vector>;

    fn mul(self, rhs: Vector) -> Self::Output {
        self.mul_vec(&rhs)
    }
}

impl Mul<&Vector> for &Matrix {
    type Output = Result<Vector>;

    fn mul(self, rhs: &Vector) -> Self::Output {
        self.mul_vec(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m = Matrix::new(3, 4);
        assert_eq!(m.dims(), (3, 4));
        assert_eq!(m.data.len(), 12);

        let identity = Matrix::identity(3);
        assert_eq!(identity[(0, 0)], 1.0);
        assert_eq!(identity[(0, 1)], 0.0);
        assert_eq!(identity[(1, 1)], 1.0);
    }

    #[test]
    fn test_matrix_vector_multiplication() {
        let mut m = Matrix::new(2, 3);
        m[(0, 0)] = 1.0; m[(0, 1)] = 2.0; m[(0, 2)] = 3.0;
        m[(1, 0)] = 4.0; m[(1, 1)] = 5.0; m[(1, 2)] = 6.0;

        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = m.mul_vec(&v).unwrap();
        
        assert_eq!(result.as_slice(), &[14.0, 32.0]); // [1*1+2*2+3*3, 4*1+5*2+6*3]
    }

    #[test]
    fn test_matrix_multiplication() {
        let mut a = Matrix::new(2, 3);
        a[(0, 0)] = 1.0; a[(0, 1)] = 2.0; a[(0, 2)] = 3.0;
        a[(1, 0)] = 4.0; a[(1, 1)] = 5.0; a[(1, 2)] = 6.0;

        let mut b = Matrix::new(3, 2);
        b[(0, 0)] = 7.0; b[(0, 1)] = 8.0;
        b[(1, 0)] = 9.0; b[(1, 1)] = 10.0;
        b[(2, 0)] = 11.0; b[(2, 1)] = 12.0;

        let result = a.mul_matrix(&b).unwrap();
        assert_eq!(result.dims(), (2, 2));
        assert_eq!(result[(0, 0)], 58.0); // 1*7 + 2*9 + 3*11
        assert_eq!(result[(0, 1)], 64.0); // 1*8 + 2*10 + 3*12
    }

    #[test]
    fn test_transpose() {
        let mut m = Matrix::new(2, 3);
        m[(0, 0)] = 1.0; m[(0, 1)] = 2.0; m[(0, 2)] = 3.0;
        m[(1, 0)] = 4.0; m[(1, 1)] = 5.0; m[(1, 2)] = 6.0;

        let t = m.transpose();
        assert_eq!(t.dims(), (3, 2));
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(1, 0)], 2.0);
        assert_eq!(t[(2, 1)], 6.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let m = Matrix::new(2, 3);
        let v = Vector::zeros(4); // Wrong size
        assert!(matches!(m.mul_vec(&v), Err(MathError::DimensionMismatch { .. })));
    }
}

// Standalone matrix functions for compatibility
pub fn matrix_multiply(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    a.mul_matrix(b)
}

pub fn transpose(matrix: &Matrix) -> Matrix {
    matrix.transpose()
}

pub fn inverse_2x2(matrix: &Matrix) -> Result<Matrix> {
    if matrix.rows() != 2 || matrix.cols() != 2 {
        return Err(MathError::DimensionMismatch { 
            expected: 2, 
            got: matrix.rows() 
        });
    }
    
    let a = matrix.get(0, 0)?;
    let b = matrix.get(0, 1)?;
    let c = matrix.get(1, 0)?;
    let d = matrix.get(1, 1)?;
    
    let det = a * d - b * c;
    if det.abs() < crate::constants::EPSILON {
        return Err(MathError::SingularMatrix);
    }
    
    let inv_det = 1.0 / det;
    Matrix::from_data(vec![
        vec![d * inv_det, -b * inv_det],
        vec![-c * inv_det, a * inv_det],
    ])
}

pub fn determinant(matrix: &Matrix) -> Result<f32> {
    if matrix.rows() != matrix.cols() {
        return Err(MathError::DimensionMismatch { 
            expected: matrix.rows(), 
            got: matrix.cols() 
        });
    }
    
    if matrix.rows() == 2 {
        let a = matrix.get(0, 0)?;
        let b = matrix.get(0, 1)?;
        let c = matrix.get(1, 0)?;
        let d = matrix.get(1, 1)?;
        Ok(a * d - b * c)
    } else {
        // For larger matrices, we'd need a more complex implementation
        // For now, return an error
        Err(MathError::UnsupportedOperation)
    }
}