//! Sparse data structures optimized for synaptic connectivity patterns
//! 
//! Provides memory-efficient storage for sparse matrices and vectors commonly
//! found in neural network connectivity patterns, with zero external dependencies.

use core::cmp::Ordering;
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::{Float, Result, MathError, Vector};

/// Compressed Sparse Row (CSR) matrix format
/// Optimized for matrix-vector multiplication in neural networks
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Non-zero values
    values: Vec<Float>,
    /// Column indices for each value
    col_indices: Vec<usize>,
    /// Row pointers (cumulative count of non-zeros)
    row_ptrs: Vec<usize>,
    /// Matrix dimensions
    rows: usize,
    cols: usize,
}

impl SparseMatrix {
    /// Create new empty sparse matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptrs: vec![0; rows + 1],
            rows,
            cols,
        }
    }

    /// Create sparse matrix with estimated capacity
    pub fn with_capacity(rows: usize, cols: usize, nnz_estimate: usize) -> Self {
        Self {
            values: Vec::with_capacity(nnz_estimate),
            col_indices: Vec::with_capacity(nnz_estimate),
            row_ptrs: vec![0; rows + 1],
            rows,
            cols,
        }
    }

    /// Create sparse matrix from triplets (row, col, value)
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: Vec<(usize, usize, Float)>,
    ) -> Result<Self> {
        // Sort triplets by row, then column
        let mut sorted_triplets = triplets;
        sorted_triplets.sort_by(|a, b| {
            match a.0.cmp(&b.0) {
                Ordering::Equal => a.1.cmp(&b.1),
                other => other,
            }
        });

        let mut matrix = Self::with_capacity(rows, cols, sorted_triplets.len());
        
        let mut current_row = 0;
        for (row, col, value) in sorted_triplets {
            if row >= rows || col >= cols {
                return Err(MathError::IndexOutOfBounds { 
                    index: row * cols + col, 
                    len: rows * cols 
                });
            }

            // Update row pointers
            while current_row <= row {
                matrix.row_ptrs[current_row + 1] = matrix.values.len();
                current_row += 1;
            }

            matrix.values.push(value);
            matrix.col_indices.push(col);
        }

        // Finalize row pointers
        while current_row < rows {
            matrix.row_ptrs[current_row + 1] = matrix.values.len();
            current_row += 1;
        }

        Ok(matrix)
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

    /// Get number of non-zero elements
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get sparsity ratio (0.0 = dense, 1.0 = empty)
    #[inline]
    pub fn sparsity(&self) -> Float {
        1.0 - (self.nnz() as Float) / (self.rows * self.cols) as Float
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Result<Float> {
        if row >= self.rows || col >= self.cols {
            return Err(MathError::IndexOutOfBounds { 
                index: row * self.cols + col, 
                len: self.rows * self.cols 
            });
        }

        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];

        // Binary search for column index
        let row_cols = &self.col_indices[start..end];
        match row_cols.binary_search(&col) {
            Ok(idx) => Ok(self.values[start + idx]),
            Err(_) => Ok(0.0),
        }
    }

    /// Set element at (row, col) - Note: This is expensive for CSR format
    pub fn set(&mut self, row: usize, col: usize, value: Float) -> Result<()> {
        if row >= self.rows || col >= self.cols {
            return Err(MathError::IndexOutOfBounds { 
                index: row * self.cols + col, 
                len: self.rows * self.cols 
            });
        }

        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];

        // Binary search for column index
        let row_cols = &self.col_indices[start..end];
        match row_cols.binary_search(&col) {
            Ok(idx) => {
                // Element exists, update value
                if value.abs() < crate::constants::EPSILON {
                    // Remove zero element
                    self.values.remove(start + idx);
                    self.col_indices.remove(start + idx);
                    // Update row pointers
                    for i in (row + 1)..=self.rows {
                        self.row_ptrs[i] -= 1;
                    }
                } else {
                    self.values[start + idx] = value;
                }
            }
            Err(insert_idx) => {
                // Element doesn't exist, insert if non-zero
                if value.abs() >= crate::constants::EPSILON {
                    self.values.insert(start + insert_idx, value);
                    self.col_indices.insert(start + insert_idx, col);
                    // Update row pointers
                    for i in (row + 1)..=self.rows {
                        self.row_ptrs[i] += 1;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get row data (values and column indices)
    pub fn row(&self, row: usize) -> Result<(&[Float], &[usize])> {
        if row >= self.rows {
            return Err(MathError::IndexOutOfBounds { index: row, len: self.rows });
        }

        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];

        Ok((&self.values[start..end], &self.col_indices[start..end]))
    }

    /// Sparse matrix-vector multiplication: y = A * x
    pub fn mul_vec(&self, x: &Vector) -> Result<Vector> {
        if self.cols != x.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.cols,
                got: x.len(),
            });
        }

        let mut y = Vector::zeros(self.rows);
        self.mul_vec_into(x, &mut y)?;
        Ok(y)
    }

    /// Sparse matrix-vector multiplication into existing vector
    pub fn mul_vec_into(&self, x: &Vector, y: &mut Vector) -> Result<()> {
        if self.cols != x.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.cols,
                got: x.len(),
            });
        }
        if self.rows != y.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.rows,
                got: y.len(),
            });
        }

        let x_data = x.as_slice();
        let y_data = y.as_mut_slice();

        // Zero output vector
        y_data.fill(0.0);

        // Perform sparse matrix-vector multiplication
        for row in 0..self.rows {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];

            let mut sum = 0.0;
            for k in start..end {
                sum += self.values[k] * x_data[self.col_indices[k]];
            }
            y_data[row] = sum;
        }

        Ok(())
    }

    /// Transpose sparse matrix
    pub fn transpose(&self) -> SparseMatrix {
        let mut triplets = Vec::with_capacity(self.nnz());

        for row in 0..self.rows {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];

            for k in start..end {
                triplets.push((self.col_indices[k], row, self.values[k]));
            }
        }

        // This will sort and build the transposed matrix
        SparseMatrix::from_triplets(self.cols, self.rows, triplets)
            .expect("Transpose should always succeed")
    }

    /// Scale all values by a scalar
    pub fn scale(&mut self, scalar: Float) {
        for value in &mut self.values {
            *value *= scalar;
        }
    }

    /// Apply threshold to remove small values
    pub fn threshold(&mut self, threshold: Float) {
        let mut write_idx = 0;
        let mut row_offsets = vec![0; self.rows];

        for row in 0..self.rows {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];

            for k in start..end {
                if self.values[k].abs() >= threshold {
                    self.values[write_idx] = self.values[k];
                    self.col_indices[write_idx] = self.col_indices[k];
                    write_idx += 1;
                }
            }

            row_offsets[row] = write_idx;
        }

        // Truncate vectors and update row pointers
        self.values.truncate(write_idx);
        self.col_indices.truncate(write_idx);

        self.row_ptrs[0] = 0;
        for row in 0..self.rows {
            self.row_ptrs[row + 1] = row_offsets[row];
        }
    }

    /// Extract diagonal elements
    pub fn diagonal(&self) -> Vector {
        let size = self.rows.min(self.cols);
        let mut diag = Vector::zeros(size);

        for i in 0..size {
            if let Ok(value) = self.get(i, i) {
                diag.as_mut_slice()[i] = value;
            }
        }

        diag
    }

    /// Create random sparse matrix with given sparsity
    pub fn random_sparse(rows: usize, cols: usize, sparsity: Float) -> Self {
        let total_elements = rows * cols;
        let nnz = ((1.0 - sparsity) * total_elements as Float) as usize;

        let mut triplets = Vec::with_capacity(nnz);
        let mut rng_state = 67890u64;

        for _ in 0..nnz {
            // Generate random row and column
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let row = (rng_state % rows as u64) as usize;

            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let col = (rng_state % cols as u64) as usize;

            // Generate random value
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let value = ((rng_state >> 16) as f32 / (1u32 << 16) as f32) * 2.0 - 1.0;

            triplets.push((row, col, value));
        }

        SparseMatrix::from_triplets(rows, cols, triplets)
            .expect("Random sparse matrix creation should succeed")
    }
}

/// Sparse vector using coordinate format
#[derive(Debug, Clone)]
pub struct SparseVector {
    /// Non-zero values
    values: Vec<Float>,
    /// Indices of non-zero values
    indices: Vec<usize>,
    /// Vector length
    length: usize,
}

impl SparseVector {
    /// Create new empty sparse vector
    pub fn new(length: usize) -> Self {
        Self {
            values: Vec::new(),
            indices: Vec::new(),
            length,
        }
    }

    /// Create sparse vector with estimated capacity
    pub fn with_capacity(length: usize, nnz_estimate: usize) -> Self {
        Self {
            values: Vec::with_capacity(nnz_estimate),
            indices: Vec::with_capacity(nnz_estimate),
            length,
        }
    }

    /// Create sparse vector from coordinate pairs
    pub fn from_coords(length: usize, coords: Vec<(usize, Float)>) -> Result<Self> {
        let mut sorted_coords = coords;
        sorted_coords.sort_by_key(|&(idx, _)| idx);

        let mut vector = Self::with_capacity(length, sorted_coords.len());

        for (idx, value) in sorted_coords {
            if idx >= length {
                return Err(MathError::IndexOutOfBounds { index: idx, len: length });
            }

            if value.abs() >= crate::constants::EPSILON {
                vector.values.push(value);
                vector.indices.push(idx);
            }
        }

        Ok(vector)
    }

    /// Get vector length
    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if vector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get number of non-zero elements
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get sparsity ratio
    #[inline]
    pub fn sparsity(&self) -> Float {
        1.0 - (self.nnz() as Float) / self.length as Float
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Result<Float> {
        if index >= self.length {
            return Err(MathError::IndexOutOfBounds { index, len: self.length });
        }

        match self.indices.binary_search(&index) {
            Ok(idx) => Ok(self.values[idx]),
            Err(_) => Ok(0.0),
        }
    }

    /// Set element at index
    pub fn set(&mut self, index: usize, value: Float) -> Result<()> {
        if index >= self.length {
            return Err(MathError::IndexOutOfBounds { index, len: self.length });
        }

        match self.indices.binary_search(&index) {
            Ok(idx) => {
                if value.abs() < crate::constants::EPSILON {
                    // Remove zero element
                    self.values.remove(idx);
                    self.indices.remove(idx);
                } else {
                    self.values[idx] = value;
                }
            }
            Err(insert_idx) => {
                if value.abs() >= crate::constants::EPSILON {
                    self.values.insert(insert_idx, value);
                    self.indices.insert(insert_idx, index);
                }
            }
        }

        Ok(())
    }

    /// Convert to dense vector
    pub fn to_dense(&self) -> Vector {
        let mut dense = Vector::zeros(self.length);
        for (&idx, &value) in self.indices.iter().zip(self.values.iter()) {
            dense.as_mut_slice()[idx] = value;
        }
        dense
    }

    /// Create sparse vector from dense vector
    pub fn from_dense(dense: &Vector) -> Self {
        let mut coords = Vec::new();
        for (idx, &value) in dense.as_slice().iter().enumerate() {
            if value.abs() >= crate::constants::EPSILON {
                coords.push((idx, value));
            }
        }

        Self::from_coords(dense.len(), coords)
            .expect("Conversion from dense should always succeed")
    }

    /// Dot product with another sparse vector
    pub fn dot(&self, other: &SparseVector) -> Result<Float> {
        if self.length != other.length {
            return Err(MathError::DimensionMismatch {
                expected: self.length,
                got: other.length,
            });
        }

        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;

        // Merge-like algorithm for sparse dot product
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                Ordering::Equal => {
                    result += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
            }
        }

        Ok(result)
    }

    /// Dot product with dense vector
    pub fn dot_dense(&self, dense: &Vector) -> Result<Float> {
        if self.length != dense.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.length,
                got: dense.len(),
            });
        }

        let mut result = 0.0;
        let dense_data = dense.as_slice();

        for (&idx, &value) in self.indices.iter().zip(self.values.iter()) {
            result += value * dense_data[idx];
        }

        Ok(result)
    }

    /// Scale vector by scalar
    pub fn scale(&mut self, scalar: Float) {
        for value in &mut self.values {
            *value *= scalar;
        }
    }

    /// Apply threshold to remove small values
    pub fn threshold(&mut self, threshold: Float) {
        let mut write_idx = 0;

        for read_idx in 0..self.values.len() {
            if self.values[read_idx].abs() >= threshold {
                self.values[write_idx] = self.values[read_idx];
                self.indices[write_idx] = self.indices[read_idx];
                write_idx += 1;
            }
        }

        self.values.truncate(write_idx);
        self.indices.truncate(write_idx);
    }

    /// Get non-zero elements as slices
    pub fn non_zeros(&self) -> (&[usize], &[Float]) {
        (&self.indices, &self.values)
    }
}

/// Sparse connectivity pattern for neural networks
#[derive(Debug, Clone)]
pub struct SynapticConnectivity {
    /// Connection matrix (pre-synaptic -> post-synaptic)
    connections: SparseMatrix,
    /// Synaptic weights
    weights: SparseMatrix,
    /// Delay matrix (in time steps)
    delays: Vec<u8>,
}

impl SynapticConnectivity {
    /// Create new synaptic connectivity pattern
    pub fn new(pre_neurons: usize, post_neurons: usize) -> Self {
        Self {
            connections: SparseMatrix::new(post_neurons, pre_neurons),
            weights: SparseMatrix::new(post_neurons, pre_neurons),
            delays: Vec::new(),
        }
    }

    /// Add synaptic connection
    pub fn add_connection(
        &mut self,
        pre: usize,
        post: usize,
        weight: Float,
        delay: u8,
    ) -> Result<()> {
        self.connections.set(post, pre, 1.0)?;
        self.weights.set(post, pre, weight)?;
        
        // Store delay information (simplified - could use sparse structure)
        while self.delays.len() <= post * self.connections.cols() + pre {
            self.delays.push(0);
        }
        self.delays[post * self.connections.cols() + pre] = delay;

        Ok(())
    }

    /// Get connection weight
    pub fn get_weight(&self, pre: usize, post: usize) -> Result<Float> {
        self.weights.get(post, pre)
    }

    /// Get connection delay
    pub fn get_delay(&self, pre: usize, post: usize) -> u8 {
        let idx = post * self.connections.cols() + pre;
        self.delays.get(idx).copied().unwrap_or(0)
    }

    /// Apply synaptic scaling
    pub fn apply_scaling(&mut self, factor: Float) {
        self.weights.scale(factor);
    }

    /// Apply spike-timing dependent plasticity (simplified)
    pub fn apply_stdp(&mut self, pre_spikes: &[bool], post_spikes: &[bool], learning_rate: Float) {
        // Simplified STDP implementation
        for (post, pre_spike) in pre_spikes.iter().enumerate() {
            for (pre, post_spike) in post_spikes.iter().enumerate() {
                if *pre_spike && *post_spike {
                    // Potentiation (simplified)
                    if let Ok(current_weight) = self.weights.get(post, pre) {
                        let _ = self.weights.set(post, pre, current_weight + learning_rate);
                    }
                } else if *pre_spike || *post_spike {
                    // Depression (simplified)
                    if let Ok(current_weight) = self.weights.get(post, pre) {
                        let _ = self.weights.set(post, pre, current_weight - learning_rate * 0.5);
                    }
                }
            }
        }
    }

    /// Get connection statistics
    pub fn stats(&self) -> (usize, usize, Float) {
        let total_connections = self.connections.nnz();
        let total_possible = self.connections.rows() * self.connections.cols();
        let connectivity_ratio = total_connections as Float / total_possible as Float;
        
        (total_connections, total_possible, connectivity_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_matrix_creation() {
        let triplets = vec![
            (0, 0, 1.0),
            (0, 2, 3.0),
            (1, 1, 2.0),
            (2, 0, 4.0),
        ];

        let sparse = SparseMatrix::from_triplets(3, 3, triplets).unwrap();
        assert_eq!(sparse.nnz(), 4);
        assert_eq!(sparse.get(0, 0).unwrap(), 1.0);
        assert_eq!(sparse.get(0, 1).unwrap(), 0.0);
        assert_eq!(sparse.get(1, 1).unwrap(), 2.0);
    }

    #[test]
    fn test_sparse_matrix_vector_multiplication() {
        let triplets = vec![
            (0, 0, 2.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 2, 3.0),
        ];

        let sparse = SparseMatrix::from_triplets(2, 3, triplets).unwrap();
        let x = Vector::from_slice(&[1.0, 2.0, 1.0]);
        let y = sparse.mul_vec(&x).unwrap();

        assert_eq!(y.as_slice(), &[4.0, 4.0]); // [2*1 + 1*2, 1*1 + 3*1]
    }

    #[test]
    fn test_sparse_vector() {
        let coords = vec![(0, 1.0), (2, 3.0), (4, 2.0)];
        let sparse = SparseVector::from_coords(5, coords).unwrap();

        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.get(0).unwrap(), 1.0);
        assert_eq!(sparse.get(1).unwrap(), 0.0);
        assert_eq!(sparse.get(2).unwrap(), 3.0);
    }

    #[test]
    fn test_sparse_vector_dot_product() {
        let coords1 = vec![(0, 2.0), (2, 1.0)];
        let coords2 = vec![(0, 1.0), (1, 3.0), (2, 2.0)];

        let v1 = SparseVector::from_coords(3, coords1).unwrap();
        let v2 = SparseVector::from_coords(3, coords2).unwrap();

        let result = v1.dot(&v2).unwrap();
        assert_eq!(result, 4.0); // 2*1 + 1*2 = 4
    }

    #[test]
    fn test_sparse_transpose() {
        let triplets = vec![
            (0, 1, 2.0),
            (1, 0, 3.0),
            (1, 2, 4.0),
        ];

        let sparse = SparseMatrix::from_triplets(2, 3, triplets).unwrap();
        let transposed = sparse.transpose();

        assert_eq!(transposed.dims(), (3, 2));
        assert_eq!(transposed.get(1, 0).unwrap(), 2.0);
        assert_eq!(transposed.get(0, 1).unwrap(), 3.0);
        assert_eq!(transposed.get(2, 1).unwrap(), 4.0);
    }

    #[test]
    fn test_synaptic_connectivity() {
        let mut connectivity = SynapticConnectivity::new(3, 2);
        
        connectivity.add_connection(0, 1, 0.5, 2).unwrap();
        connectivity.add_connection(1, 0, -0.3, 1).unwrap();

        assert_eq!(connectivity.get_weight(0, 1).unwrap(), 0.5);
        assert_eq!(connectivity.get_delay(0, 1), 2);
        assert_eq!(connectivity.get_weight(1, 0).unwrap(), -0.3);
    }
}

// Type alias for compatibility
pub type CSRMatrix = SparseMatrix;

// Standalone sparse matrix functions for compatibility
pub fn sparse_multiply(a: &SparseMatrix, b: &SparseMatrix) -> Result<SparseMatrix> {
    // Basic placeholder implementation - optimize later
    if a.cols() != b.rows() {
        return Err(MathError::DimensionMismatch {
            expected: a.cols(),
            got: b.rows(),
        });
    }
    
    let mut result = SparseMatrix::new(a.rows(), b.cols());
    // TODO: Implement actual sparse matrix multiplication
    Ok(result)
}

pub fn sparse_add(a: &SparseMatrix, b: &SparseMatrix) -> Result<SparseMatrix> {
    // Basic placeholder implementation - optimize later
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(MathError::DimensionMismatch {
            expected: a.rows(),
            got: b.rows(),
        });
    }
    
    let mut result = SparseMatrix::new(a.rows(), a.cols());
    // TODO: Implement actual sparse matrix addition
    Ok(result)
}