//! Comprehensive tests for SHNN math library refactoring
//!
//! This test suite validates all the zero-dependency math components
//! that replaced nalgebra and ndarray functionality.

use shnn_math::{
    Vector, Matrix, SparseMatrix,
    activation::{sigmoid, tanh, relu, leaky_relu, softmax},
    math::{FloatMath, exp_approx, ln_approx, sqrt_approx, sin_approx, cos_approx, safe_divide},
    linalg::{dot_product, matrix_multiply, vector_add, vector_subtract, vector_scale},
    stats::{mean, variance, standard_deviation, correlation},
};
use std::f32::consts::PI;

const EPSILON: f32 = 1e-4;

/// Test basic vector operations
#[test]
fn test_vector_operations() {
    let v1 = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let v2 = Vector::from_slice(&[4.0, 5.0, 6.0]);
    
    // Test vector addition
    let sum = vector_add(&v1, &v2);
    assert_eq!(sum.data(), &[5.0, 7.0, 9.0]);
    
    // Test vector subtraction
    let diff = vector_subtract(&v2, &v1);
    assert_eq!(diff.data(), &[3.0, 3.0, 3.0]);
    
    // Test vector scaling
    let scaled = vector_scale(&v1, 2.0);
    assert_eq!(scaled.data(), &[2.0, 4.0, 6.0]);
    
    // Test dot product
    let dot = dot_product(&v1, &v2);
    assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

/// Test vector creation and basic properties
#[test]
fn test_vector_creation() {
    // Test zeros
    let zeros = Vector::zeros(5);
    assert_eq!(zeros.len(), 5);
    assert!(zeros.data().iter().all(|&x| x == 0.0));
    
    // Test ones
    let ones = Vector::ones(3);
    assert_eq!(ones.len(), 3);
    assert!(ones.data().iter().all(|&x| x == 1.0));
    
    // Test filled
    let filled = Vector::filled(4, 2.5);
    assert_eq!(filled.len(), 4);
    assert!(filled.data().iter().all(|&x| x == 2.5));
    
    // Test from slice
    let from_slice = Vector::from_slice(&[1.0, 2.0, 3.0]);
    assert_eq!(from_slice.len(), 3);
    assert_eq!(from_slice.data(), &[1.0, 2.0, 3.0]);
}

/// Test matrix operations
#[test]
fn test_matrix_operations() {
    let m1 = Matrix::from_rows(&[
        &[1.0, 2.0],
        &[3.0, 4.0],
