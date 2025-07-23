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
    ]);
    
    let m2 = Matrix::from_rows(&[
        &[5.0, 6.0],
        &[7.0, 8.0],
    ]);
    
    // Test matrix multiplication
    let product = matrix_multiply(&m1, &m2);
    // [1 2] [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
    // [3 4] [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
    assert_eq!(product.rows(), 2);
    assert_eq!(product.cols(), 2);
    assert!((product.get(0, 0) - 19.0).abs() < EPSILON);
    assert!((product.get(0, 1) - 22.0).abs() < EPSILON);
    assert!((product.get(1, 0) - 43.0).abs() < EPSILON);
    assert!((product.get(1, 1) - 50.0).abs() < EPSILON);
}

/// Test matrix creation and properties
#[test]
fn test_matrix_creation() {
    // Test zeros
    let zeros = Matrix::zeros(3, 4);
    assert_eq!(zeros.rows(), 3);
    assert_eq!(zeros.cols(), 4);
    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(zeros.get(i, j), 0.0);
        }
    }
    
    // Test identity
    let identity = Matrix::identity(3);
    assert_eq!(identity.rows(), 3);
    assert_eq!(identity.cols(), 3);
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                assert_eq!(identity.get(i, j), 1.0);
            } else {
                assert_eq!(identity.get(i, j), 0.0);
            }
        }
    }
    
    // Test filled
    let filled = Matrix::filled(2, 3, 5.0);
    assert_eq!(filled.rows(), 2);
    assert_eq!(filled.cols(), 3);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(filled.get(i, j), 5.0);
        }
    }
}

/// Test sparse matrix operations
#[test]
fn test_sparse_matrix() {
    let mut sparse = SparseMatrix::new(3, 3);
    
    // Set some values
    sparse.set(0, 0, 1.0);
    sparse.set(1, 2, 2.0);
    sparse.set(2, 1, 3.0);
    
    // Test get
    assert_eq!(sparse.get(0, 0), 1.0);
    assert_eq!(sparse.get(1, 2), 2.0);
    assert_eq!(sparse.get(2, 1), 3.0);
    assert_eq!(sparse.get(0, 1), 0.0); // Default value
    
    // Test nnz (number of non-zeros)
    assert_eq!(sparse.nnz(), 3);
    
    // Test with capacity
    let sparse_with_cap = SparseMatrix::with_capacity(4, 4, 10);
    assert_eq!(sparse_with_cap.rows(), 4);
    assert_eq!(sparse_with_cap.cols(), 4);
    assert_eq!(sparse_with_cap.nnz(), 0);
}

/// Test activation functions
#[test]
fn test_activation_functions() {
    let x = 0.5;
    
    // Test sigmoid
    let sig = sigmoid(x);
    let expected_sig = 1.0 / (1.0 + (-x).exp());
    assert!((sig - expected_sig).abs() < EPSILON);
    
    // Test tanh
    let tanh_val = tanh(x);
    let expected_tanh = x.tanh();
    assert!((tanh_val - expected_tanh).abs() < EPSILON);
    
    // Test ReLU
    assert_eq!(relu(2.0), 2.0);
    assert_eq!(relu(-1.0), 0.0);
    assert_eq!(relu(0.0), 0.0);
    
    // Test Leaky ReLU
    assert_eq!(leaky_relu(2.0, 0.1), 2.0);
    assert!((leaky_relu(-1.0, 0.1) - (-0.1)).abs() < EPSILON);
    assert_eq!(leaky_relu(0.0, 0.1), 0.0);
}

/// Test softmax function
#[test]
fn test_softmax() {
    let input = vec![1.0, 2.0, 3.0];
    let result = softmax(&input);
    
    // Softmax should sum to 1
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < EPSILON);
    
    // All values should be positive
    assert!(result.iter().all(|&x| x > 0.0));
    
    // Larger inputs should have larger outputs
    assert!(result[2] > result[1]);
    assert!(result[1] > result[0]);
}

/// Test math approximation functions
#[test]
fn test_math_approximations() {
    let x = 2.0;
    
    // Test exp approximation
    let exp_approx_val = exp_approx(x);
    let exp_actual = x.exp();
    let exp_error = (exp_approx_val - exp_actual).abs() / exp_actual;
    assert!(exp_error < 0.1, "exp approximation error too large: {}", exp_error);
    
    // Test ln approximation
    let ln_approx_val = ln_approx(x);
    let ln_actual = x.ln();
    let ln_error = (ln_approx_val - ln_actual).abs() / ln_actual.abs();
    assert!(ln_error < 0.1, "ln approximation error too large: {}", ln_error);
    
    // Test sqrt approximation
    let sqrt_approx_val = sqrt_approx(x);
    let sqrt_actual = x.sqrt();
    let sqrt_error = (sqrt_approx_val - sqrt_actual).abs() / sqrt_actual;
    assert!(sqrt_error < 0.01, "sqrt approximation error too large: {}", sqrt_error);
}

/// Test trigonometric approximations
#[test]
fn test_trig_approximations() {
    let angles = [0.0, PI/6.0, PI/4.0, PI/3.0, PI/2.0];
    
    for &angle in &angles {
        // Test sin approximation
        let sin_approx_val = sin_approx(angle);
        let sin_actual = angle.sin();
        let sin_error = (sin_approx_val - sin_actual).abs();
        assert!(sin_error < 0.01, "sin approximation error too large at {}: {}", angle, sin_error);
        
        // Test cos approximation
        let cos_approx_val = cos_approx(angle);
        let cos_actual = angle.cos();
        let cos_error = (cos_approx_val - cos_actual).abs();
        assert!(cos_error < 0.01, "cos approximation error too large at {}: {}", angle, cos_error);
    }
}

/// Test FloatMath trait implementation
#[test]
fn test_float_math_trait() {
    let x = 2.5f32;
    
    // Test trait methods
    assert!((x.exp_approx() - exp_approx(x)).abs() < EPSILON);
    assert!((x.ln_approx() - ln_approx(x)).abs() < EPSILON);
    assert!((x.sqrt_approx() - sqrt_approx(x)).abs() < EPSILON);
    assert!((x.sin_approx() - sin_approx(x)).abs() < EPSILON);
    assert!((x.cos_approx() - cos_approx(x)).abs() < EPSILON);
}

/// Test safe division
#[test]
fn test_safe_division() {
    // Normal division
    assert_eq!(safe_divide(10.0, 2.0), 5.0);
    
    // Division by zero
    assert_eq!(safe_divide(10.0, 0.0), 0.0);
    
    // Division by very small number
    assert_eq!(safe_divide(10.0, 1e-10), 0.0);
    
    // Normal small result
    let result = safe_divide(1.0, 100.0);
    assert!((result - 0.01).abs() < EPSILON);
}

/// Test statistical functions
#[test]
fn test_statistics() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    // Test mean
    let mean_val = mean(&data);
    assert!((mean_val - 3.0).abs() < EPSILON);
    
    // Test variance
    let var_val = variance(&data);
    let expected_var = 2.0; // For this dataset
    assert!((var_val - expected_var).abs() < EPSILON);
    
    // Test standard deviation
    let std_val = standard_deviation(&data);
    let expected_std = expected_var.sqrt();
    assert!((std_val - expected_std).abs() < EPSILON);
}

/// Test correlation
#[test]
fn test_correlation() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation
    
    let corr = correlation(&x, &y);
    assert!((corr - 1.0).abs() < EPSILON, "Perfect positive correlation should be 1.0, got {}", corr);
    
    // Test negative correlation
    let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
    let corr_neg = correlation(&x, &y_neg);
    assert!((corr_neg - (-1.0)).abs() < EPSILON, "Perfect negative correlation should be -1.0, got {}", corr_neg);
}

/// Test vectorized operations
#[test]
fn test_vectorized_operations() {
    let input = vec![0.0, 0.5, 1.0, -0.5, -1.0];
    
    // Test vectorized sigmoid
    let sigmoid_results: Vec<f32> = input.iter().map(|&x| sigmoid(x)).collect();
    assert_eq!(sigmoid_results.len(), input.len());
    assert!(sigmoid_results.iter().all(|&x| x >= 0.0 && x <= 1.0));
    
    // Test vectorized ReLU
    let relu_results: Vec<f32> = input.iter().map(|&x| relu(x)).collect();
    assert_eq!(relu_results, vec![0.0, 0.5, 1.0, 0.0, 0.0]);
}

/// Test edge cases and numerical stability
#[test]
fn test_edge_cases() {
    // Test with very large numbers
    let large = 1e10;
    assert!(sigmoid(large) > 0.99);
    assert!(sigmoid(-large) < 0.01);
    
    // Test with very small numbers
    let small = 1e-10;
    assert!((sigmoid(small) - 0.5).abs() < 0.1);
    
    // Test with infinity and NaN handling
    assert!(!sigmoid(f32::INFINITY).is_nan());
    assert!(!sigmoid(f32::NEG_INFINITY).is_nan());
    
    // Test zero cases
    assert_eq!(relu(0.0), 0.0);
    assert!((sigmoid(0.0) - 0.5).abs() < EPSILON);
    assert!((tanh(0.0) - 0.0).abs() < EPSILON);
}

/// Test performance of approximation functions
#[test]
fn test_approximation_performance() {
    let test_values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
    
    let start = std::time::Instant::now();
    for &x in &test_values {
        let _ = exp_approx(x);
    }
    let approx_time = start.elapsed();
    
    let start = std::time::Instant::now();
    for &x in &test_values {
        let _ = x.exp();
    }
    let std_time = start.elapsed();
    
    println!("Approximation time: {:?}, Standard time: {:?}", approx_time, std_time);
    // Approximation should be faster or at least competitive
    assert!(approx_time <= std_time * 2); // Allow some tolerance
}

/// Test matrix-vector multiplication
#[test]
fn test_matrix_vector_multiply() {
    let matrix = Matrix::from_rows(&[
        &[1.0, 2.0, 3.0],
        &[4.0, 5.0, 6.0],
    ]);
    
    let vector = Vector::from_slice(&[1.0, 2.0, 3.0]);
    
    // Matrix-vector multiplication: [2x3] * [3x1] = [2x1]
    let result = matrix.multiply_vector(&vector);
    
    // [1 2 3] [1]   [1*1 + 2*2 + 3*3]   [14]
    // [4 5 6] [2] = [4*1 + 5*2 + 6*3] = [32]
    //         [3]
    
    assert_eq!(result.len(), 2);
    assert!((result.data()[0] - 14.0).abs() < EPSILON);
    assert!((result.data()[1] - 32.0).abs() < EPSILON);
}

/// Test sparse matrix operations
#[test]
fn test_sparse_matrix_operations() {
    let mut sparse = SparseMatrix::new(1000, 1000);
    
    // Set some scattered values
    sparse.set(0, 999, 1.0);
    sparse.set(500, 500, 2.0);
    sparse.set(999, 0, 3.0);
    
    assert_eq!(sparse.nnz(), 3);
    assert_eq!(sparse.get(0, 999), 1.0);
    assert_eq!(sparse.get(500, 500), 2.0);
    assert_eq!(sparse.get(999, 0), 3.0);
    assert_eq!(sparse.get(100, 100), 0.0);
}

/// Test numerical precision and accuracy
#[test]
fn test_numerical_precision() {
    // Test that our implementations maintain reasonable precision
    let x = 1.0f32;
    let iterations = 1000;
    
    // Accumulate using our safe operations
    let mut sum = 0.0f32;
    for _ in 0..iterations {
        sum = safe_divide(sum + x, 1.0);
    }
    
    assert!((sum - iterations as f32).abs() < 1.0);
}

/// Test thread safety of math functions
#[test]
fn test_thread_safety() {
    use std::thread;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    
    let counter = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::new();
    
    for i in 0..4 {
        let counter_clone = counter.clone();
        let handle = thread::spawn(move || {
            for j in 0..1000 {
                let x = (i * 1000 + j) as f32 * 0.001;
                let _result = sigmoid(x) + tanh(x) + relu(x);
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    assert_eq!(counter.load(Ordering::SeqCst), 4000);
}