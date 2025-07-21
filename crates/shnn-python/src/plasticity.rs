//! Python bindings for plasticity rules and learning mechanisms
//!
//! This module provides Python interfaces for spike-timing dependent plasticity (STDP),
//! homeostatic mechanisms, and other learning rules in spiking neural networks.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::{PyRuntimeError, PyValueError};

use shnn_core::{
    plasticity::{
        STDPRule, STDPType, PlasticityRule, HomeostaticRule,
        LearningWindow, WeightUpdate, PlasticityState,
        BCMRule, OjaRule, HebbianRule,
    },
    spike::{Spike, SpikeTime},
    time::TimeStep,
};

use crate::neuron::PySpike;
use crate::error_conversion::core_error_to_py_err;

/// Python wrapper for STDP rule
#[pyclass(name = "STDPRule")]
pub struct PySTDPRule {
    rule: STDPRule,
}

#[pymethods]
impl PySTDPRule {
    #[new]
    #[pyo3(signature = (
        a_plus=0.01,
        a_minus=0.01,
        tau_plus=20.0,
        tau_minus=20.0,
        stdp_type="additive",
        w_min=0.0,
        w_max=1.0,
        **kwargs
    ))]
    fn new(
        a_plus: f32,
        a_minus: f32,
        tau_plus: f32,
        tau_minus: f32,
        stdp_type: &str,
        w_min: f32,
        w_max: f32,
        kwargs: Option<&PyDict>,
    ) -> PyResult<Self> {
        if a_plus <= 0.0 || a_minus <= 0.0 {
            return Err(PyValueError::new_err("Learning rates must be positive"));
        }
        if tau_plus <= 0.0 || tau_minus <= 0.0 {
            return Err(PyValueError::new_err("Time constants must be positive"));
        }
        if w_min >= w_max {
            return Err(PyValueError::new_err("w_min must be less than w_max"));
        }
        
        let rule_type = match stdp_type.to_lowercase().as_str() {
            "additive" => STDPType::Additive,
            "multiplicative" => STDPType::Multiplicative,
            "mixed" => STDPType::Mixed,
            _ => return Err(PyValueError::new_err(format!("Unknown STDP type: {}", stdp_type))),
        };
        
        let mut rule = STDPRule::new(
            a_plus,
            a_minus,
            TimeStep::from_secs_f64(tau_plus as f64 / 1000.0), // Convert ms to seconds
            TimeStep::from_secs_f64(tau_minus as f64 / 1000.0),
            rule_type,
            w_min,
            w_max,
        ).map_err(core_error_to_py_err)?;
        
        // Process additional parameters
        if let Some(kwargs) = kwargs {
            if let Some(mu_plus) = kwargs.get_item("mu_plus")? {
                rule.set_mu_plus(mu_plus.extract()?);
            }
            if let Some(mu_minus) = kwargs.get_item("mu_minus")? {
                rule.set_mu_minus(mu_minus.extract()?);
            }
            if let Some(alpha) = kwargs.get_item("alpha")? {
                rule.set_alpha(alpha.extract()?);
            }
        }
        
        Ok(Self { rule })
    }
    
    /// Create standard additive STDP rule
    #[classmethod]
    fn additive(_cls: &PyType, a_plus: Option<f32>, a_minus: Option<f32>) -> PyResult<Self> {
        let rule = STDPRule::additive(
            a_plus.unwrap_or(0.01),
            a_minus.unwrap_or(0.01),
        ).map_err(core_error_to_py_err)?;
        
        Ok(Self { rule })
    }
    
    /// Create multiplicative STDP rule
    #[classmethod]
    fn multiplicative(_cls: &PyType, a_plus: Option<f32>, a_minus: Option<f32>) -> PyResult<Self> {
        let rule = STDPRule::multiplicative(
            a_plus.unwrap_or(0.01),
            a_minus.unwrap_or(0.01),
        ).map_err(core_error_to_py_err)?;
        
        Ok(Self { rule })
    }
    
    /// Create triplet STDP rule
    #[classmethod]
    fn triplet(_cls: &PyType, a2_plus: Option<f32>, a2_minus: Option<f32>, a3_plus: Option<f32>) -> PyResult<Self> {
        let rule = STDPRule::triplet(
            a2_plus.unwrap_or(0.01),
            a2_minus.unwrap_or(0.01),
            a3_plus.unwrap_or(0.001),
        ).map_err(core_error_to_py_err)?;
        
        Ok(Self { rule })
    }
    
    /// Update weight based on spike timing
    fn update_weight(
        &self,
        current_weight: f32,
        pre_spike_time: f64,
        post_spike_time: f64,
    ) -> PyResult<f32> {
        let pre_time = SpikeTime::from_secs_f64(pre_spike_time);
        let post_time = SpikeTime::from_secs_f64(post_spike_time);
        
        self.rule.update_weight(current_weight, pre_time, post_time)
            .map_err(core_error_to_py_err)
    }
    
    /// Calculate weight change from spike pair
    fn calculate_weight_change(
        &self,
        current_weight: f32,
        delta_t: f64,
    ) -> f32 {
        let dt = TimeStep::from_secs_f64(delta_t);
        self.rule.calculate_weight_change(current_weight, dt)
    }
    
    /// Get learning window function values
    fn get_learning_window(&self, dt_range: (f64, f64), num_points: Option<u32>) -> PyResult<Vec<(f64, f32)>> {
        let (start, end) = dt_range;
        let points = num_points.unwrap_or(100);
        
        let mut window = Vec::new();
        let step = (end - start) / (points as f64);
        
        for i in 0..points {
            let dt = start + step * (i as f64);
            let dt_time = TimeStep::from_secs_f64(dt);
            let dw = self.rule.calculate_weight_change(0.5, dt_time); // Use middle weight
            window.push((dt, dw));
        }
        
        Ok(window)
    }
    
    /// Batch update weights for multiple spike pairs
    fn batch_update(
        &self,
        weights: Vec<f32>,
        spike_pairs: &PyList,
    ) -> PyResult<Vec<f32>> {
        if weights.len() != spike_pairs.len() {
            return Err(PyValueError::new_err("Weights and spike pairs must have same length"));
        }
        
        let mut updated_weights = Vec::with_capacity(weights.len());
        
        for (i, weight) in weights.iter().enumerate() {
            let pair = spike_pairs.get_item(i)?;
            let tuple = pair.downcast::<pyo3::types::PyTuple>()?;
            
            if tuple.len() != 2 {
                return Err(PyValueError::new_err("Spike pair must be (pre_time, post_time)"));
            }
            
            let pre_time: f64 = tuple.get_item(0)?.extract()?;
            let post_time: f64 = tuple.get_item(1)?.extract()?;
            
            let new_weight = self.update_weight(*weight, pre_time, post_time)?;
            updated_weights.push(new_weight);
        }
        
        Ok(updated_weights)
    }
    
    // Getters and setters
    #[getter]
    fn a_plus(&self) -> f32 {
        self.rule.get_a_plus()
    }
    
    #[setter]
    fn set_a_plus(&mut self, value: f32) -> PyResult<()> {
        if value <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive"));
        }
        self.rule.set_a_plus(value);
        Ok(())
    }
    
    #[getter]
    fn a_minus(&self) -> f32 {
        self.rule.get_a_minus()
    }
    
    #[setter]
    fn set_a_minus(&mut self, value: f32) -> PyResult<()> {
        if value <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive"));
        }
        self.rule.set_a_minus(value);
        Ok(())
    }
    
    #[getter]
    fn tau_plus(&self) -> f64 {
        self.rule.get_tau_plus().as_secs_f64() * 1000.0 // Convert to ms
    }
    
    #[setter]
    fn set_tau_plus(&mut self, value: f64) -> PyResult<()> {
        if value <= 0.0 {
            return Err(PyValueError::new_err("Time constant must be positive"));
        }
        self.rule.set_tau_plus(TimeStep::from_secs_f64(value / 1000.0));
        Ok(())
    }
    
    #[getter]
    fn tau_minus(&self) -> f64 {
        self.rule.get_tau_minus().as_secs_f64() * 1000.0 // Convert to ms
    }
    
    #[setter]
    fn set_tau_minus(&mut self, value: f64) -> PyResult<()> {
        if value <= 0.0 {
            return Err(PyValueError::new_err("Time constant must be positive"));
        }
        self.rule.set_tau_minus(TimeStep::from_secs_f64(value / 1000.0));
        Ok(())
    }
    
    #[getter]
    fn w_min(&self) -> f32 {
        self.rule.get_w_min()
    }
    
    #[setter]
    fn set_w_min(&mut self, value: f32) -> PyResult<()> {
        if value >= self.rule.get_w_max() {
            return Err(PyValueError::new_err("w_min must be less than w_max"));
        }
        self.rule.set_w_min(value);
        Ok(())
    }
    
    #[getter]
    fn w_max(&self) -> f32 {
        self.rule.get_w_max()
    }
    
    #[setter]
    fn set_w_max(&mut self, value: f32) -> PyResult<()> {
        if value <= self.rule.get_w_min() {
            return Err(PyValueError::new_err("w_max must be greater than w_min"));
        }
        self.rule.set_w_max(value);
        Ok(())
    }
    
    /// Get rule parameters as dictionary
    fn get_parameters(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("a_plus", self.rule.get_a_plus())?;
            dict.set_item("a_minus", self.rule.get_a_minus())?;
            dict.set_item("tau_plus", self.tau_plus())?;
            dict.set_item("tau_minus", self.tau_minus())?;
            dict.set_item("w_min", self.rule.get_w_min())?;
            dict.set_item("w_max", self.rule.get_w_max())?;
            dict.set_item("stdp_type", match self.rule.get_type() {
                STDPType::Additive => "additive",
                STDPType::Multiplicative => "multiplicative",
                STDPType::Mixed => "mixed",
            })?;
            Ok(dict.to_object(py))
        })
    }
    
    fn __repr__(&self) -> String {
        format!(
            "STDPRule(A+={:.3}, A-={:.3}, τ+={:.1}ms, τ-={:.1}ms)",
            self.rule.get_a_plus(),
            self.rule.get_a_minus(),
            self.tau_plus(),
            self.tau_minus()
        )
    }
}

/// Python wrapper for homeostatic plasticity rule
#[pyclass(name = "HomeostaticRule")]
pub struct PyHomeostaticRule {
    rule: HomeostaticRule,
}

#[pymethods]
impl PyHomeostaticRule {
    #[new]
    fn new(
        target_rate: f32,
        learning_rate: f32,
        time_constant: f64,
    ) -> PyResult<Self> {
        if target_rate <= 0.0 {
            return Err(PyValueError::new_err("Target rate must be positive"));
        }
        if learning_rate <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive"));
        }
        if time_constant <= 0.0 {
            return Err(PyValueError::new_err("Time constant must be positive"));
        }
        
        let tau = TimeStep::from_secs_f64(time_constant);
        let rule = HomeostaticRule::new(target_rate, learning_rate, tau)
            .map_err(core_error_to_py_err)?;
        
        Ok(Self { rule })
    }
    
    /// Update synaptic scaling based on activity
    fn update_scaling(
        &mut self,
        current_rate: f32,
        dt: f64,
    ) -> PyResult<f32> {
        let time_step = TimeStep::from_secs_f64(dt);
        self.rule.update_scaling(current_rate, time_step)
            .map_err(core_error_to_py_err)
    }
    
    /// Get current scaling factor
    fn get_scaling_factor(&self) -> f32 {
        self.rule.get_scaling_factor()
    }
    
    /// Reset scaling to initial value
    fn reset(&mut self) {
        self.rule.reset();
    }
    
    #[getter]
    fn target_rate(&self) -> f32 {
        self.rule.get_target_rate()
    }
    
    #[setter]
    fn set_target_rate(&mut self, value: f32) -> PyResult<()> {
        if value <= 0.0 {
            return Err(PyValueError::new_err("Target rate must be positive"));
        }
        self.rule.set_target_rate(value);
        Ok(())
    }
    
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.rule.get_learning_rate()
    }
    
    #[setter]
    fn set_learning_rate(&mut self, value: f32) -> PyResult<()> {
        if value <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive"));
        }
        self.rule.set_learning_rate(value);
        Ok(())
    }
    
    fn __repr__(&self) -> String {
        format!(
            "HomeostaticRule(target={:.1} Hz, lr={:.4}, scale={:.3})",
            self.rule.get_target_rate(),
            self.rule.get_learning_rate(),
            self.rule.get_scaling_factor()
        )
    }
}

/// Python wrapper for BCM (Bienenstock-Cooper-Munro) rule
#[pyclass(name = "BCMRule")]
pub struct PyBCMRule {
    rule: BCMRule,
}

#[pymethods]
impl PyBCMRule {
    #[new]
    fn new(
        learning_rate: f32,
        tau_theta: f64,
        p: Option<f32>,
    ) -> PyResult<Self> {
        if learning_rate <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive"));
        }
        if tau_theta <= 0.0 {
            return Err(PyValueError::new_err("Time constant must be positive"));
        }
        
        let tau = TimeStep::from_secs_f64(tau_theta);
        let rule = BCMRule::new(learning_rate, tau, p.unwrap_or(2.0))
            .map_err(core_error_to_py_err)?;
        
        Ok(Self { rule })
    }
    
    /// Update weight based on pre and post activities
    fn update_weight(
        &mut self,
        current_weight: f32,
        pre_activity: f32,
        post_activity: f32,
        dt: f64,
    ) -> PyResult<f32> {
        let time_step = TimeStep::from_secs_f64(dt);
        self.rule.update_weight(current_weight, pre_activity, post_activity, time_step)
            .map_err(core_error_to_py_err)
    }
    
    /// Get current threshold
    fn get_threshold(&self) -> f32 {
        self.rule.get_threshold()
    }
    
    /// Update threshold based on activity
    fn update_threshold(&mut self, post_activity: f32, dt: f64) {
        let time_step = TimeStep::from_secs_f64(dt);
        self.rule.update_threshold(post_activity, time_step);
    }
    
    fn __repr__(&self) -> String {
        format!(
            "BCMRule(lr={:.4}, threshold={:.3})",
            self.rule.get_learning_rate(),
            self.rule.get_threshold()
        )
    }
}

/// Python wrapper for Oja's rule
#[pyclass(name = "OjaRule")]
pub struct PyOjaRule {
    rule: OjaRule,
}

#[pymethods]
impl PyOjaRule {
    #[new]
    fn new(learning_rate: f32) -> PyResult<Self> {
        if learning_rate <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive"));
        }
        
        let rule = OjaRule::new(learning_rate)
            .map_err(core_error_to_py_err)?;
        
        Ok(Self { rule })
    }
    
    /// Update weight based on Oja's rule
    fn update_weight(
        &self,
        current_weight: f32,
        pre_activity: f32,
        post_activity: f32,
    ) -> f32 {
        self.rule.update_weight(current_weight, pre_activity, post_activity)
    }
    
    /// Batch update for multiple weights
    fn batch_update(
        &self,
        weights: Vec<f32>,
        pre_activities: Vec<f32>,
        post_activity: f32,
    ) -> PyResult<Vec<f32>> {
        if weights.len() != pre_activities.len() {
            return Err(PyValueError::new_err("Weights and pre_activities must have same length"));
        }
        
        Ok(weights.iter()
            .zip(pre_activities.iter())
            .map(|(&w, &pre)| self.rule.update_weight(w, pre, post_activity))
            .collect())
    }
    
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.rule.get_learning_rate()
    }
    
    #[setter]
    fn set_learning_rate(&mut self, value: f32) -> PyResult<()> {
        if value <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive"));
        }
        self.rule.set_learning_rate(value);
        Ok(())
    }
    
    fn __repr__(&self) -> String {
        format!("OjaRule(lr={:.4})", self.rule.get_learning_rate())
    }
}

/// Python wrapper for Hebbian rule
#[pyclass(name = "HebbianRule")]
pub struct PyHebbianRule {
    rule: HebbianRule,
}

#[pymethods]
impl PyHebbianRule {
    #[new]
    fn new(learning_rate: f32, decay_rate: Option<f32>) -> PyResult<Self> {
        if learning_rate <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive"));
        }
        
        let rule = HebbianRule::new(learning_rate, decay_rate.unwrap_or(0.0))
            .map_err(core_error_to_py_err)?;
        
        Ok(Self { rule })
    }
    
    /// Update weight based on Hebbian rule
    fn update_weight(
        &self,
        current_weight: f32,
        pre_activity: f32,
        post_activity: f32,
        dt: f64,
    ) -> f32 {
        let time_step = TimeStep::from_secs_f64(dt);
        self.rule.update_weight(current_weight, pre_activity, post_activity, time_step)
    }
    
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.rule.get_learning_rate()
    }
    
    #[setter]
    fn set_learning_rate(&mut self, value: f32) -> PyResult<()> {
        if value <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive"));
        }
        self.rule.set_learning_rate(value);
        Ok(())
    }
    
    #[getter]
    fn decay_rate(&self) -> f32 {
        self.rule.get_decay_rate()
    }
    
    #[setter]
    fn set_decay_rate(&mut self, value: f32) -> PyResult<()> {
        if value < 0.0 {
            return Err(PyValueError::new_err("Decay rate must be non-negative"));
        }
        self.rule.set_decay_rate(value);
        Ok(())
    }
    
    fn __repr__(&self) -> String {
        format!(
            "HebbianRule(lr={:.4}, decay={:.4})",
            self.rule.get_learning_rate(),
            self.rule.get_decay_rate()
        )
    }
}

/// Utility functions for plasticity analysis
#[pyfunction]
pub fn plot_stdp_window(
    stdp_rule: &PySTDPRule,
    dt_range: (f64, f64),
    num_points: Option<u32>,
) -> PyResult<(Vec<f64>, Vec<f32>)> {
    let window = stdp_rule.get_learning_window(dt_range, num_points)?;
    let (times, weights): (Vec<_>, Vec<_>) = window.into_iter().unzip();
    Ok((times, weights))
}

#[pyfunction]
pub fn simulate_weight_evolution(
    initial_weight: f32,
    spike_times_pre: Vec<f64>,
    spike_times_post: Vec<f64>,
    stdp_rule: &PySTDPRule,
) -> PyResult<Vec<f32>> {
    let mut weight = initial_weight;
    let mut weights = vec![weight];
    
    // Combine and sort all spike times
    let mut all_events: Vec<(f64, bool)> = Vec::new();
    for time in spike_times_pre {
        all_events.push((time, true)); // true for pre-synaptic
    }
    for time in spike_times_post {
        all_events.push((time, false)); // false for post-synaptic
    }
    all_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    let mut last_pre_time: Option<f64> = None;
    let mut last_post_time: Option<f64> = None;
    
    for (time, is_pre) in all_events {
        if is_pre {
            last_pre_time = Some(time);
            // Check for post-before-pre
            if let Some(post_time) = last_post_time {
                if post_time < time {
                    weight = stdp_rule.update_weight(weight, time, post_time)?;
                    weights.push(weight);
                }
            }
        } else {
            last_post_time = Some(time);
            // Check for pre-before-post
            if let Some(pre_time) = last_pre_time {
                if pre_time < time {
                    weight = stdp_rule.update_weight(weight, pre_time, time)?;
                    weights.push(weight);
                }
            }
        }
    }
    
    Ok(weights)
}

#[pyfunction]
pub fn calculate_weight_distribution(
    weights: Vec<f32>,
    num_bins: Option<u32>,
) -> PyResult<(Vec<f32>, Vec<u32>)> {
    if weights.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    
    let bins = num_bins.unwrap_or(20);
    let min_weight = weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_weight = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    if min_weight == max_weight {
        return Ok((vec![min_weight], vec![weights.len() as u32]));
    }
    
    let bin_width = (max_weight - min_weight) / bins as f32;
    let mut bin_edges = Vec::with_capacity(bins as usize + 1);
    let mut bin_counts = vec![0u32; bins as usize];
    
    // Create bin edges
    for i in 0..=bins {
        bin_edges.push(min_weight + i as f32 * bin_width);
    }
    
    // Count weights in each bin
    for &weight in &weights {
        let bin_idx = ((weight - min_weight) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(bins as usize - 1);
        bin_counts[bin_idx] += 1;
    }
    
    // Return bin centers and counts
    let bin_centers: Vec<f32> = (0..bins)
        .map(|i| min_weight + (i as f32 + 0.5) * bin_width)
        .collect();
    
    Ok((bin_centers, bin_counts))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stdp_rule() {
        let mut stdp = PySTDPRule::new(
            0.01, 0.01, 20.0, 20.0, "additive", 0.0, 1.0, None
        ).unwrap();
        
        assert_eq!(stdp.a_plus(), 0.01);
        assert_eq!(stdp.a_minus(), 0.01);
        assert_eq!(stdp.tau_plus(), 20.0);
        assert_eq!(stdp.tau_minus(), 20.0);
        assert_eq!(stdp.w_min(), 0.0);
        assert_eq!(stdp.w_max(), 1.0);
        
        // Test parameter updates
        stdp.set_a_plus(0.02).unwrap();
        assert_eq!(stdp.a_plus(), 0.02);
        
        // Test invalid parameters
        assert!(stdp.set_a_plus(-0.01).is_err());
        assert!(stdp.set_tau_plus(-10.0).is_err());
        
        // Test weight update
        let weight = 0.5;
        let pre_time = 0.001;
        let post_time = 0.002; // Post after pre -> potentiation
        let new_weight = stdp.update_weight(weight, pre_time, post_time).unwrap();
        assert!(new_weight > weight); // Should be potentiated
        
        // Test depression (pre after post)
        let new_weight2 = stdp.update_weight(weight, post_time, pre_time).unwrap();
        assert!(new_weight2 < weight); // Should be depressed
    }
    
    #[test]
    fn test_homeostatic_rule() {
        let mut homeo = PyHomeostaticRule::new(10.0, 0.001, 1.0).unwrap();
        
        assert_eq!(homeo.target_rate(), 10.0);
        assert_eq!(homeo.learning_rate(), 0.001);
        assert_eq!(homeo.get_scaling_factor(), 1.0);
        
        // Test scaling update with low activity
        let scaling = homeo.update_scaling(5.0, 0.1).unwrap(); // Below target
        assert!(scaling > 1.0); // Should increase scaling
        
        // Test parameter updates
        homeo.set_target_rate(15.0).unwrap();
        assert_eq!(homeo.target_rate(), 15.0);
        
        // Test invalid parameters
        assert!(homeo.set_target_rate(-5.0).is_err());
        assert!(homeo.set_learning_rate(-0.001).is_err());
        
        // Test reset
        homeo.reset();
        assert_eq!(homeo.get_scaling_factor(), 1.0);
    }
    
    #[test]
    fn test_bcm_rule() {
        let mut bcm = PyBCMRule::new(0.001, 1.0, Some(2.0)).unwrap();
        
        let initial_threshold = bcm.get_threshold();
        
        // Test weight update
        let weight = 0.5;
        let pre_activity = 1.0;
        let post_activity = 2.0;
        let new_weight = bcm.update_weight(weight, pre_activity, post_activity, 0.001).unwrap();
        
        // Test threshold update
        bcm.update_threshold(post_activity, 0.001);
        let new_threshold = bcm.get_threshold();
        assert!(new_threshold != initial_threshold);
        
        // Test invalid parameters
        assert!(PyBCMRule::new(-0.001, 1.0, None).is_err());
        assert!(PyBCMRule::new(0.001, -1.0, None).is_err());
    }
    
    #[test]
    fn test_oja_rule() {
        let mut oja = PyOjaRule::new(0.01).unwrap();
        
        assert_eq!(oja.learning_rate(), 0.01);
        
        // Test weight update
        let weight = 0.5;
        let pre_activity = 1.0;
        let post_activity = 0.8;
        let new_weight = oja.update_weight(weight, pre_activity, post_activity);
        
        // Test batch update
        let weights = vec![0.1, 0.5, 0.9];
        let pre_activities = vec![1.0, 0.5, 0.2];
        let new_weights = oja.batch_update(weights.clone(), pre_activities, 0.7).unwrap();
        assert_eq!(new_weights.len(), weights.len());
        
        // Test parameter update
        oja.set_learning_rate(0.02).unwrap();
        assert_eq!(oja.learning_rate(), 0.02);
        
        // Test invalid parameters
        assert!(oja.set_learning_rate(-0.01).is_err());
        
        // Test invalid batch update
        let short_pre = vec![1.0];
        assert!(oja.batch_update(weights, short_pre, 0.7).is_err());
    }
    
    #[test]
    fn test_hebbian_rule() {
        let mut hebb = PyHebbianRule::new(0.01, Some(0.001)).unwrap();
        
        assert_eq!(hebb.learning_rate(), 0.01);
        assert_eq!(hebb.decay_rate(), 0.001);
        
        // Test weight update
        let weight = 0.5;
        let pre_activity = 1.0;
        let post_activity = 0.8;
        let new_weight = hebb.update_weight(weight, pre_activity, post_activity, 0.001);
        
        // Test parameter updates
        hebb.set_learning_rate(0.02).unwrap();
        hebb.set_decay_rate(0.002).unwrap();
        assert_eq!(hebb.learning_rate(), 0.02);
        assert_eq!(hebb.decay_rate(), 0.002);
        
        // Test invalid parameters
        assert!(hebb.set_learning_rate(-0.01).is_err());
        assert!(hebb.set_decay_rate(-0.001).is_err());
    }
    
    #[test]
    fn test_utility_functions() {
        let stdp = PySTDPRule::additive(Some(0.01), Some(0.01)).unwrap();
        
        // Test STDP window plotting
        let (times, weights) = plot_stdp_window(&stdp, (-0.1, 0.1), Some(20)).unwrap();
        assert_eq!(times.len(), 20);
        assert_eq!(weights.len(), 20);
        
        // Test weight evolution simulation
        let spike_times_pre = vec![0.001, 0.003, 0.005];
        let spike_times_post = vec![0.002, 0.004];
        let weights = simulate_weight_evolution(0.5, spike_times_pre, spike_times_post, &stdp).unwrap();
        assert!(!weights.is_empty());
        
        // Test weight distribution calculation
        let test_weights = vec![0.1, 0.2, 0.2, 0.3, 0.4, 0.4, 0.4, 0.5];
        let (bin_centers, bin_counts) = calculate_weight_distribution(test_weights, Some(5)).unwrap();
        assert_eq!(bin_centers.len(), 5);
        assert_eq!(bin_counts.len(), 5);
        assert_eq!(bin_counts.iter().sum::<u32>(), 8);
    }
}