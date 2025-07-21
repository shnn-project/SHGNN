//! Python bindings for neuron models and operations
//!
//! This module provides Python classes for different neuron types including
//! Leaky Integrate-and-Fire (LIF), Adaptive Exponential (AdEx), and Izhikevich models.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::{PyRuntimeError, PyValueError};

use shnn_core::{
    neuron::{
        NeuronType, NeuronState, SynapticInput, RefractoryState,
        LIFNeuron, AdExNeuron, IzhikevichNeuron, NeuronParameters,
    },
    spike::{Spike, SpikeTime},
    time::TimeStep,
};

use crate::error_conversion::core_error_to_py_err;

/// Python wrapper for neuron state
#[pyclass(name = "NeuronState")]
#[derive(Clone, Debug)]
pub struct PyNeuronState {
    pub state: NeuronState,
}

#[pymethods]
impl PyNeuronState {
    #[new]
    fn new(membrane_potential: f32, recovery_variable: Option<f32>) -> Self {
        Self {
            state: NeuronState {
                membrane_potential,
                recovery_variable: recovery_variable.unwrap_or(0.0),
                last_spike_time: None,
                refractory_state: RefractoryState::Ready,
                adaptation_current: 0.0,
                calcium_concentration: 0.0,
            },
        }
    }
    
    #[getter]
    fn membrane_potential(&self) -> f32 {
        self.state.membrane_potential
    }
    
    #[setter]
    fn set_membrane_potential(&mut self, value: f32) {
        self.state.membrane_potential = value;
    }
    
    #[getter]
    fn recovery_variable(&self) -> f32 {
        self.state.recovery_variable
    }
    
    #[setter]
    fn set_recovery_variable(&mut self, value: f32) {
        self.state.recovery_variable = value;
    }
    
    #[getter]
    fn last_spike_time(&self) -> Option<f64> {
        self.state.last_spike_time.map(|t| t.as_secs_f64())
    }
    
    #[getter]
    fn adaptation_current(&self) -> f32 {
        self.state.adaptation_current
    }
    
    #[setter]
    fn set_adaptation_current(&mut self, value: f32) {
        self.state.adaptation_current = value;
    }
    
    #[getter]
    fn calcium_concentration(&self) -> f32 {
        self.state.calcium_concentration
    }
    
    #[setter]
    fn set_calcium_concentration(&mut self, value: f32) {
        self.state.calcium_concentration = value;
    }
    
    #[getter]
    fn is_refractory(&self) -> bool {
        matches!(self.state.refractory_state, RefractoryState::Refractory(_))
    }
    
    /// Reset neuron to resting state
    fn reset(&mut self) {
        self.state.membrane_potential = -70.0; // Typical resting potential
        self.state.recovery_variable = 0.0;
        self.state.last_spike_time = None;
        self.state.refractory_state = RefractoryState::Ready;
        self.state.adaptation_current = 0.0;
        self.state.calcium_concentration = 0.0;
    }
    
    /// Clone the state
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "NeuronState(V={:.2}mV, u={:.2}, refractory={})",
            self.state.membrane_potential,
            self.state.recovery_variable,
            self.is_refractory()
        )
    }
}

/// Python wrapper for neuron parameters
#[pyclass(name = "NeuronParameters")]
#[derive(Clone, Debug)]
pub struct PyNeuronParameters {
    pub params: NeuronParameters,
}

#[pymethods]
impl PyNeuronParameters {
    #[new]
    #[pyo3(signature = (neuron_type="LIF", **kwargs))]
    fn new(neuron_type: &str, kwargs: Option<&PyDict>) -> PyResult<Self> {
        let mut params = match neuron_type.to_uppercase().as_str() {
            "LIF" => NeuronParameters::lif_default(),
            "ADEX" => NeuronParameters::adex_default(),
            "IZHIKEVICH" | "IZH" => NeuronParameters::izhikevich_default(),
            _ => return Err(PyValueError::new_err(format!("Unknown neuron type: {}", neuron_type))),
        };
        
        // Process keyword arguments
        if let Some(kwargs) = kwargs {
            if let Some(tau_m) = kwargs.get_item("tau_m")? {
                params.tau_m = tau_m.extract()?;
            }
            if let Some(tau_s) = kwargs.get_item("tau_s")? {
                params.tau_s = tau_s.extract()?;
            }
            if let Some(v_threshold) = kwargs.get_item("v_threshold")? {
                params.v_threshold = v_threshold.extract()?;
            }
            if let Some(v_reset) = kwargs.get_item("v_reset")? {
                params.v_reset = v_reset.extract()?;
            }
            if let Some(v_rest) = kwargs.get_item("v_rest")? {
                params.v_rest = v_rest.extract()?;
            }
            if let Some(refractory_period) = kwargs.get_item("refractory_period")? {
                params.refractory_period = refractory_period.extract()?;
            }
            if let Some(resistance) = kwargs.get_item("resistance")? {
                params.resistance = resistance.extract()?;
            }
            if let Some(capacitance) = kwargs.get_item("capacitance")? {
                params.capacitance = capacitance.extract()?;
            }
            
            // AdEx specific parameters
            if let Some(delta_t) = kwargs.get_item("delta_t")? {
                params.delta_t = Some(delta_t.extract()?);
            }
            if let Some(v_spike) = kwargs.get_item("v_spike")? {
                params.v_spike = Some(v_spike.extract()?);
            }
            if let Some(tau_w) = kwargs.get_item("tau_w")? {
                params.tau_w = Some(tau_w.extract()?);
            }
            if let Some(a) = kwargs.get_item("a")? {
                params.a = Some(a.extract()?);
            }
            if let Some(b) = kwargs.get_item("b")? {
                params.b = Some(b.extract()?);
            }
            
            // Izhikevich specific parameters
            if let Some(izh_a) = kwargs.get_item("izh_a")? {
                params.izh_a = Some(izh_a.extract()?);
            }
            if let Some(izh_b) = kwargs.get_item("izh_b")? {
                params.izh_b = Some(izh_b.extract()?);
            }
            if let Some(izh_c) = kwargs.get_item("izh_c")? {
                params.izh_c = Some(izh_c.extract()?);
            }
            if let Some(izh_d) = kwargs.get_item("izh_d")? {
                params.izh_d = Some(izh_d.extract()?);
            }
        }
        
        Ok(Self { params })
    }
    
    /// Create LIF neuron parameters
    #[classmethod]
    fn lif(_cls: &PyType, tau_m: Option<f32>, v_threshold: Option<f32>, v_reset: Option<f32>) -> Self {
        let mut params = NeuronParameters::lif_default();
        if let Some(tau_m) = tau_m {
            params.tau_m = tau_m;
        }
        if let Some(v_threshold) = v_threshold {
            params.v_threshold = v_threshold;
        }
        if let Some(v_reset) = v_reset {
            params.v_reset = v_reset;
        }
        Self { params }
    }
    
    /// Create AdEx neuron parameters
    #[classmethod]
    fn adex(
        _cls: &PyType,
        tau_m: Option<f32>,
        delta_t: Option<f32>,
        v_spike: Option<f32>,
        tau_w: Option<f32>,
    ) -> Self {
        let mut params = NeuronParameters::adex_default();
        if let Some(tau_m) = tau_m {
            params.tau_m = tau_m;
        }
        if let Some(delta_t) = delta_t {
            params.delta_t = Some(delta_t);
        }
        if let Some(v_spike) = v_spike {
            params.v_spike = Some(v_spike);
        }
        if let Some(tau_w) = tau_w {
            params.tau_w = Some(tau_w);
        }
        Self { params }
    }
    
    /// Create Izhikevich neuron parameters
    #[classmethod]
    fn izhikevich(_cls: &PyType, a: Option<f32>, b: Option<f32>, c: Option<f32>, d: Option<f32>) -> Self {
        let mut params = NeuronParameters::izhikevich_default();
        if let Some(a) = a {
            params.izh_a = Some(a);
        }
        if let Some(b) = b {
            params.izh_b = Some(b);
        }
        if let Some(c) = c {
            params.izh_c = Some(c);
        }
        if let Some(d) = d {
            params.izh_d = Some(d);
        }
        Self { params }
    }
    
    /// Create regular spiking neuron (Izhikevich)
    #[classmethod]
    fn regular_spiking(_cls: &PyType) -> Self {
        Self {
            params: NeuronParameters::izhikevich_regular_spiking(),
        }
    }
    
    /// Create fast spiking neuron (Izhikevich)
    #[classmethod]
    fn fast_spiking(_cls: &PyType) -> Self {
        Self {
            params: NeuronParameters::izhikevich_fast_spiking(),
        }
    }
    
    /// Create chattering neuron (Izhikevich)
    #[classmethod]
    fn chattering(_cls: &PyType) -> Self {
        Self {
            params: NeuronParameters::izhikevich_chattering(),
        }
    }
    
    /// Create bursting neuron (Izhikevich)
    #[classmethod]
    fn bursting(_cls: &PyType) -> Self {
        Self {
            params: NeuronParameters::izhikevich_bursting(),
        }
    }
    
    // Getters and setters for all parameters
    #[getter]
    fn tau_m(&self) -> f32 {
        self.params.tau_m
    }
    
    #[setter]
    fn set_tau_m(&mut self, value: f32) -> PyResult<()> {
        if value <= 0.0 {
            return Err(PyValueError::new_err("Membrane time constant must be positive"));
        }
        self.params.tau_m = value;
        Ok(())
    }
    
    #[getter]
    fn v_threshold(&self) -> f32 {
        self.params.v_threshold
    }
    
    #[setter]
    fn set_v_threshold(&mut self, value: f32) {
        self.params.v_threshold = value;
    }
    
    #[getter]
    fn v_reset(&self) -> f32 {
        self.params.v_reset
    }
    
    #[setter]
    fn set_v_reset(&mut self, value: f32) {
        self.params.v_reset = value;
    }
    
    #[getter]
    fn v_rest(&self) -> f32 {
        self.params.v_rest
    }
    
    #[setter]
    fn set_v_rest(&mut self, value: f32) {
        self.params.v_rest = value;
    }
    
    #[getter]
    fn refractory_period(&self) -> f32 {
        self.params.refractory_period
    }
    
    #[setter]
    fn set_refractory_period(&mut self, value: f32) -> PyResult<()> {
        if value < 0.0 {
            return Err(PyValueError::new_err("Refractory period must be non-negative"));
        }
        self.params.refractory_period = value;
        Ok(())
    }
    
    /// Get parameters as dictionary
    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("tau_m", self.params.tau_m)?;
            dict.set_item("tau_s", self.params.tau_s)?;
            dict.set_item("v_threshold", self.params.v_threshold)?;
            dict.set_item("v_reset", self.params.v_reset)?;
            dict.set_item("v_rest", self.params.v_rest)?;
            dict.set_item("refractory_period", self.params.refractory_period)?;
            dict.set_item("resistance", self.params.resistance)?;
            dict.set_item("capacitance", self.params.capacitance)?;
            
            // Optional parameters
            if let Some(delta_t) = self.params.delta_t {
                dict.set_item("delta_t", delta_t)?;
            }
            if let Some(v_spike) = self.params.v_spike {
                dict.set_item("v_spike", v_spike)?;
            }
            if let Some(tau_w) = self.params.tau_w {
                dict.set_item("tau_w", tau_w)?;
            }
            if let Some(a) = self.params.a {
                dict.set_item("a", a)?;
            }
            if let Some(b) = self.params.b {
                dict.set_item("b", b)?;
            }
            if let Some(izh_a) = self.params.izh_a {
                dict.set_item("izh_a", izh_a)?;
            }
            if let Some(izh_b) = self.params.izh_b {
                dict.set_item("izh_b", izh_b)?;
            }
            if let Some(izh_c) = self.params.izh_c {
                dict.set_item("izh_c", izh_c)?;
            }
            if let Some(izh_d) = self.params.izh_d {
                dict.set_item("izh_d", izh_d)?;
            }
            
            Ok(dict.to_object(py))
        })
    }
    
    /// Validate parameters
    fn validate(&self) -> PyResult<()> {
        if self.params.tau_m <= 0.0 {
            return Err(PyValueError::new_err("Membrane time constant must be positive"));
        }
        if self.params.tau_s <= 0.0 {
            return Err(PyValueError::new_err("Synaptic time constant must be positive"));
        }
        if self.params.refractory_period < 0.0 {
            return Err(PyValueError::new_err("Refractory period must be non-negative"));
        }
        if self.params.resistance <= 0.0 {
            return Err(PyValueError::new_err("Membrane resistance must be positive"));
        }
        if self.params.capacitance <= 0.0 {
            return Err(PyValueError::new_err("Membrane capacitance must be positive"));
        }
        Ok(())
    }
    
    fn __repr__(&self) -> String {
        format!(
            "NeuronParameters(tau_m={:.2}, V_th={:.1}, V_reset={:.1})",
            self.params.tau_m,
            self.params.v_threshold,
            self.params.v_reset
        )
    }
}

/// Python wrapper for LIF neuron
#[pyclass(name = "LIFNeuron")]
pub struct PyLIFNeuron {
    neuron: LIFNeuron,
}

#[pymethods]
impl PyLIFNeuron {
    #[new]
    fn new(parameters: Option<PyNeuronParameters>) -> Self {
        let params = parameters.map(|p| p.params).unwrap_or_else(NeuronParameters::lif_default);
        Self {
            neuron: LIFNeuron::new(params),
        }
    }
    
    /// Update neuron with input current and return spike if occurred
    fn update(&mut self, input_current: f32, dt: f32) -> PyResult<Option<PyObject>> {
        let time_step = TimeStep::from_secs_f64(dt as f64);
        let synaptic_input = SynapticInput {
            current: input_current,
            conductance_excitatory: 0.0,
            conductance_inhibitory: 0.0,
            reversal_potential_excitatory: 0.0,
            reversal_potential_inhibitory: -70.0,
        };
        
        let spike = self.neuron.update(&synaptic_input, time_step)
            .map_err(core_error_to_py_err)?;
        
        Python::with_gil(|py| {
            match spike {
                Some(spike) => {
                    let py_spike = PySpike::from_spike(spike);
                    Ok(Some(py_spike.into_py(py)))
                },
                None => Ok(None),
            }
        })
    }
    
    /// Get current neuron state
    fn get_state(&self) -> PyNeuronState {
        PyNeuronState {
            state: self.neuron.get_state().clone(),
        }
    }
    
    /// Set neuron state
    fn set_state(&mut self, state: PyNeuronState) {
        self.neuron.set_state(state.state);
    }
    
    /// Get neuron parameters
    fn get_parameters(&self) -> PyNeuronParameters {
        PyNeuronParameters {
            params: self.neuron.get_parameters().clone(),
        }
    }
    
    /// Reset neuron to initial state
    fn reset(&mut self) {
        self.neuron.reset();
    }
    
    /// Get membrane potential
    fn get_membrane_potential(&self) -> f32 {
        self.neuron.get_state().membrane_potential
    }
    
    /// Check if neuron is in refractory period
    fn is_refractory(&self) -> bool {
        matches!(self.neuron.get_state().refractory_state, RefractoryState::Refractory(_))
    }
    
    fn __repr__(&self) -> String {
        format!(
            "LIFNeuron(V={:.2}mV, refractory={})",
            self.neuron.get_state().membrane_potential,
            self.is_refractory()
        )
    }
}

/// Python wrapper for AdEx neuron
#[pyclass(name = "AdExNeuron")]
pub struct PyAdExNeuron {
    neuron: AdExNeuron,
}

#[pymethods]
impl PyAdExNeuron {
    #[new]
    fn new(parameters: Option<PyNeuronParameters>) -> Self {
        let params = parameters.map(|p| p.params).unwrap_or_else(NeuronParameters::adex_default);
        Self {
            neuron: AdExNeuron::new(params),
        }
    }
    
    /// Update neuron and return spike if occurred
    fn update(&mut self, input_current: f32, dt: f32) -> PyResult<Option<PyObject>> {
        let time_step = TimeStep::from_secs_f64(dt as f64);
        let synaptic_input = SynapticInput {
            current: input_current,
            conductance_excitatory: 0.0,
            conductance_inhibitory: 0.0,
            reversal_potential_excitatory: 0.0,
            reversal_potential_inhibitory: -70.0,
        };
        
        let spike = self.neuron.update(&synaptic_input, time_step)
            .map_err(core_error_to_py_err)?;
        
        Python::with_gil(|py| {
            match spike {
                Some(spike) => {
                    let py_spike = PySpike::from_spike(spike);
                    Ok(Some(py_spike.into_py(py)))
                },
                None => Ok(None),
            }
        })
    }
    
    /// Get current neuron state
    fn get_state(&self) -> PyNeuronState {
        PyNeuronState {
            state: self.neuron.get_state().clone(),
        }
    }
    
    /// Set neuron state
    fn set_state(&mut self, state: PyNeuronState) {
        self.neuron.set_state(state.state);
    }
    
    /// Get neuron parameters
    fn get_parameters(&self) -> PyNeuronParameters {
        PyNeuronParameters {
            params: self.neuron.get_parameters().clone(),
        }
    }
    
    /// Reset neuron
    fn reset(&mut self) {
        self.neuron.reset();
    }
    
    /// Get membrane potential
    fn get_membrane_potential(&self) -> f32 {
        self.neuron.get_state().membrane_potential
    }
    
    /// Get adaptation current
    fn get_adaptation_current(&self) -> f32 {
        self.neuron.get_state().adaptation_current
    }
    
    fn __repr__(&self) -> String {
        format!(
            "AdExNeuron(V={:.2}mV, w={:.2})",
            self.neuron.get_state().membrane_potential,
            self.neuron.get_state().adaptation_current
        )
    }
}

/// Python wrapper for Izhikevich neuron
#[pyclass(name = "IzhikevichNeuron")]
pub struct PyIzhikevichNeuron {
    neuron: IzhikevichNeuron,
}

#[pymethods]
impl PyIzhikevichNeuron {
    #[new]
    fn new(parameters: Option<PyNeuronParameters>) -> Self {
        let params = parameters.map(|p| p.params).unwrap_or_else(NeuronParameters::izhikevich_default);
        Self {
            neuron: IzhikevichNeuron::new(params),
        }
    }
    
    /// Update neuron and return spike if occurred
    fn update(&mut self, input_current: f32, dt: f32) -> PyResult<Option<PyObject>> {
        let time_step = TimeStep::from_secs_f64(dt as f64);
        let synaptic_input = SynapticInput {
            current: input_current,
            conductance_excitatory: 0.0,
            conductance_inhibitory: 0.0,
            reversal_potential_excitatory: 0.0,
            reversal_potential_inhibitory: -70.0,
        };
        
        let spike = self.neuron.update(&synaptic_input, time_step)
            .map_err(core_error_to_py_err)?;
        
        Python::with_gil(|py| {
            match spike {
                Some(spike) => {
                    let py_spike = PySpike::from_spike(spike);
                    Ok(Some(py_spike.into_py(py)))
                },
                None => Ok(None),
            }
        })
    }
    
    /// Get current neuron state
    fn get_state(&self) -> PyNeuronState {
        PyNeuronState {
            state: self.neuron.get_state().clone(),
        }
    }
    
    /// Set neuron state
    fn set_state(&mut self, state: PyNeuronState) {
        self.neuron.set_state(state.state);
    }
    
    /// Get neuron parameters
    fn get_parameters(&self) -> PyNeuronParameters {
        PyNeuronParameters {
            params: self.neuron.get_parameters().clone(),
        }
    }
    
    /// Reset neuron
    fn reset(&mut self) {
        self.neuron.reset();
    }
    
    /// Get membrane potential
    fn get_membrane_potential(&self) -> f32 {
        self.neuron.get_state().membrane_potential
    }
    
    /// Get recovery variable
    fn get_recovery_variable(&self) -> f32 {
        self.neuron.get_state().recovery_variable
    }
    
    fn __repr__(&self) -> String {
        format!(
            "IzhikevichNeuron(v={:.2}, u={:.2})",
            self.neuron.get_state().membrane_potential,
            self.neuron.get_state().recovery_variable
        )
    }
}

/// Python wrapper for spike
#[pyclass(name = "Spike")]
#[derive(Clone, Debug)]
pub struct PySpike {
    pub spike: Spike,
}

#[pymethods]
impl PySpike {
    #[new]
    fn new(neuron_id: u32, time: f64, amplitude: Option<f32>) -> Self {
        Self {
            spike: Spike {
                neuron_id,
                time: SpikeTime::from_secs_f64(time),
                amplitude: amplitude.unwrap_or(1.0),
                payload: Vec::new(),
            },
        }
    }
    
    #[getter]
    fn neuron_id(&self) -> u32 {
        self.spike.neuron_id
    }
    
    #[getter]
    fn time(&self) -> f64 {
        self.spike.time.as_secs_f64()
    }
    
    #[getter]
    fn amplitude(&self) -> f32 {
        self.spike.amplitude
    }
    
    #[setter]
    fn set_amplitude(&mut self, value: f32) {
        self.spike.amplitude = value;
    }
    
    fn __repr__(&self) -> String {
        format!(
            "Spike(neuron={}, time={:.3}ms, amp={:.2})",
            self.spike.neuron_id,
            self.spike.time.as_secs_f64() * 1000.0,
            self.spike.amplitude
        )
    }
    
    fn __lt__(&self, other: &Self) -> bool {
        self.spike.time < other.spike.time
    }
    
    fn __le__(&self, other: &Self) -> bool {
        self.spike.time <= other.spike.time
    }
    
    fn __gt__(&self, other: &Self) -> bool {
        self.spike.time > other.spike.time
    }
    
    fn __ge__(&self, other: &Self) -> bool {
        self.spike.time >= other.spike.time
    }
    
    fn __eq__(&self, other: &Self) -> bool {
        self.spike.neuron_id == other.spike.neuron_id && 
        self.spike.time == other.spike.time &&
        (self.spike.amplitude - other.spike.amplitude).abs() < 1e-6
    }
    
    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        self.spike.neuron_id.hash(&mut hasher);
        (self.spike.time.as_secs_f64() * 1e6) as u64.hash(&mut hasher);
        (self.spike.amplitude * 1e6) as u64.hash(&mut hasher);
        hasher.finish()
    }
}

impl PySpike {
    pub fn from_spike(spike: Spike) -> Self {
        Self { spike }
    }
    
    pub fn to_spike(&self) -> Spike {
        self.spike.clone()
    }
}

/// Create spike trains for testing and simulation
#[pyfunction]
pub fn create_poisson_spike_train(
    rate: f32,
    duration: f32,
    neuron_id: u32,
    seed: Option<u64>,
) -> PyResult<Vec<PySpike>> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = match seed {
        Some(seed) => ChaCha8Rng::seed_from_u64(seed),
        None => ChaCha8Rng::from_entropy(),
    };
    
    let mut spikes = Vec::new();
    let mut time = 0.0;
    
    while time < duration {
        let interval = -((1.0 - rng.gen::<f32>()).ln()) / rate;
        time += interval;
        
        if time < duration {
            spikes.push(PySpike::new(neuron_id, time as f64, Some(1.0)));
        }
    }
    
    Ok(spikes)
}

/// Create regular spike train
#[pyfunction]
pub fn create_regular_spike_train(
    frequency: f32,
    duration: f32,
    neuron_id: u32,
    phase: Option<f32>,
) -> Vec<PySpike> {
    let mut spikes = Vec::new();
    let period = 1.0 / frequency;
    let phase_offset = phase.unwrap_or(0.0);
    
    let mut time = phase_offset;
    while time < duration {
        if time >= 0.0 {
            spikes.push(PySpike::new(neuron_id, time as f64, Some(1.0)));
        }
        time += period;
    }
    
    spikes
}

/// Create burst spike train
#[pyfunction]
pub fn create_burst_spike_train(
    burst_rate: f32,
    burst_duration: f32,
    spike_frequency: f32,
    total_duration: f32,
    neuron_id: u32,
) -> Vec<PySpike> {
    let mut spikes = Vec::new();
    let burst_period = 1.0 / burst_rate;
    let spike_period = 1.0 / spike_frequency;
    
    let mut burst_time = 0.0;
    while burst_time < total_duration {
        // Generate spikes within burst
        let mut spike_time = burst_time;
        let burst_end = (burst_time + burst_duration).min(total_duration);
        
        while spike_time < burst_end {
            spikes.push(PySpike::new(neuron_id, spike_time as f64, Some(1.0)));
            spike_time += spike_period;
        }
        
        burst_time += burst_period;
    }
    
    spikes
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::IntoPyDict;
    
    #[test]
    fn test_neuron_state() {
        let mut state = PyNeuronState::new(-70.0, Some(0.0));
        assert_eq!(state.membrane_potential(), -70.0);
        assert_eq!(state.recovery_variable(), 0.0);
        assert!(!state.is_refractory());
        
        state.set_membrane_potential(-50.0);
        assert_eq!(state.membrane_potential(), -50.0);
        
        state.reset();
        assert_eq!(state.membrane_potential(), -70.0);
    }
    
    #[test]
    fn test_neuron_parameters() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Test LIF parameters
            let lif_params = PyNeuronParameters::lif(
                py.get_type::<PyNeuronParameters>(),
                Some(20.0),
                Some(-50.0),
                Some(-70.0),
            );
            assert_eq!(lif_params.tau_m(), 20.0);
            assert_eq!(lif_params.v_threshold(), -50.0);
            assert_eq!(lif_params.v_reset(), -70.0);
            
            // Test validation
            assert!(lif_params.validate().is_ok());
            
            // Test parameter setting
            let mut params = PyNeuronParameters::new("LIF", None).unwrap();
            params.set_tau_m(15.0).unwrap();
            assert_eq!(params.tau_m(), 15.0);
            
            // Test invalid parameter
            assert!(params.set_tau_m(-1.0).is_err());
        });
    }
    
    #[test]
    fn test_lif_neuron() {
        let mut neuron = PyLIFNeuron::new(None);
        
        // Test initial state
        assert_eq!(neuron.get_membrane_potential(), -70.0);
        assert!(!neuron.is_refractory());
        
        // Test update with small current (should not spike)
        let result = neuron.update(1.0, 0.001).unwrap();
        assert!(result.is_none());
        
        // Test reset
        neuron.reset();
        assert_eq!(neuron.get_membrane_potential(), -70.0);
    }
    
    #[test]
    fn test_spike() {
        let spike = PySpike::new(42, 0.001, Some(1.5));
        assert_eq!(spike.neuron_id(), 42);
        assert_eq!(spike.time(), 0.001);
        assert_eq!(spike.amplitude(), 1.5);
        
        let spike2 = PySpike::new(42, 0.002, Some(1.5));
        assert!(spike < spike2);
        assert!(spike <= spike2);
        assert!(spike2 > spike);
        assert!(spike2 >= spike);
        assert!(spike != spike2);
    }
    
    #[test]
    fn test_poisson_spike_train() {
        let spikes = create_poisson_spike_train(100.0, 0.1, 0, Some(42)).unwrap();
        assert!(!spikes.is_empty());
        
        // Check spikes are sorted by time
        for i in 1..spikes.len() {
            assert!(spikes[i-1].time() <= spikes[i].time());
        }
        
        // Check all spikes are within duration
        for spike in &spikes {
            assert!(spike.time() >= 0.0 && spike.time() <= 0.1);
            assert_eq!(spike.neuron_id(), 0);
        }
    }
    
    #[test]
    fn test_regular_spike_train() {
        let spikes = create_regular_spike_train(10.0, 1.0, 1, None);
        assert_eq!(spikes.len(), 10);
        
        // Check regular intervals
        for i in 1..spikes.len() {
            let interval = spikes[i].time() - spikes[i-1].time();
            assert!((interval - 0.1).abs() < 1e-6);
        }
    }
}