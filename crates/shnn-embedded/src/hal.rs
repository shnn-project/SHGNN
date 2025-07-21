//! Hardware Abstraction Layer (HAL) for embedded neuromorphic systems
//!
//! This module provides platform-specific interfaces for different microcontrollers
//! and hardware accelerators, enabling SHNN deployment across various embedded platforms.

use crate::{
    error::{EmbeddedError, EmbeddedResult},
    fixed_point::{FixedPoint, Q16_16, FixedSpike},
};
use heapless::Vec;
use core::{marker::PhantomData, fmt::Debug};

/// Maximum number of hardware timers
pub const MAX_TIMERS: usize = 8;

/// Maximum number of GPIO pins
pub const MAX_GPIO_PINS: usize = 64;

/// Maximum number of ADC channels
pub const MAX_ADC_CHANNELS: usize = 16;

/// Hardware abstraction trait for embedded platforms
pub trait EmbeddedHAL<T: FixedPoint> {
    /// Hardware timer type
    type Timer: HardwareTimer<T>;
    
    /// GPIO pin type
    type GpioPin: GpioPin;
    
    /// ADC type
    type Adc: AnalogToDigital<T>;
    
    /// PWM output type
    type Pwm: PulseWidthModulation<T>;
    
    /// Initialize hardware platform
    fn init() -> EmbeddedResult<Self> where Self: Sized;
    
    /// Get system clock frequency in Hz
    fn system_clock_freq(&self) -> u32;
    
    /// Get a hardware timer
    fn get_timer(&mut self, id: u8) -> EmbeddedResult<&mut Self::Timer>;
    
    /// Get a GPIO pin
    fn get_gpio(&mut self, pin: u8) -> EmbeddedResult<&mut Self::GpioPin>;
    
    /// Get ADC converter
    fn get_adc(&mut self) -> EmbeddedResult<&mut Self::Adc>;
    
    /// Get PWM output
    fn get_pwm(&mut self, channel: u8) -> EmbeddedResult<&mut Self::Pwm>;
    
    /// Enter low-power mode
    fn enter_low_power(&mut self) -> EmbeddedResult<()>;
    
    /// Exit low-power mode
    fn exit_low_power(&mut self) -> EmbeddedResult<()>;
}

/// Hardware timer trait for precise timing control
pub trait HardwareTimer<T: FixedPoint> {
    /// Start the timer with specified period
    fn start(&mut self, period: T) -> EmbeddedResult<()>;
    
    /// Stop the timer
    fn stop(&mut self) -> EmbeddedResult<()>;
    
    /// Reset timer counter
    fn reset(&mut self) -> EmbeddedResult<()>;
    
    /// Get current timer value
    fn current_value(&self) -> T;
    
    /// Set timer period
    fn set_period(&mut self, period: T) -> EmbeddedResult<()>;
    
    /// Check if timer has expired
    fn has_expired(&self) -> bool;
    
    /// Enable timer interrupt
    fn enable_interrupt(&mut self) -> EmbeddedResult<()>;
    
    /// Disable timer interrupt
    fn disable_interrupt(&mut self) -> EmbeddedResult<()>;
}

/// GPIO pin abstraction for digital I/O
pub trait GpioPin {
    /// Set pin as output
    fn set_output(&mut self) -> EmbeddedResult<()>;
    
    /// Set pin as input
    fn set_input(&mut self) -> EmbeddedResult<()>;
    
    /// Set pin high
    fn set_high(&mut self) -> EmbeddedResult<()>;
    
    /// Set pin low
    fn set_low(&mut self) -> EmbeddedResult<()>;
    
    /// Read pin state
    fn is_high(&self) -> bool;
    
    /// Toggle pin state
    fn toggle(&mut self) -> EmbeddedResult<()>;
    
    /// Enable pull-up resistor
    fn enable_pullup(&mut self) -> EmbeddedResult<()>;
    
    /// Enable pull-down resistor
    fn enable_pulldown(&mut self) -> EmbeddedResult<()>;
    
    /// Disable pull resistors
    fn disable_pull(&mut self) -> EmbeddedResult<()>;
}

/// Analog-to-Digital Converter trait
pub trait AnalogToDigital<T: FixedPoint> {
    /// Read analog value from channel
    fn read_channel(&mut self, channel: u8) -> EmbeddedResult<T>;
    
    /// Start continuous conversion
    fn start_continuous(&mut self, channels: &[u8]) -> EmbeddedResult<()>;
    
    /// Stop continuous conversion
    fn stop_continuous(&mut self) -> EmbeddedResult<()>;
    
    /// Get conversion resolution in bits
    fn resolution(&self) -> u8;
    
    /// Set reference voltage
    fn set_reference_voltage(&mut self, voltage: T) -> EmbeddedResult<()>;
    
    /// Enable DMA for ADC
    fn enable_dma(&mut self) -> EmbeddedResult<()>;
}

/// Pulse Width Modulation output trait
pub trait PulseWidthModulation<T: FixedPoint> {
    /// Set PWM frequency
    fn set_frequency(&mut self, freq: u32) -> EmbeddedResult<()>;
    
    /// Set duty cycle (0.0 to 1.0)
    fn set_duty_cycle(&mut self, duty: T) -> EmbeddedResult<()>;
    
    /// Enable PWM output
    fn enable(&mut self) -> EmbeddedResult<()>;
    
    /// Disable PWM output
    fn disable(&mut self) -> EmbeddedResult<()>;
    
    /// Get current duty cycle
    fn get_duty_cycle(&self) -> T;
}

/// Platform-specific implementations

/// ARM Cortex-M HAL implementation
#[cfg(target_arch = "arm")]
pub mod cortex_m {
    use super::*;
    
    /// Cortex-M HAL implementation
    pub struct CortexMHAL<T: FixedPoint> {
        timers: Vec<CortexMTimer<T>, MAX_TIMERS>,
        gpio_pins: Vec<CortexMGpio, MAX_GPIO_PINS>,
        adc: Option<CortexMAdc<T>>,
        system_freq: u32,
        _phantom: PhantomData<T>,
    }
    
    /// Cortex-M Timer implementation
    pub struct CortexMTimer<T: FixedPoint> {
        id: u8,
        period: T,
        current: T,
        running: bool,
        interrupt_enabled: bool,
        _phantom: PhantomData<T>,
    }
    
    /// Cortex-M GPIO implementation
    pub struct CortexMGpio {
        pin: u8,
        is_output: bool,
        state: bool,
        pull_config: PullConfig,
    }
    
    /// Cortex-M ADC implementation
    pub struct CortexMAdc<T: FixedPoint> {
        resolution: u8,
        reference_voltage: T,
        continuous_mode: bool,
        _phantom: PhantomData<T>,
    }
    
    /// GPIO pull configuration
    #[derive(Debug, Clone, Copy)]
    pub enum PullConfig {
        None,
        PullUp,
        PullDown,
    }
    
    impl<T: FixedPoint> CortexMHAL<T> {
        /// Create new Cortex-M HAL
        pub fn new(system_freq: u32) -> Self {
            Self {
                timers: Vec::new(),
                gpio_pins: Vec::new(),
                adc: None,
                system_freq,
                _phantom: PhantomData,
            }
        }
    }
    
    impl<T: FixedPoint> EmbeddedHAL<T> for CortexMHAL<T> {
        type Timer = CortexMTimer<T>;
        type GpioPin = CortexMGpio;
        type Adc = CortexMAdc<T>;
        type Pwm = CortexMPwm<T>;
        
        fn init() -> EmbeddedResult<Self> {
            let mut hal = Self::new(72_000_000); // 72 MHz default
            
            // Initialize timers
            for i in 0..4 {
                let timer = CortexMTimer::new(i);
                hal.timers.push(timer)
                    .map_err(|_| EmbeddedError::BufferFull)?;
            }
            
            // Initialize GPIO pins
            for pin in 0..32 {
                let gpio = CortexMGpio::new(pin);
                hal.gpio_pins.push(gpio)
                    .map_err(|_| EmbeddedError::BufferFull)?;
            }
            
            // Initialize ADC
            hal.adc = Some(CortexMAdc::new());
            
            Ok(hal)
        }
        
        fn system_clock_freq(&self) -> u32 {
            self.system_freq
        }
        
        fn get_timer(&mut self, id: u8) -> EmbeddedResult<&mut Self::Timer> {
            self.timers.iter_mut()
                .find(|t| t.id == id)
                .ok_or(EmbeddedError::InvalidIndex)
        }
        
        fn get_gpio(&mut self, pin: u8) -> EmbeddedResult<&mut Self::GpioPin> {
            self.gpio_pins.iter_mut()
                .find(|g| g.pin == pin)
                .ok_or(EmbeddedError::InvalidIndex)
        }
        
        fn get_adc(&mut self) -> EmbeddedResult<&mut Self::Adc> {
            self.adc.as_mut().ok_or(EmbeddedError::HardwareNotAvailable)
        }
        
        fn get_pwm(&mut self, _channel: u8) -> EmbeddedResult<&mut Self::Pwm> {
            // PWM implementation would go here
            Err(EmbeddedError::HardwareNotAvailable)
        }
        
        fn enter_low_power(&mut self) -> EmbeddedResult<()> {
            // Platform-specific low-power mode entry
            Ok(())
        }
        
        fn exit_low_power(&mut self) -> EmbeddedResult<()> {
            // Platform-specific low-power mode exit
            Ok(())
        }
    }
    
    impl<T: FixedPoint> CortexMTimer<T> {
        pub fn new(id: u8) -> Self {
            Self {
                id,
                period: T::zero(),
                current: T::zero(),
                running: false,
                interrupt_enabled: false,
                _phantom: PhantomData,
            }
        }
    }
    
    impl<T: FixedPoint> HardwareTimer<T> for CortexMTimer<T> {
        fn start(&mut self, period: T) -> EmbeddedResult<()> {
            self.period = period;
            self.current = T::zero();
            self.running = true;
            Ok(())
        }
        
        fn stop(&mut self) -> EmbeddedResult<()> {
            self.running = false;
            Ok(())
        }
        
        fn reset(&mut self) -> EmbeddedResult<()> {
            self.current = T::zero();
            Ok(())
        }
        
        fn current_value(&self) -> T {
            self.current
        }
        
        fn set_period(&mut self, period: T) -> EmbeddedResult<()> {
            self.period = period;
            Ok(())
        }
        
        fn has_expired(&self) -> bool {
            self.running && self.current >= self.period
        }
        
        fn enable_interrupt(&mut self) -> EmbeddedResult<()> {
            self.interrupt_enabled = true;
            Ok(())
        }
        
        fn disable_interrupt(&mut self) -> EmbeddedResult<()> {
            self.interrupt_enabled = false;
            Ok(())
        }
    }
    
    impl CortexMGpio {
        pub fn new(pin: u8) -> Self {
            Self {
                pin,
                is_output: false,
                state: false,
                pull_config: PullConfig::None,
            }
        }
    }
    
    impl GpioPin for CortexMGpio {
        fn set_output(&mut self) -> EmbeddedResult<()> {
            self.is_output = true;
            Ok(())
        }
        
        fn set_input(&mut self) -> EmbeddedResult<()> {
            self.is_output = false;
            Ok(())
        }
        
        fn set_high(&mut self) -> EmbeddedResult<()> {
            if self.is_output {
                self.state = true;
                Ok(())
            } else {
                Err(EmbeddedError::InvalidConfiguration)
            }
        }
        
        fn set_low(&mut self) -> EmbeddedResult<()> {
            if self.is_output {
                self.state = false;
                Ok(())
            } else {
                Err(EmbeddedError::InvalidConfiguration)
            }
        }
        
        fn is_high(&self) -> bool {
            self.state
        }
        
        fn toggle(&mut self) -> EmbeddedResult<()> {
            if self.is_output {
                self.state = !self.state;
                Ok(())
            } else {
                Err(EmbeddedError::InvalidConfiguration)
            }
        }
        
        fn enable_pullup(&mut self) -> EmbeddedResult<()> {
            self.pull_config = PullConfig::PullUp;
            Ok(())
        }
        
        fn enable_pulldown(&mut self) -> EmbeddedResult<()> {
            self.pull_config = PullConfig::PullDown;
            Ok(())
        }
        
        fn disable_pull(&mut self) -> EmbeddedResult<()> {
            self.pull_config = PullConfig::None;
            Ok(())
        }
    }
    
    impl<T: FixedPoint> CortexMAdc<T> {
        pub fn new() -> Self {
            Self {
                resolution: 12, // 12-bit ADC
                reference_voltage: T::from_float(3.3),
                continuous_mode: false,
                _phantom: PhantomData,
            }
        }
    }
    
    impl<T: FixedPoint> AnalogToDigital<T> for CortexMAdc<T> {
        fn read_channel(&mut self, _channel: u8) -> EmbeddedResult<T> {
            // Simulate ADC reading
            Ok(T::from_float(1.65)) // Half of 3.3V reference
        }
        
        fn start_continuous(&mut self, _channels: &[u8]) -> EmbeddedResult<()> {
            self.continuous_mode = true;
            Ok(())
        }
        
        fn stop_continuous(&mut self) -> EmbeddedResult<()> {
            self.continuous_mode = false;
            Ok(())
        }
        
        fn resolution(&self) -> u8 {
            self.resolution
        }
        
        fn set_reference_voltage(&mut self, voltage: T) -> EmbeddedResult<()> {
            self.reference_voltage = voltage;
            Ok(())
        }
        
        fn enable_dma(&mut self) -> EmbeddedResult<()> {
            // Enable DMA for ADC
            Ok(())
        }
    }
    
    /// Cortex-M PWM implementation placeholder
    pub struct CortexMPwm<T: FixedPoint> {
        channel: u8,
        frequency: u32,
        duty_cycle: T,
        enabled: bool,
        _phantom: PhantomData<T>,
    }
    
    impl<T: FixedPoint> PulseWidthModulation<T> for CortexMPwm<T> {
        fn set_frequency(&mut self, freq: u32) -> EmbeddedResult<()> {
            self.frequency = freq;
            Ok(())
        }
        
        fn set_duty_cycle(&mut self, duty: T) -> EmbeddedResult<()> {
            self.duty_cycle = duty;
            Ok(())
        }
        
        fn enable(&mut self) -> EmbeddedResult<()> {
            self.enabled = true;
            Ok(())
        }
        
        fn disable(&mut self) -> EmbeddedResult<()> {
            self.enabled = false;
            Ok(())
        }
        
        fn get_duty_cycle(&self) -> T {
            self.duty_cycle
        }
    }
}

/// RISC-V HAL implementation
#[cfg(target_arch = "riscv32")]
pub mod riscv {
    use super::*;
    
    /// RISC-V HAL implementation
    pub struct RiscVHAL<T: FixedPoint> {
        system_freq: u32,
        _phantom: PhantomData<T>,
    }
    
    impl<T: FixedPoint> EmbeddedHAL<T> for RiscVHAL<T> {
        type Timer = GenericTimer<T>;
        type GpioPin = GenericGpio;
        type Adc = GenericAdc<T>;
        type Pwm = GenericPwm<T>;
        
        fn init() -> EmbeddedResult<Self> {
            Ok(Self {
                system_freq: 100_000_000, // 100 MHz default
                _phantom: PhantomData,
            })
        }
        
        fn system_clock_freq(&self) -> u32 {
            self.system_freq
        }
        
        fn get_timer(&mut self, _id: u8) -> EmbeddedResult<&mut Self::Timer> {
            Err(EmbeddedError::HardwareNotAvailable)
        }
        
        fn get_gpio(&mut self, _pin: u8) -> EmbeddedResult<&mut Self::GpioPin> {
            Err(EmbeddedError::HardwareNotAvailable)
        }
        
        fn get_adc(&mut self) -> EmbeddedResult<&mut Self::Adc> {
            Err(EmbeddedError::HardwareNotAvailable)
        }
        
        fn get_pwm(&mut self, _channel: u8) -> EmbeddedResult<&mut Self::Pwm> {
            Err(EmbeddedError::HardwareNotAvailable)
        }
        
        fn enter_low_power(&mut self) -> EmbeddedResult<()> {
            Ok(())
        }
        
        fn exit_low_power(&mut self) -> EmbeddedResult<()> {
            Ok(())
        }
    }
}

/// Generic implementations for unsupported platforms
pub struct GenericTimer<T: FixedPoint> {
    _phantom: PhantomData<T>,
}

pub struct GenericGpio;

pub struct GenericAdc<T: FixedPoint> {
    _phantom: PhantomData<T>,
}

pub struct GenericPwm<T: FixedPoint> {
    _phantom: PhantomData<T>,
}

/// Neuromorphic-specific hardware interface
pub trait NeuromorphicHardware<T: FixedPoint> {
    /// Send spike to hardware accelerator
    fn send_spike(&mut self, spike: &FixedSpike<T>) -> EmbeddedResult<()>;
    
    /// Receive spikes from hardware
    fn receive_spikes(&mut self) -> EmbeddedResult<Vec<FixedSpike<T>, 32>>;
    
    /// Configure hardware neural network
    fn configure_network(&mut self, config: &HardwareNetworkConfig<T>) -> EmbeddedResult<()>;
    
    /// Get hardware status
    fn get_status(&self) -> HardwareStatus;
    
    /// Reset hardware accelerator
    fn reset_hardware(&mut self) -> EmbeddedResult<()>;
}

/// Hardware network configuration
#[derive(Debug, Clone)]
pub struct HardwareNetworkConfig<T: FixedPoint> {
    /// Number of neurons
    pub neuron_count: u16,
    /// Synaptic connectivity matrix
    pub connectivity: Vec<(u16, u16, T), 512>, // (src, dst, weight)
    /// Network topology
    pub topology: NetworkTopology,
    /// Clock frequency for hardware
    pub clock_freq: u32,
}

/// Network topology types for hardware
#[derive(Debug, Clone, Copy)]
pub enum NetworkTopology {
    Feedforward,
    Recurrent,
    ConvolutionalSNN,
    CustomTopology,
}

/// Hardware accelerator status
#[derive(Debug, Clone, Copy)]
pub struct HardwareStatus {
    /// Hardware is initialized
    pub initialized: bool,
    /// Current network is loaded
    pub network_loaded: bool,
    /// Processing spikes
    pub processing: bool,
    /// Error state
    pub error: bool,
    /// Power consumption (mW)
    pub power_consumption: u32,
    /// Temperature (°C)
    pub temperature: i16,
}

/// Platform detection and HAL factory
pub struct HALFactory;

impl HALFactory {
    /// Create appropriate HAL for current platform
    pub fn create_hal<T: FixedPoint>() -> EmbeddedResult<Box<dyn EmbeddedHAL<T, Timer = GenericTimer<T>, GpioPin = GenericGpio, Adc = GenericAdc<T>, Pwm = GenericPwm<T>>>> {
        #[cfg(target_arch = "arm")]
        {
            use cortex_m::CortexMHAL;
            Ok(Box::new(CortexMHAL::init()?))
        }
        
        #[cfg(target_arch = "riscv32")]
        {
            use riscv::RiscVHAL;
            Ok(Box::new(RiscVHAL::init()?))
        }
        
        #[cfg(not(any(target_arch = "arm", target_arch = "riscv32")))]
        {
            Err(EmbeddedError::UnsupportedPlatform)
        }
    }
    
    /// Get platform information
    pub fn platform_info() -> PlatformInfo {
        #[cfg(target_arch = "arm")]
        {
            PlatformInfo {
                architecture: "ARM Cortex-M",
                features: &["FPU", "DSP", "DMA", "Timers"],
                max_clock_freq: 180_000_000, // 180 MHz typical
                ram_size: 512 * 1024, // 512 KB typical
                flash_size: 2 * 1024 * 1024, // 2 MB typical
            }
        }
        
        #[cfg(target_arch = "riscv32")]
        {
            PlatformInfo {
                architecture: "RISC-V 32-bit",
                features: &["Integer", "Compressed", "Multiply"],
                max_clock_freq: 100_000_000, // 100 MHz typical
                ram_size: 256 * 1024, // 256 KB typical
                flash_size: 1024 * 1024, // 1 MB typical
            }
        }
        
        #[cfg(not(any(target_arch = "arm", target_arch = "riscv32")))]
        {
            PlatformInfo {
                architecture: "Unknown",
                features: &[],
                max_clock_freq: 0,
                ram_size: 0,
                flash_size: 0,
            }
        }
    }
}

/// Platform information structure
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    /// CPU architecture
    pub architecture: &'static str,
    /// Available hardware features
    pub features: &'static [&'static str],
    /// Maximum clock frequency
    pub max_clock_freq: u32,
    /// RAM size in bytes
    pub ram_size: usize,
    /// Flash memory size in bytes
    pub flash_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_platform_info() {
        let info = HALFactory::platform_info();
        assert!(!info.architecture.is_empty());
    }
    
    #[cfg(target_arch = "arm")]
    #[test]
    fn test_cortex_m_hal() {
        use cortex_m::CortexMHAL;
        let hal = CortexMHAL::<Q16_16>::init().unwrap();
        assert!(hal.system_clock_freq() > 0);
    }
    
    #[test]
    fn test_hardware_config() {
        let config = HardwareNetworkConfig::<Q16_16> {
            neuron_count: 10,
            connectivity: Vec::new(),
            topology: NetworkTopology::Feedforward,
            clock_freq: 100_000_000,
        };
        
        assert_eq!(config.neuron_count, 10);
        assert_eq!(config.clock_freq, 100_000_000);
    }
}