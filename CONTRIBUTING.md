# Contributing to SHNN

Thank you for your interest in contributing to the Spiking Hypergraph Neural Network (SHNN) project! This document outlines the guidelines and processes for contributing to this neuromorphic computing framework.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- **Rust**: Latest stable version (recommended via [rustup](https://rustup.rs/))
- **Python**: 3.8+ (for Python bindings development)
- **Node.js**: 18+ (for WebAssembly development)
- **Git**: Version control

### Additional Tools

```bash
# Development tools
cargo install cargo-watch cargo-audit cargo-llvm-cov
cargo install wasm-pack maturin

# For embedded development
rustup target add thumbv7em-none-eabihf wasm32-unknown-unknown

# For documentation
cargo install mdbook mdbook-linkcheck
```

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/SHNN.git
   cd SHNN
   ```

2. **Build the Project**
   ```bash
   cargo build --all-features
   ```

3. **Run Tests**
   ```bash
   cargo test --all-features
   ```

4. **Build Documentation**
   ```bash
   cargo doc --open --all-features
   ```

## Project Structure

```
SHNN/
├── crates/
│   ├── shnn-core/          # Core neuromorphic primitives
│   ├── shnn-async/         # Async processing extensions
│   ├── shnn-wasm/          # WebAssembly bindings
│   ├── shnn-embedded/      # Embedded/no-std support
│   ├── shnn-ffi/           # Hardware acceleration FFI
│   ├── shnn-python/        # Python bindings
│   └── shnn-bench/         # Benchmarking suite
├── docs/                   # Documentation
├── examples/               # Example projects
└── .github/workflows/      # CI/CD configuration
```

### Crate Dependencies

- **shnn-core**: Foundation crate (no dependencies on other SHNN crates)
- **shnn-async**: Depends on shnn-core
- **shnn-embedded**: Depends on shnn-core (no-std compatible)
- **shnn-wasm**: Depends on shnn-core
- **shnn-ffi**: Depends on shnn-core
- **shnn-python**: Depends on shnn-core
- **shnn-bench**: Depends on all crates for benchmarking

## Contributing Guidelines

### Code Style

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` for consistent formatting
- Run `cargo clippy` and address all warnings
- Use descriptive variable and function names
- Add comprehensive documentation for public APIs

### Neuromorphic Computing Principles

When contributing to neuromorphic components:

- **Biological Realism**: Maintain biological plausibility in neuron models
- **Temporal Dynamics**: Ensure proper handling of time-dependent processes
- **Event-Driven**: Design for sparse, event-driven computation
- **Energy Efficiency**: Optimize for low-power consumption patterns
- **Scalability**: Consider both small embedded and large-scale deployments

### Memory Management

- Use Rust's ownership system for automatic memory safety
- Prefer zero-copy operations where possible
- Use sparse data structures for spike representations
- Consider cache-friendly data layouts
- Document memory complexity of algorithms

### Performance Considerations

- Profile performance-critical code paths
- Use `#[inline]` judiciously for hot functions
- Prefer compile-time optimizations over runtime
- Consider SIMD operations for parallel computation
- Benchmark against existing implementations

## Testing

### Unit Tests

```bash
# Run all unit tests
cargo test --all-features

# Run tests for specific crate
cargo test --package shnn-core

# Run tests with coverage
cargo llvm-cov --all-features
```

### Integration Tests

```bash
# Test WebAssembly compilation
wasm-pack test crates/shnn-wasm --headless --firefox

# Test embedded compilation
cargo build --package shnn-embedded --target thumbv7em-none-eabihf

# Test Python bindings
cd crates/shnn-python
maturin develop
python -m pytest tests/
```

### Benchmarks

```bash
# Run performance benchmarks
cargo bench --package shnn-bench

# Compare with baseline
cargo bench --package shnn-bench -- --save-baseline main
```

### Test Guidelines

- Write tests for all public APIs
- Include edge cases and error conditions
- Test both success and failure paths
- Verify neuromorphic properties (e.g., spike timing)
- Use property-based testing for complex algorithms

## Documentation

### Code Documentation

- Document all public functions, structs, and traits
- Include examples in documentation
- Explain neuromorphic concepts and biological motivation
- Document safety requirements for `unsafe` code
- Use `#[doc(hidden)]` for internal APIs

### Architecture Documentation

- Update design documents for significant changes
- Include performance characteristics
- Document hardware acceleration interfaces
- Explain deployment scenarios and trade-offs

### Examples

- Provide runnable examples for new features
- Include both simple and complex use cases
- Document expected outputs and performance
- Test examples in CI pipeline

## Pull Request Process

### Before Submitting

1. **Create an Issue**: Discuss significant changes before implementation
2. **Branch**: Create a feature branch from `main`
3. **Implement**: Follow coding guidelines and test thoroughly
4. **Document**: Update documentation and examples as needed

### Submission Requirements

- [ ] All tests pass (`cargo test --all-features`)
- [ ] Code is properly formatted (`cargo fmt`)
- [ ] No clippy warnings (`cargo clippy --all-features`)
- [ ] Documentation is complete and accurate
- [ ] CHANGELOG.md is updated (for significant changes)
- [ ] Examples work and are tested
- [ ] Performance impact is documented (if applicable)

### Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed
- [ ] Benchmarks updated (if applicable)

## Neuromorphic Impact
- How do these changes affect neuromorphic computation?
- Are biological principles maintained?
- Performance implications for different deployment targets?

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
```

### Review Process

1. **Automated Checks**: CI pipeline must pass
2. **Code Review**: At least one maintainer approval required
3. **Testing**: Thorough testing on multiple platforms
4. **Documentation**: Review of documentation completeness
5. **Performance**: Benchmark review for performance-critical changes

## Release Process

### Version Numbers

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

### Release Checklist

- [ ] Update version numbers in all `Cargo.toml` files
- [ ] Update `CHANGELOG.md` with release notes
- [ ] Tag release with `git tag vX.Y.Z`
- [ ] GitHub Actions handles automated publishing
- [ ] Verify packages on crates.io, PyPI, and npm
- [ ] Update documentation deployment

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Documentation**: Comprehensive guides and API reference

### Neuromorphic Computing Resources

- [Neuromorphic Engineering Community](https://neuromorphic.eecn.fsu.edu/)
- [SpiNNaker Project](https://spinnakermanchester.github.io/)
- [Intel Loihi](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html)
- [NEST Simulator](https://www.nest-simulator.org/)

### Rust Resources

- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Embedded Rust Book](https://docs.rust-embedded.org/book/)

## Recognition

Contributors will be recognized in:
- `CHANGELOG.md` for significant contributions
- GitHub contributors list
- Project documentation acknowledgments

Thank you for contributing to advancing neuromorphic computing with Rust!