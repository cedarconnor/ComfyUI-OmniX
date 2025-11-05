# Contributing to ComfyUI-OmniX

Thank you for your interest in contributing to ComfyUI-OmniX! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/ComfyUI-OmniX.git
cd ComfyUI-OmniX
```

2. **Install development dependencies**
```bash
pip install -r requirements.txt
pip install pytest black flake8  # Development tools
```

3. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

## Code Style

We follow Python best practices:

- **Formatting**: Use `black` for code formatting
- **Linting**: Code should pass `flake8` checks
- **Type Hints**: Add type hints to function signatures
- **Docstrings**: Document all public functions and classes

```python
def example_function(param: str, value: int = 0) -> bool:
    """
    Brief description of function.

    Args:
        param: Description of param
        value: Description of value (default: 0)

    Returns:
        Description of return value

    Raises:
        ValueError: When value is negative
    """
    pass
```

## Testing

Before submitting a pull request:

1. **Run unit tests**
```bash
pytest tests/
```

2. **Test in ComfyUI**
   - Copy your changes to `ComfyUI/custom_nodes/ComfyUI-OmniX/`
   - Test all modified nodes in actual workflows
   - Verify error handling works correctly

3. **Check memory usage**
   - Monitor VRAM usage during operations
   - Test on different hardware configs if possible

## Submitting Changes

1. **Commit your changes**
```bash
git add .
git commit -m "Add feature: brief description"
```

2. **Push to your fork**
```bash
git push origin feature/your-feature-name
```

3. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your feature branch
   - Fill out the PR template

### Pull Request Guidelines

- **Title**: Clear and descriptive (e.g., "Add cubemap conversion node")
- **Description**: Explain what changes were made and why
- **Testing**: Describe how you tested the changes
- **Screenshots**: Include screenshots for UI changes
- **Breaking Changes**: Clearly mark any breaking changes

## Areas for Contribution

### High Priority

- [ ] **Performance Optimization**: Improve inference speed
- [ ] **Memory Efficiency**: Reduce VRAM usage
- [ ] **Error Handling**: Better error messages and recovery
- [ ] **Documentation**: Improve README, add tutorials

### Feature Additions

- [ ] **Cubemap Conversion**: Equirectangular to cubemap
- [ ] **Material Packer**: Combine maps for game engines
- [ ] **Batch Processing**: Process multiple panoramas
- [ ] **Preview Node**: Interactive 360° viewer
- [ ] **Advanced Adapters**: Support for new OmniX features

### Testing & Quality

- [ ] **Unit Tests**: Expand test coverage
- [ ] **Integration Tests**: End-to-end workflow tests
- [ ] **Performance Benchmarks**: Track speed improvements
- [ ] **Cross-platform Testing**: Test on Windows/Linux/Mac

### Documentation

- [ ] **Video Tutorials**: Workflow demonstrations
- [ ] **API Documentation**: Detailed node reference
- [ ] **Example Workflows**: More use cases
- [ ] **Troubleshooting**: Common issues and solutions

## Code Review Process

1. **Automated Checks**: CI runs tests and linting
2. **Maintainer Review**: Code review by project maintainers
3. **Community Feedback**: Other contributors may provide input
4. **Approval**: Once approved, changes will be merged

## Bug Reports

When reporting bugs, please include:

- **ComfyUI Version**: Your ComfyUI version
- **OmniX Version**: Your ComfyUI-OmniX version
- **Hardware**: GPU model, VRAM amount
- **Error Message**: Full error traceback
- **Steps to Reproduce**: How to trigger the bug
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens

## Feature Requests

When requesting features, please include:

- **Use Case**: Why you need this feature
- **Proposed Solution**: How you envision it working
- **Alternatives**: Other approaches you've considered
- **Additional Context**: Examples, mockups, etc.

## Community

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **ComfyUI Discord**: Real-time community chat

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

Thank you for helping make ComfyUI-OmniX better!
