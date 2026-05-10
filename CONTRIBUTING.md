# Contributing to DSALT

Thank you for your interest in contributing to DSALT! We welcome contributions from the community.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/LeonardoCofone/dsalt-library.git
   cd dsalt-pytorch
   ```

3. Install development dependencies:
   ```bash
   make install-dev
   # or
   pip install -e ".[dev]"
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards

3. Run tests and linters:
   ```bash
   make test
   make lint
   ```

4. Format your code:
   ```bash
   make format
   ```

5. Commit your changes:
   ```bash
   git commit -m "Add your descriptive commit message"
   ```

6. Push to your fork and create a pull request

## Coding Standards

### Python Style
- Follow PEP 8
- Use type hints for all function signatures
- Write docstrings for all public functions/classes
- Keep line length under 88 characters (Black default)

### Code Quality Tools
We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

### Testing
- Write tests for all new functionality
- Maintain test coverage above 80%
- Run the full test suite before submitting PRs:
  ```bash
  make test-cov
  ```

## Pull Request Guidelines

### PR Title
Use a clear, descriptive title that explains what the PR does.

### PR Description
Include:
- What changes were made
- Why the changes were needed
- How to test the changes
- Any breaking changes

### Checklist
- [ ] Tests pass locally
- [ ] Code is properly formatted
- [ ] Type hints are correct
- [ ] Documentation is updated
- [ ] No new linting errors

## Reporting Issues

When reporting bugs, please include:
- Python version
- PyTorch version
- Triton version (if applicable)
- CUDA version (if applicable)
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

## Feature Requests

We welcome feature requests! Please:
- Check if the feature is already planned or implemented
- Describe the use case clearly
- Explain why it would be valuable
- Consider implementation complexity

## License

By contributing to DSALT, you agree that your contributions will be licensed under the Apache 2.0 License.