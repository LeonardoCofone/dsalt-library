# Contributing to DSALT

Thank you for your interest in contributing to DSALT! We welcome contributions from the community.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/LeonardoCofone/dsalt-library.git
   cd dsalt-library
   ```

3. Install in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

   On a machine with a CUDA GPU, add the Triton extra for the custom kernels:
   ```bash
   pip install -e ".[dev,triton]"
   ```
   Without Triton the library still works and falls back to the SDPA path.

   The available extras are `triton`, `dev`, `docs`, `build`, and `all`
   (see `pyproject.toml`); `dev` already pulls in `pre-commit`.

4. (Optional) Enable the pre-commit hooks so formatting/linting run on every commit:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards below.

3. Format and lint:
   ```bash
   black dsalt
   isort dsalt
   flake8 dsalt
   mypy dsalt
   ```

4. Commit and push to your fork, then open a pull request.

## Coding Standards

### Python Style
- Target Python 3.10+ (the codebase uses `X | None` / `tuple[...]` syntax).
- Use type hints for all function signatures.
- Write docstrings for all public functions/classes.
- Keep line length under 88 characters (Black default).

### Code Quality Tools
- **Black**: code formatting
- **isort**: import sorting
- **flake8**: linting
- **mypy**: type checking

## Pull Request Guidelines

### PR Description
Include:
- What changes were made
- Why the changes were needed
- How to test the changes
- Any breaking changes (e.g. anything that alters trained-model numerics)

## Reporting Issues

When reporting bugs, please include:
- Python version
- PyTorch version
- Triton version (if applicable)
- CUDA version and GPU model (if applicable)
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

## License

By contributing to DSALT, you agree that your contributions will be licensed under the Apache 2.0 License.
