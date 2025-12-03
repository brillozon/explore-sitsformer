# Contributing to SITSFormer

Thank you for your interest in contributing to SITSFormer! We welcome contributions from the community and are pleased to have you join us.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [existing issues](https://github.com/brillozon/explore-sitsformer/issues) as you might find that the problem has already been reported. When you are creating a bug report, please include as many details as possible:

- Use a clear and descriptive title
- Describe the exact steps to reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed after following the steps
- Explain which behavior you expected to see instead and why
- Include screenshots if applicable
- Include your environment details (OS, Python version, PyTorch version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as [GitHub issues](https://github.com/brillozon/explore-sitsformer/issues). When creating an enhancement suggestion, please include:

- Use a clear and descriptive title
- Provide a step-by-step description of the suggested enhancement
- Provide specific examples to demonstrate the steps
- Describe the current behavior and explain which behavior you expected to see instead
- Explain why this enhancement would be useful

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Update documentation if needed
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8+
- Poetry (for dependency management)
- Git

### Setting Up Your Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/brillozon/explore-sitsformer.git
cd explore-sitsformer

# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --extras dev

# Activate the virtual environment
poetry shell
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/sitsformer --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run tests matching a pattern
pytest -k "test_attention"
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Type checking with mypy
mypy src/

# Linting with flake8
flake8 src/ tests/

# Security scanning with bandit
bandit -r src/
```

### Documentation

We use Sphinx for documentation. To build docs locally:

```bash
cd docs
make html
# Open docs/_build/html/index.html in your browser
```

## Style Guides

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line
- Consider starting the commit message with an applicable emoji:
  - 🎨 `:art:` when improving the format/structure of the code
  - 🐎 `:racehorse:` when improving performance
  - 📝 `:memo:` when writing docs
  - 🐧 `:penguin:` when fixing something on Linux
  - 🍎 `:apple:` when fixing something on macOS
  - 🏁 `:checkered_flag:` when fixing something on Windows
  - 🐛 `:bug:` when fixing a bug
  - 🔥 `:fire:` when removing code or files
  - ✅ `:white_check_mark:` when adding tests
  - 🔒 `:lock:` when dealing with security
  - ⬆️ `:arrow_up:` when upgrading dependencies
  - ⬇️ `:arrow_down:` when downgrading dependencies

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with these additions:

- Line length: 88 characters (Black's default)
- Use double quotes for strings
- Use type hints for all public functions
- Docstrings should follow Google style

### Documentation Style Guide

- Use reStructuredText for documentation files
- Include docstrings for all public classes and functions
- Keep examples simple and focused
- Include links to related functions/classes

## Branch Organization

We use a simplified git flow:

- `main` - Latest stable release
- `develop` - Integration branch for features
- `feature/feature-name` - Feature branches
- `fix/bug-description` - Bug fix branches
- `release/version` - Release preparation branches

## Issue and Pull Request Labels

### Issue Labels

- `bug` - Something isn't working
- `documentation` - Improvements or additions to documentation
- `duplicate` - This issue or pull request already exists
- `enhancement` - New feature or request
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention is needed
- `invalid` - This doesn't seem right
- `question` - Further information is requested
- `wontfix` - This will not be worked on

### Pull Request Labels

- `work in progress` - Not ready for review
- `needs review` - Ready for review
- `needs changes` - Changes requested
- `ready to merge` - Approved and ready to merge

## Recognition

Contributors who make significant contributions will be recognized in our [AUTHORS](AUTHORS.md) file and release notes.

## Questions?

Feel free to contact the maintainers if you have any questions. You can:

- Open an issue with the `question` label
- Start a discussion in [GitHub Discussions](https://github.com/brillozon/explore-sitsformer/discussions)
- Reach out via email (if provided)

Thank you for contributing to explore-SITSFormer! 🚀