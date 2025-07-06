# Contributing to Marathon Time Predictor

Thank you for your interest in contributing to the Marathon Time Predictor project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

We welcome contributions from the community! Here are the main ways you can contribute:

### 🐛 Report Bugs

- Use the [GitHub Issues](https://github.com/yourusername/marathon-time-predictor/issues) page
- Include a clear description of the bug
- Provide steps to reproduce the issue
- Include your environment details (OS, Python version, etc.)

### 💡 Suggest Enhancements

- Use the [GitHub Discussions](https://github.com/yourusername/marathon-time-predictor/discussions) page
- Describe the feature you'd like to see
- Explain why this feature would be useful
- Provide examples if possible

### 🔧 Submit Code Changes

- Fork the repository
- Create a feature branch
- Make your changes
- Add tests for new functionality
- Submit a pull request

## 🚀 Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda

### Local Development Environment

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/marathon-time-predictor.git
   cd marathon-time-predictor
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

5. **Run tests to ensure everything works**
   ```bash
   pytest
   ```

## 📝 Coding Standards

### Python Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Import sorting**: Use `isort`
- **Code formatting**: Use `black`
- **Type hints**: Use type hints for all function parameters and return values
- **Docstrings**: Use Google-style docstrings

### Code Quality Tools

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **isort**: Import sorting
- **mypy**: Type checking
- **pytest**: Testing

### Running Quality Checks

```bash
# Format code
black .

# Sort imports
isort .

# Run linter
flake8 .

# Type checking
mypy .

# Run all checks
pre-commit run --all-files
```

## 🧪 Testing

### Writing Tests

- Write tests for all new functionality
- Use `pytest` as the testing framework
- Place tests in the `tests/` directory
- Use descriptive test names
- Include both unit tests and integration tests

### Test Structure

```
tests/
├── test_marathon_prediction.py  # Core model tests
├── test_api.py                  # API endpoint tests
├── test_validation.py           # Input validation tests
└── conftest.py                  # Pytest configuration
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=marathon_prediction

# Run specific test file
pytest tests/test_marathon_prediction.py

# Run with verbose output
pytest -v
```

## 🔄 Pull Request Process

### Before Submitting a PR

1. **Ensure your code follows the style guidelines**

   ```bash
   black .
   isort .
   flake8 .
   ```

2. **Run the test suite**

   ```bash
   pytest
   ```

3. **Update documentation** if needed

   - Update README.md if you've added new features
   - Add docstrings for new functions
   - Update API documentation if endpoints changed

4. **Check that all CI checks pass**

### PR Guidelines

1. **Create a descriptive title** for your PR
2. **Provide a detailed description** of your changes
3. **Reference any related issues** using `#issue_number`
4. **Include tests** for new functionality
5. **Update documentation** as needed
6. **Keep PRs focused** - one feature or bug fix per PR

### PR Template

```markdown
## Description

Brief description of the changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Testing

- [ ] Added tests for new functionality
- [ ] All existing tests pass
- [ ] Manual testing completed

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented if necessary)

## Related Issues

Closes #issue_number
```

## 📚 Documentation

### Code Documentation

- Use Google-style docstrings for all functions and classes
- Include type hints for all parameters and return values
- Document exceptions that may be raised
- Provide usage examples for complex functions

### API Documentation

- Update API documentation when endpoints change
- Include request/response examples
- Document error codes and messages

### README Updates

- Update README.md for new features
- Keep installation instructions current
- Update usage examples as needed

## 🏗️ Project Structure

### Adding New Features

1. **Create a feature branch** from `main`
2. **Implement the feature** following coding standards
3. **Add tests** for the new functionality
4. **Update documentation** as needed
5. **Submit a PR** with a clear description

### Adding New Dependencies

- Add to `requirements.txt` for production dependencies
- Add to `requirements-dev.txt` for development dependencies
- Document why the dependency is needed
- Ensure it's actively maintained and secure

## 🐛 Bug Reports

### Bug Report Template

```markdown
## Bug Description

Clear description of the bug

## Steps to Reproduce

1. Step 1
2. Step 2
3. Step 3

## Expected Behavior

What should happen

## Actual Behavior

What actually happens

## Environment

- OS: [e.g., macOS 12.0]
- Python version: [e.g., 3.9.7]
- Package versions: [output of `pip freeze`]

## Additional Information

Any other relevant information
```

## 🎯 Feature Requests

### Feature Request Template

```markdown
## Feature Description

Clear description of the requested feature

## Use Case

Why this feature would be useful

## Proposed Implementation

How you think this could be implemented (optional)

## Alternatives Considered

Other approaches you've considered (optional)
```

## 📞 Getting Help

If you need help with contributing:

1. **Check existing issues** and discussions
2. **Read the documentation** in the README
3. **Ask questions** in GitHub Discussions
4. **Join our community** (if applicable)

## 🏆 Recognition

Contributors will be recognized in:

- The project README
- Release notes
- GitHub contributors page

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to Marathon Time Predictor! 🏃‍♂️
