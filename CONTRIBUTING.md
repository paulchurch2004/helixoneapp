# 🤝 Contributing to HelixOne

First off, thank you for considering contributing to HelixOne! It's people like you that make HelixOne such a great tool.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)

## 📜 Code of Conduct

This project and everyone participating in it is governed by a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to support@helixone.fr.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/helixoneapp.git
   cd helixoneapp
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/paulchurch2004/helixoneapp.git
   ```

## 💻 Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- macOS (for Mac builds) or Windows (for Windows builds)

### Installation

1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```
   This will automatically run linters and formatters before each commit.

4. **Run the app in dev mode**:
   ```bash
   ./dev.sh  # On Windows: python src/main.py
   ```

## 🔧 How to Contribute

### Reporting Bugs

Bugs are tracked as GitHub issues. When creating a bug report, please include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Screenshots** if applicable
- **Environment details** (OS, Python version, HelixOne version)

### Suggesting Enhancements

Enhancement suggestions are also tracked as GitHub issues. When suggesting an enhancement:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful**
- **Include mockups or examples** if applicable

### Code Contributions

1. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```

2. **Make your changes** following our [coding standards](#coding-standards)

3. **Write or update tests** as needed

4. **Run the test suite**:
   ```bash
   pytest tests/
   ```

5. **Run linters**:
   ```bash
   ruff check src/ tests/
   black src/ tests/
   ```

6. **Commit your changes**:
   ```bash
   git commit -m "feat: add new feature"
   ```
   Follow [Conventional Commits](https://www.conventionalcommits.org/) format:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test changes
   - `refactor:` for code refactoring
   - `chore:` for maintenance tasks

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub

## 📏 Coding Standards

### Python Style Guide

- Follow **PEP 8** style guide
- Use **Black** for code formatting (line length: 100)
- Use **Ruff** for linting
- Use **type hints** where appropriate
- Write **docstrings** for all public functions and classes

### Code Quality

- Keep functions small and focused
- Use meaningful variable and function names
- Add comments for complex logic
- Avoid deep nesting (max 3 levels)
- No hardcoded credentials or sensitive data

### Example

```python
"""
Module docstring explaining what this module does
"""
from typing import Optional


def calculate_profit(
    buy_price: float,
    sell_price: float,
    quantity: int,
    commission: float = 0.001,
) -> float:
    """
    Calculate profit from a trade

    Args:
        buy_price: Price at which asset was bought
        sell_price: Price at which asset was sold
        quantity: Number of units traded
        commission: Trading commission (default: 0.1%)

    Returns:
        Profit or loss amount

    Example:
        >>> calculate_profit(100, 110, 10)
        99.0
    """
    gross_profit = (sell_price - buy_price) * quantity
    commission_cost = (buy_price + sell_price) * quantity * commission
    return gross_profit - commission_cost
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_auth_manager.py -v
```

### Writing Tests

- Write tests for all new features
- Aim for >80% code coverage
- Use descriptive test names
- Use fixtures for common setup
- Mock external dependencies

### Test Structure

```python
import pytest
from src.module import function_to_test


class TestFeature:
    """Test suite for feature X"""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data"""
        return {"key": "value"}

    def test_basic_functionality(self, sample_data):
        """Test that basic functionality works"""
        result = function_to_test(sample_data)
        assert result == expected_value

    @pytest.mark.parametrize(
        "input,expected",
        [
            (1, 2),
            (2, 4),
            (3, 6),
        ],
    )
    def test_multiple_cases(self, input, expected):
        """Test multiple cases"""
        assert function_to_test(input) == expected
```

## 🔄 Pull Request Process

1. **Ensure all tests pass** and code is properly formatted
2. **Update documentation** if you've changed APIs or added features
3. **Add entries to CHANGELOG.md** if applicable
4. **Link related issues** in the PR description
5. **Wait for review** from maintainers
6. **Address review comments** promptly
7. **Squash commits** if requested
8. **Celebrate** when your PR is merged! 🎉

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added for new features
- [ ] All CI checks pass

## 📁 Project Structure

```
helixoneapp/
├── .github/
│   └── workflows/          # GitHub Actions CI/CD
├── assets/                 # Images, icons, sounds
├── helixone-backend/       # FastAPI backend
├── installer/              # Installer configurations
├── src/
│   ├── interface/          # UI components
│   ├── updater/            # Auto-update logic
│   ├── analytics.py        # Usage analytics
│   ├── auth_manager.py     # Authentication
│   ├── monitoring.py       # Crash reporting
│   └── ...
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
├── pyproject.toml          # Project configuration
└── README.md
```

## 🐛 Debugging

### Enable Debug Mode

Set environment variable:
```bash
export HELIXONE_DEV=1
./dev.sh
```

### Check Logs

- **macOS**: `~/Library/Logs/HelixOne/`
- **Windows**: `%APPDATA%\HelixOne\Logs\`
- **Linux**: `~/.local/share/HelixOne/logs/`

### Common Issues

**Import errors**: Make sure virtual environment is activated
**Tests fail**: Check that all dependencies are installed
**Pre-commit hooks fail**: Run `black` and `ruff` manually to fix issues

## 💡 Tips

- Keep PRs focused and small
- Write clear commit messages
- Test on different platforms when possible
- Ask questions if something is unclear
- Be patient and respectful

## 📧 Contact

- **Email**: support@helixone.fr
- **GitHub Issues**: https://github.com/paulchurch2004/helixoneapp/issues
- **Website**: https://helixone.fr

## 🙏 Recognition

Contributors will be recognized in the project README and release notes.

---

**Thank you for contributing to HelixOne!** 🚀
