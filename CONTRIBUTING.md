# Contributing to Chaukas SDK

Thank you for your interest in contributing to the Chaukas SDK! This document provides guidelines for contributing to this Python SDK for agent instrumentation and observability.

## Ways to Contribute

### Bug Reports

Before creating a bug report:
- Check existing issues to avoid duplicates
- Verify the bug with the latest version of the SDK

When submitting a bug report, include:
- **Clear title**: Briefly describe the issue
- **Reproduction steps**: Step-by-step instructions to reproduce the bug
- **Code example**: Minimal code snippet that demonstrates the issue
- **Expected vs. actual behavior**: What should happen vs. what actually happens
- **Environment details**: Python version, SDK version, framework version (CrewAI/OpenAI/etc.)
- **Error logs**: Full stack traces and error messages

### Feature Requests & Enhancements

Feature requests should include:
- **Descriptive title**: Clear summary of the feature
- **Detailed description**: What the feature should do and why it's valuable
- **Use case**: Real-world scenarios where this feature would be useful
- **Proposed implementation**: (Optional) How you envision this working
- **Backward compatibility**: Whether this affects existing functionality

### Questions & Support

For questions about using the SDK:
- Check the [README](README.md) and [examples](examples/) first
- Search existing issues for similar questions
- Use the Question issue template to get help

## Development Guidelines

### Getting Started

1. **Fork the repository** and clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/chaukas-sdk.git
   cd chaukas-sdk
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify setup**:
   ```bash
   make test
   make lint
   ```

### Development Workflow

#### Branch Naming

Use conventional prefixes for your branches:
- `feature/your-feature-name` - New features
- `fix/issue-description` or `bugfix/issue-description` - Bug fixes
- `docs/what-you-are-documenting` - Documentation updates
- `chore/task-description` - Maintenance tasks
- `refactor/what-you-are-refactoring` - Code refactoring

#### Code Style

This project follows Python best practices:

- **Formatting**: Code is formatted with `black` (line length: 88)
- **Import sorting**: Imports are sorted with `isort`
- **Type checking**: Type hints are checked with `mypy`
- **Linting**: Code is linted for common issues

Run formatters before committing:
```bash
make format      # Auto-format code with black and isort
make lint        # Check formatting and types
```

#### Testing

All contributions must include appropriate tests:

- **Unit tests**: For core functionality
- **Integration tests**: For framework integrations (CrewAI, OpenAI, etc.)
- **Example validation**: Ensure examples still work

Run tests:
```bash
make test                    # Run all tests
pytest tests/test_file.py    # Run specific test file
pytest -k test_name          # Run specific test
```

Test requirements:
- All tests must pass
- New features require new tests
- Bug fixes should include regression tests
- Maintain or improve code coverage

#### Documentation

Update documentation when:
- Adding new features
- Changing existing behavior
- Adding new configuration options
- Creating new examples

Documentation includes:
- **Docstrings**: All public classes, methods, and functions
- **README**: Update if adding major features
- **Examples**: Add examples for new integrations or features
- **Type hints**: All function signatures should have type annotations

### Pull Request Process

#### Before Submitting

1. **Update from main**:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Run all checks**:
   ```bash
   make format      # Format code
   make lint        # Check style
   make test        # Run tests
   ```

3. **Test examples** (if applicable):
   ```bash
   python examples/crewai/crewai_example.py
   python examples/openai/openai_simple.py
   ```

4. **Update dependencies** (if you added any):
   ```bash
   # Add to pyproject.toml [project.dependencies] or [project.optional-dependencies]
   ```

#### Submitting the PR

1. **Push your branch**:
   ```bash
   git push origin your-branch
   ```

2. **Create a pull request** on GitHub

3. **Fill out the PR template** with:
   - Description of changes
   - Type of change (bug fix, feature, etc.)
   - Related issue number
   - Testing performed
   - Breaking changes (if any)

4. **Respond to feedback**: Address review comments promptly

#### PR Checklist

- [ ] Code follows the project's style guidelines
- [ ] Self-reviewed the code
- [ ] Added/updated tests for changes
- [ ] All tests pass locally
- [ ] Updated documentation as needed
- [ ] Added examples for new features
- [ ] No breaking changes (or clearly documented if unavoidable)
- [ ] Commit messages are clear and descriptive

## Code Organization

### Project Structure

```
chaukas-sdk/
├── src/chaukas/sdk/
│   ├── core/              # Core SDK functionality
│   │   ├── client.py      # ChaukasClient for event ingestion
│   │   ├── tracer.py      # Distributed tracing
│   │   ├── event_builder.py  # Event construction
│   │   └── config.py      # Configuration management
│   ├── integrations/      # Framework integrations
│   │   ├── crewai.py      # CrewAI integration
│   │   ├── openai_agents.py  # OpenAI Agents integration
│   │   └── google_adk.py  # Google ADK integration
│   └── utils/             # Utility modules
├── tests/                 # Test suite
├── examples/              # Usage examples
└── docs/                  # Documentation
```

### Adding a New Integration

To add support for a new agent framework:

1. **Create integration file**: `src/chaukas/sdk/integrations/your_framework.py`
2. **Implement wrapper class**: Extend `BaseIntegration` if applicable
3. **Add event capture**: Implement hooks for relevant framework events
4. **Add tests**: Create `tests/test_your_framework.py`
5. **Add example**: Create `examples/your_framework/example.py`
6. **Update docs**: Document the integration in README.md

## Backward Compatibility

We strive to maintain backward compatibility. Breaking changes should:

- Be discussed in an issue first
- Be clearly documented
- Provide migration guide
- Follow semantic versioning (major version bump)

### Safe Changes

- Adding new optional parameters
- Adding new methods/classes
- Adding new integrations
- Improving performance
- Fixing bugs
- Adding documentation

### Breaking Changes

Changes that require discussion and major version bump:
- Removing or renaming public APIs
- Changing function signatures
- Changing default behavior
- Removing support for frameworks/Python versions

## Release Process

Releases are managed by maintainers:

1. Version updates in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release tag
4. GitHub Actions automatically builds and publishes to PyPI

## Community

### Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

### Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Email**: Contact maintainers at 2153483+ranesidd@users.noreply.github.com

## Recognition

Contributors are recognized:
- In commit history
- In release notes for significant contributions
- As co-authors in commits when applicable

Thank you for contributing to Chaukas SDK! 🙏
