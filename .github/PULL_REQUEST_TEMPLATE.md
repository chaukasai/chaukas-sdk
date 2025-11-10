## Description

<!-- Provide a brief description of the changes in this PR -->

## Type of Change

<!-- Check all that apply -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement
- [ ] Code refactoring (no functional changes)
- [ ] Documentation update
- [ ] New example/tutorial
- [ ] Dependency update
- [ ] CI/CD improvement
- [ ] Other (please describe):

## Related Issue

<!-- Link to the issue this PR addresses -->

Fixes #(issue number)

## Changes Made

<!-- Provide a detailed description of your changes -->

### Core Changes
<!-- List main changes to SDK core functionality -->

-
-
-

### Integration Changes
<!-- If this affects framework integrations (CrewAI, OpenAI, etc.) -->

- [ ] CrewAI integration
- [ ] OpenAI Agents integration
- [ ] Google ADK integration
- [ ] New integration: ___________

**Details:**
-
-

### Configuration Changes
<!-- If this adds/changes configuration options -->

-
-

## Testing

<!-- Describe the tests you ran and their results -->

### Test Checklist

- [ ] All existing tests pass (`make test`)
- [ ] Added new unit tests for new functionality
- [ ] Added/updated integration tests
- [ ] Manually tested with example code
- [ ] Tested with all supported Python versions (3.9, 3.10, 3.11, 3.12)
- [ ] Verified backward compatibility

### Manual Testing Performed

<!-- Describe manual testing steps -->

```bash
# Commands run for testing
pytest -v
python examples/...
```

**Results:**
-
-

### Example Code

<!-- If applicable, provide example code demonstrating the changes -->

```python
from chaukas.sdk import enable_chaukas

# Example showing how to use the new feature/fix
```

## Code Quality

<!-- Confirm you've completed these steps -->

- [ ] Code follows the project's style guidelines (`black`, `isort`)
- [ ] Ran linting checks (`make lint`)
- [ ] Ran type checking (`mypy src/`)
- [ ] Self-reviewed my own code
- [ ] Commented complex or non-obvious code
- [ ] Added/updated docstrings for public APIs
- [ ] Added type hints to all new functions/methods

## Documentation

<!-- Check all that apply -->

- [ ] Updated README.md
- [ ] Updated docstrings
- [ ] Added/updated code examples
- [ ] Updated CONTRIBUTING.md (if process changed)
- [ ] Added inline comments for complex logic
- [ ] No documentation changes needed

## Backward Compatibility

<!-- Describe backward compatibility considerations -->

- [ ] This change is fully backward compatible
- [ ] This change includes deprecation warnings for old behavior
- [ ] This is a breaking change (requires major version bump)

**Breaking Changes:**
<!-- If this is a breaking change, describe: -->
<!-- - What breaks -->
<!-- - Why the breaking change is necessary -->
<!-- - Migration guide for users -->

## Dependencies

<!-- List any new dependencies or version changes -->

- [ ] No new dependencies
- [ ] Added new dependencies (listed below)
- [ ] Updated existing dependencies (listed below)

**Dependency Changes:**
-
-

## Performance Impact

<!-- Describe any performance implications -->

- [ ] No performance impact expected
- [ ] Performance improvement (describe below)
- [ ] Potential performance regression (describe below and justify)

**Details:**


## Examples Updated

<!-- Check all examples that were tested/updated -->

- [ ] examples/crewai/crewai_example.py
- [ ] examples/crewai/crewai_comprehensive_example.py
- [ ] examples/openai/openai_simple.py
- [ ] examples/openai/openai_comprehensive_example.py
- [ ] Other: ___________
- [ ] Added new example: ___________
- [ ] No examples affected

## Checklist

<!-- Final checklist before requesting review -->

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published
- [ ] I have checked my code and corrected any misspellings
- [ ] I have updated the version number (if applicable)

## Screenshots / Logs

<!-- If applicable, add screenshots or logs demonstrating the changes -->

## Additional Notes

<!-- Any additional information that reviewers should know -->

## Reviewers

<!-- Tag specific reviewers if needed -->

@chaukasai/maintainers

---

**For Maintainers:**

- [ ] Version bump required?
- [ ] CHANGELOG.md updated?
- [ ] Release notes prepared?
