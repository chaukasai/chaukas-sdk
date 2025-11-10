# Security Policy

## Supported Versions

We release security patches for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Reporting a Vulnerability

We take security seriously and appreciate responsible disclosure. If you discover a security vulnerability in the Chaukas SDK, please report it privately.

### How to Report

**Email**: 2153483+ranesidd@users.noreply.github.com

**Subject**: `[SECURITY] Brief description of the issue`

**Please DO NOT**:
- Open a public GitHub issue for security vulnerabilities
- Disclose the vulnerability publicly before we've had a chance to address it

### Information to Include

To help us understand and address the issue quickly, please include:

1. **Description**: Clear description of the vulnerability
2. **Steps to Reproduce**: Detailed steps to reproduce the issue
3. **Impact Assessment**: Potential impact and severity
4. **Affected Versions**: Which versions are affected
5. **Proof of Concept**: (Optional) Code demonstrating the vulnerability
6. **Suggested Fix**: (Optional) Your ideas for addressing the issue
7. **Disclosure Timeline**: Your expected timeline for public disclosure

### Our Commitment

When you report a vulnerability, we commit to:

1. **Acknowledge**: Respond within 48 hours to confirm receipt
2. **Assess**: Evaluate the vulnerability and determine severity
3. **Update**: Provide regular updates on our progress
4. **Fix**: Develop and test a patch
5. **Release**: Publish a security advisory and patched version
6. **Credit**: Recognize your responsible disclosure (unless you prefer anonymity)

## Security Scope

The following areas are within the scope of our security policy:

### SDK Vulnerabilities

- **Code Injection**: Vulnerabilities allowing arbitrary code execution
- **Data Exposure**: Unintended exposure of sensitive data (API keys, credentials, event data)
- **Authentication/Authorization**: Bypass of security controls
- **Input Validation**: Improper validation leading to security issues
- **Dependency Vulnerabilities**: Security issues in third-party dependencies

### Integration Security

- **Framework Integration Issues**: Security problems in CrewAI, OpenAI, Google ADK integrations
- **Event Data Handling**: Improper sanitization of captured event data
- **Monkey Patching Safety**: Security issues from instrumentation hooks
- **Context Propagation**: Information leakage through trace/span context

### Infrastructure & Supply Chain

- **Dependency Vulnerabilities**: Known CVEs in dependencies
- **Build Process**: Compromised build or release artifacts
- **Package Integrity**: Tampering with PyPI packages
- **CI/CD Security**: Vulnerabilities in GitHub Actions workflows

### Documentation

- **Insecure Examples**: Code examples demonstrating insecure practices
- **Misleading Guidance**: Documentation encouraging vulnerable configurations

## Out of Scope

The following are generally out of scope:

- Vulnerabilities in the Chaukas server/platform (report to the appropriate repository)
- Issues in third-party frameworks (CrewAI, OpenAI) themselves
- Social engineering attacks
- Physical attacks
- Denial of Service attacks against public infrastructure
- Reports from automated scanners without validation

## Security Best Practices

### For SDK Users

1. **API Key Management**
   - Never hardcode API keys in source code
   - Use environment variables or secure secret management
   - Rotate API keys regularly
   - Use read-only keys when possible

   ```python
   # Good
   config = ChaukasConfig(
       api_key=os.getenv("CHAUKAS_API_KEY")
   )

   # Bad
   config = ChaukasConfig(
       api_key="sk_live_abc123..."  # Never do this!
   )
   ```

2. **Sensitive Data**
   - Sanitize sensitive data before it reaches the SDK
   - Use custom hooks to filter PII from events
   - Be aware of what data your agents process

   ```python
   from chaukas.sdk import enable_chaukas

   def sanitize_event(event):
       # Remove sensitive fields before sending
       if hasattr(event, 'data'):
           event.data.password = "[REDACTED]"
       return event

   enable_chaukas(
       custom_hooks={"before_send": sanitize_event}
   )
   ```

3. **Network Security**
   - Always use HTTPS endpoints (default)
   - Implement TLS verification (enabled by default)
   - Use firewall rules to restrict outbound traffic if needed

4. **Dependency Updates**
   - Regularly update to the latest SDK version
   - Monitor security advisories
   - Use `pip-audit` or similar tools to scan dependencies

   ```bash
   pip install --upgrade chaukas-sdk
   pip-audit  # Check for known vulnerabilities
   ```

5. **Least Privilege**
   - Use minimal scopes for API keys
   - Implement rate limiting in production
   - Monitor SDK usage and event volumes

### For Contributors

1. **Code Review**
   - All changes require review before merging
   - Security-sensitive changes require extra scrutiny
   - Use static analysis tools (`mypy`, `bandit`)

2. **Input Validation**
   - Validate all external inputs
   - Sanitize data before logging or sending
   - Use parameterized queries/prepared statements

3. **Dependencies**
   - Minimize third-party dependencies
   - Keep dependencies up to date
   - Regularly audit with `pip-audit`
   - Pin versions in `pyproject.toml`

4. **Testing**
   - Include security test cases
   - Test error handling and edge cases
   - Never commit secrets or API keys

## Known Security Considerations

### Event Data Content

The SDK captures events from your agent workflows. Be aware that:
- Events may contain sensitive data from your agents
- All event data is transmitted to the Chaukas platform
- You are responsible for sanitizing sensitive data before it reaches the SDK
- Consider using custom hooks to filter events

### Instrumentation Hooks

The SDK uses monkey patching to instrument frameworks:
- Hooks modify framework behavior at runtime
- Only enable integrations you trust and need
- Review integration code if you have security concerns

### Dependencies

The SDK depends on several packages:
- `chaukas-spec-client`: Official Chaukas protobuf client
- `httpx`: HTTP client for API communication
- `protobuf`: Protocol buffer serialization
- Framework-specific dependencies (optional)

We regularly update dependencies and monitor for vulnerabilities.

## Security Updates

Security updates are released as:
- **Patch versions** (0.1.x) for minor fixes
- **Security advisories** on GitHub
- **Release notes** highlighting security fixes

Subscribe to releases to stay informed:
- Watch this repository on GitHub
- Monitor PyPI release notifications
- Follow security advisories

## Questions?

For non-security questions about this policy, contact:
- **Email**: 2153483+ranesidd@users.noreply.github.com
- **GitHub Issues**: For general security questions (not vulnerabilities)

Thank you for helping keep Chaukas SDK secure!
