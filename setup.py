from setuptools import setup, find_packages

setup(
    name="chaukas-sdk",
    version="0.1.0",
    description="One-line instrumentation for agent building SDKs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Chaukas",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "chaukas-spec-client>=1.0.0",
        "wrapt>=1.14.0",
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
        "typing-extensions>=4.0.0",
        "protobuf>=5.26.0",
        "grpcio>=1.50.0",
        "uuid6>=2024.1.12",  # For UUID7 generation
    ],
    extras_require={
        "openai": ["openai-agents"],
        "google": ["adk"],
        "crewai": ["crewai"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)