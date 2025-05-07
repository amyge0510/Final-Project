from setuptools import setup, find_packages

setup(
    name="amazon-graph-rag",
    version="1.0.0",
    packages=find_packages(include=['src', 'src.*', 'config', 'docs']),
    package_data={
        'src': ['*.py'],
        'config': ['*.yaml'],
        'docs': ['*.md']
    },
    include_package_data=True,
    install_requires=[
        "neo4j>=5.14.1",
        "pandas>=2.1.4",
        "numpy>=1.24.3",
        "tqdm>=4.66.1",
        "langchain>=0.1.0",
        "openai>=1.3.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.2",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "jupyter>=1.0.0",
        ]
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A knowledge graph-based retrieval system for Amazon product data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/amazon-graph-rag",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 