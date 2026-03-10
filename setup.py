from setuptools import setup, find_packages

setup(
    name="inflow-shield-lib",
    version="1.0.0",
    author="Jonathan",
    author_email="jonathanv@inextlabs.com",
    description="Lightweight AI guardrails — PromptInjection, Toxicity, Secrets, Vault",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "transformers>=4.38.0",
        "torch>=2.2.0",
        "huggingface-hub==0.31.0",
        "tokenizers>=0.15.0",
        "presidio-analyzer>=2.2.354",
        "presidio-anonymizer>=2.2.354",
        "spacy>=3.7.0",
    ],
)
