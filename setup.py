from setuptools import setup, find_packages

setup(
    name="dsalt",
    version="0.1.0",
    description="Dynamic Sparse Attention with Landmark Tokens — Triton implementation",
    author="Leonardo Cofone",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "triton>=2.0",
    ],
    extras_require={
        "flash": ["flash-attn>=2.0"],
    },
)