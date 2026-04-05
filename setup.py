from setuptools import setup, find_packages

setup(
    name="safety-fragility",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "trl>=0.8.0",
        "peft>=0.10.0",
        "pyhessian>=0.1",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "wandb>=0.16.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "safetensors>=0.4.0",
        "accelerate>=0.27.0",
    ],
)
