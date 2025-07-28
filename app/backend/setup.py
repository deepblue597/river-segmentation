from setuptools import setup, find_packages

setup(
    name="river-segmentation",
    version="0.1.0",
    description="River segmentation pipeline with connectors and models as modules.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "requests",
        "boto3",
        "numpy",
        "pillow",
        "sqlalchemy",
        "psycopg2-binary",
        "quixstreams",
],  # You can add requirements here or use requirements.txt
    python_requires=">=3.8",
)
