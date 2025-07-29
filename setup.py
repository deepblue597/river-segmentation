from setuptools import setup, find_packages

setup(
    name="river-segmentation",
    version="0.1.1",
    description="River segmentation pipeline with connectors and models as modules.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "torch",
        "torchvision",
        "segmentation-models-pytorch",
        "albumentations",
        "boto3",
        "sqlalchemy",
        "psycopg2-binary",
        "quixstreams",
        "numpy",
    ],  # You can add requirements here or use requirements.txt
    python_requires=">=3.8",
)
