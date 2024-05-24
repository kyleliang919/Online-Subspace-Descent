from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="online-subspace-descent",
    version="1.0",
    description="Memory-Efficient LLM Training with Online Subspace Descent",
    url="https://github.com/kyleliang919/Online-Subspace-Descent",
    author="Kaizhao Liang",
    author_email="kaizhaol@utexas.edu",
    license="Apache 2.0",
    packages=["online_subspace_descent_torch"],
    install_requires=required,
)