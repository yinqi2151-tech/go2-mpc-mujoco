from setuptools import setup, find_packages

setup(
    name="go2-mpc-mpc",
    version="0.1.0",
    description="GO2 MPC project built on Isaac Gym and unitree_rl_gym legged_gym environment",
    author="",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "isaacgym",
        "pyyaml>=5.4.0",
        "matplotlib>=3.3.0",
        "cvxpy>=1.1.0",
        "osqp>=0.6.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.7",
)


