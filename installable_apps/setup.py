from setuptools import setup, find_packages

setup(
    name="letta_installable",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Read requirements from requirements.txt
        req.strip() for req in open("requirements.txt").readlines()
    ],
    entry_points={
        'console_scripts': [
            'letta-startup=startup:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.rst', '*.html', '*.css', '*.js'],
    },
)
