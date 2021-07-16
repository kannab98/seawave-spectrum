import setuptools

print(setuptools.find_packages())

with open("README.md", "r") as fh:
    long_description = fh.read()

package_data = {
    "config files":["config.toml"]
}
setuptools.setup(
    name="seawavepy",
    version_command='git describe --tags',
    author="Kirill Ponur",
    author_email="ponur@ipfran.ru",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kannab98/seawave-spectrum",
    packages=setuptools.find_packages(),
    package_data=package_data,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'numba', 'toml'],
)
