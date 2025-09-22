from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


def _gather_deps_from_file(path: str):
    lines = []
    with open(path, "r") as file:
        lines = file.readlines()

    deps = []
    for line in lines:
        deps.append(line.strip())
    return lines


parsed_version = None
deps = None
try:
    with open("version", "r", encoding="utf-8") as version_file:
        parsed_version = version_file.read()
        print(f"Preparing to build package with version: {parsed_version}")

    deps = _gather_deps_from_file("requirements_install.txt")
    dev_deps = _gather_deps_from_file("requirements_dev.txt")

    print(f"dependencies are: {deps}")

except FileNotFoundError:
    print(
        "version and/or requirements files not found, exiting"
    )
    exit(-1)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

extensions = [
    Extension(
        "dubins",
        ["pydubins/dubins/src/dubins.c", "pydubins/dubins/dubins.pyx"],
        include_dirs = ["pydubins/dubins/include", "pydubins/dubins/core.pxd"]
    )
]

setup(
    name = "nexat-trace",
    version = parsed_version,
    author = "Fabian Tepe",
    author_email = "f.tepe@nexat.de",
    description = "Nexat Terrain Routing and Coverage Engine",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license = "Apache 2.0",
    packages = find_packages(
        include = ["nexat_trace", "nexat_trace.*"]
    ),
    python_requires = '>=3.10',
    install_requires = deps,
    ext_modules = cythonize(extensions),
    extras_require = {
        "Debugging": dev_deps
    },
    data_files = [
        ("licenses", ["LICENSE", "THIRD_PARTY"])
    ]
)
