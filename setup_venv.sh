#!/bin/bash

set -e

git submodule update --init
python3 -m venv .venv
source .venv/bin/activate
pip uninstall -y nexat-trace
pip install -r requirements_install.txt
pip install -r requirements_dev.txt

# build cython dubins extension dependency and install
python3 setup.py bdist_wheel

pip install dist/nexat_trace-* --force-reinstall --no-cache-dir
# sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' .venv/pyvenv.cfg

