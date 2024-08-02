#!/bin/bash

# Function to activate virtual environment and install packages
activate_and_install() {
    if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
        source "$1"/bin/activate || exit
    elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        "$1"\\Scripts\\activate || exit
    else
        echo "Unsupported OS: $OSTYPE"
        exit 1
    fi
    python -m pip install --upgrade pip
    pip install --upgrade -r $2
    pip install -e . # "install" DGS package
}

## MAIN
echo "updating DGS environment"
activate_and_install "venv" "requirements.txt"
# additionally install torchreid
cd ./dependencies/torchreid/ || exit
python setup.py develop
cd ../..
pip install -e . # "install" DGS package
deactivate

# tests
echo "updating test environment"
activate_and_install "tests/venv" "./tests/requirements_test.txt"
deactivate

# docs
echo "updating docs environment"
activate_and_install "docs/venv" "./docs/requirements_docs.txt"
deactivate
