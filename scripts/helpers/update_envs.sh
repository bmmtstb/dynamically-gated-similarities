## MAIN
echo "updating DGS environment"
source ./venv/bin/activate
pip install --upgrade pip
pip install --upgrade -r requirements.txt
# install torchreid
cd ./dependencies/torchreid/
python setup.py develop
cd ../..
pip install -e . # "install" DGS package
source ~/.bashrc

# tests
echo "updating test environment"
source ./tests/venv/bin/activate
pip install --upgrade pip
pip install --upgrade -r ./tests/requirements_test.txt
pip install -e . # "install" DGS package
source ~/.bashrc

# docs
echo "updating docs environment"
source ./docs/venv/bin/activate
pip install --upgrade pip
pip install --upgrade -r ./docs/requirements_docs.txt
pip install -e . # "install" DGS package
source ~/.bashrc
