# set up environment
mamba deactivate 2> /dev/null

source ~/.bashrc
source venv/bin/activate

# clean
make clean 1> /dev/null
rm -rf dgs_autosummary/ _build/

# first linkcheck without cache, then build using that cache
PYTHONWARNINGS= sphinx-build . _build/ -T -v -E -a -j auto -b linkcheck
PYTHONWARNINGS= sphinx-build . _build/ -T -v -j auto
