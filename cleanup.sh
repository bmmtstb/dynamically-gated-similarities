pylint -j 0 . &&
 black . &&
# cd docs && make clean && rm -rf _autosummary && sphinx-build . _build/ -a -j auto && cd ..
 cd docs && make clean && sphinx-build . _build/ -E -a -j auto && cd ..
