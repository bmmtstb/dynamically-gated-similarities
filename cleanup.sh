pylint -j 0 . &&
 black . &&
 cd docs && make html --jobs auto -q && cd ..
