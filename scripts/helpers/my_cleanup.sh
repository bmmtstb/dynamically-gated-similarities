# 1> /dev/null -> hide regular output
# 2> /dev/null -> hide errors

# deactivate all environments
mambda deactivate 2> /dev/null
source ~/.bashrc

# ################### #
# Update Environments #
# ################### #

#echo "updating environments"
#
#/bin/bash ./scripts/update_envs.sh
#
#echo "updates complete"

# ###################### #
# RUN TESTS AND COVERAGE #
# ###################### #

echo "testing code and coverage"

source ./tests/venv/bin/activate

coverage run -m unittest discover -s ./tests/ -p "test__*.py" 1> /dev/null
coverage html --quiet
coverage xml --quiet
coverage report

source ~/.bashrc  # deactivate test venv

rm -rf ./tests/test_data/logs

echo "testing code complete"


# #################### #
# RUN PYLINT AND BLACK #
# #################### #

echo "cleaning code"

source ./tests/venv/bin/activate

pylint .
black . 1> /dev/null

source ~/.bashrc  # deactivate test venv

echo "cleaning code complete"

# ########### #
# CREATE DOCS #
# ########### #

echo "creating docs"
cd docs

/bin/bash ../scripts/custom_build_docs.sh 1> /dev/null

cd ..
echo "created docs"