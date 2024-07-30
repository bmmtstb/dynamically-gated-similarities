source ./tests/venv/bin/activate

coverage run -m unittest discover -s ./tests/ -p "test__*.py"
coverage html --quiet
coverage xml --quiet
coverage report

# create coverage Badge
hundreds=$(grep -oPm1 "(?<=line-rate=\")[^\.]+" <<< cat ./coverage/coverage.xml)
percentage=$(grep -oPm1 "(?<=line-rate=\".\.)[\d]{2}" <<< cat ./coverage/coverage.xml)
number=$(if [ $hundreds -eq 1 ]; then echo "$hundreds$percentage"; else echo "$percentage"; fi)
color=$(if [ $number -gt 80 ]; then echo "lime";elif [ $number -gt 60 ]; then echo "yellow"; else echo "red";fi)
link=$(echo "https://img.shields.io/badge/Coverage-$number%25-$color")
badge=$(echo '[Coverage]('"$link"')')
# change README.md
sed -i "s|\[Coverage\]\(.*\)|$badge|g" ./README.md

source ~/.bashrc  # deactivate test venv

rm -rf ./tests/test_data/logs
