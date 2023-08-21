## Update local packages for distribution ##
# python -m pip install --user --upgrade setuptools wheel
# python -m pip install --user --upgrade twine

## Create distribution packages on your local machine, and ##
## check the dist/ directory for the new version files ##
python setup.py sdist bdist_wheel
ls dist

## Upload the distribution files to pypi’s test server ##
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
## Check the upload on the test.pypi server: ##
## https://test.pypi.org/project/PACKAGE/VERSION/ ##

## Test the upload with a local installation ##
# conda create --name test_pypi python=3.8
# python -m pip install --index-url https://test.pypi.org/simple/ --no-deps <PACKAGE>

## Upload the distribution files to pypi ##
python -m twine upload dist/*
## Check the upload at pypi: ##
## https://pypi.org/project/PACKAGE/VERSION/ ##