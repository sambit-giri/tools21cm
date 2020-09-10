============
Installation
============

We highly recommend the use of a virtual environement. It helps to keep dependencies required by different projects separate. Few example tools to create virtual environments are `anaconda <https://www.anaconda.com/distribution/>`_, `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ and `venv <https://docs.python.org/3/library/venv.html>`_.

The dependencies should be installed automatically during the installation process. If they fail for some reason, you can install them manually before installing `tools21cm <https://github.com/sambit-giri/tools21cm>`_. The list of required packages can be found in the *requirements.txt* file present in the root directory.

For a standard non-editable installation use::

    pip install git+git://github.com/sambit-giri/tools21cm.git [--user]

The --user is optional and only required if you don't have write permission to your main python installation.
If you wants to work on the code, you can download it directly from the `GitHub <https://github.com/sambit-giri/tools21cm>`_ page or clone the project using::

    git clone git://github.com/sambit-giri/tools21cm.git

Then, you can just install in place without copying anything using::

    pip install -e /path/to/tools21cm [--user]

The package can also be installed using the *setup.py* script. Find the file *setup.py* in the root directory. To install in the standard directory, run::

    python setup.py install

If you do not have write permissions, or you want to install somewhere else, you can specify some other installation directory, for example::

    python setup.py install --home=~/mydir

To see more options, run::

    python setup.py --help-commands

Or look `here <http://docs.python.org/2/install/>`_ for more details.

Tests
-----
For testing, one can use `pytest <https://docs.pytest.org/en/stable/>`_ or `nosetests <https://nose.readthedocs.io/en/latest/>`_. Both packages can be installed using pip. To run all the test script, run the either of the following::

    python -m pytest tests 
    nosetests -v
