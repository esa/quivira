Install
=======

Quivira uses conda to install Python and its dependencies. Run these commands
from the project folder, where ``environment.yml`` is located.

Create the environment and install Quivira:

.. code-block:: console

   conda env create -f environment.yml
   conda activate quivira

The environment installs Quivira in editable mode, so changes to the source
code are available without reinstalling the package.

To update an existing environment after the dependencies change:

.. code-block:: console

   conda env update -f environment.yml

Check that Quivira is installed:

.. code-block:: console

   python -c "import quivira; print(quivira.__version__)"