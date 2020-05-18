.. Tutorial on Documentation using Sphinx documentation master file, created by
   sphinx-quickstart on Fri May 08 16:23:30 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive. https://medium.com/@richdayandnight/a-simple-tutorial-on-how-to-document-your-python-project-using-sphinx-and-rinohtype-177c22a15b5b


Configuration of Project Environment
*************************************

This is an API that implements a system to evaluate odometry trajectories.

Overview on How to Run this API
================================
1. Either install a Python IDE or create a Python virtual environment to install the packages required
2. Install packages required

Setup procedure
================
1. Configure project environment (Either A. Install Pycharm OR B. Create a Virtual Environment)
    A. Install Pycharm (www.jetbrains.com/pycharm/download/) or Sublime (https://www.sublimetext.com/3)
        - configure pylinter

    B. Create a Python Virtual Environment
        - Install virtualenv::

            sudo pip install virtualenv

        - Create virtialenv::

            virtualenv -p python3 <name of virtualenv>

        - Install requirements::

            pip install -r requirements.txt

2. Run app.py

    python run.py path_gt path_pred -v
    python run.py path_gt path_pred -ate
    python run.py path_gt path_pred -rpe
    python run.py path_gt path_pred -v -ate -rpe


Documentation for the Code
**************************
.. toctree::
   :maxdepth: 2
   :caption: Contents:

Visualize Functions
===================
.. automodule:: src.visualize
   :members:

RPE Functions
=============
.. automodule:: src.rpe_calc
   :members:

ATE Functions
=============
.. automodule:: src.ate_calc
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
