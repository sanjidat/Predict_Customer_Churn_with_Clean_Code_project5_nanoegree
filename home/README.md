# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
We are provided us with a working ml project churn_notebook.ipynb and the goal of the project is to create a library file and a test file.
A sophisticated is presented to run this project. 
***python 3.6 is used.

## Files and data description
Check the root directory for getting the overview of the project. All the files are included to this directory. 

## Running Files
How do you run your files? What should happen when you run your files?

Necessary libraries needs to be included at the beginning of the project. Use the "requirements_py3.6.txt" or the "requirements_py3.8.txt" for installing dependencies depending on your python version.

python -m pip install -r requirements_py3.6.txt
python -m pip install -r requirements_py3.8.txt

In this project, three files are refactored. They are:
1. churn_library.py:
It is a library of functions to find customers who are liekly to churn for running the files.

2. churn_script_logging_and_tests.py:
This file contains unit tests for the churn_library.py functions. 

3. README.md:
This file provides an overview of the project.

It is necessary to log all the error and info messages. So running the command below in the terminal should test each of the functions and provide any errors to a file stored in the /logs folder.

ipython churn_script_logging_and_tests.py

To format the refactored code using PEP8-Style Guide run the commands below:

autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py

Use Pylint for the code analysis by running the commands below:

pylint churn_library.py
Your code has been rated at 8.53/10

pylint churn_script_logging_and_tests.py
Your code has been rated at 9.73/10




