# cse6242_project
DESCRIPTION- Our Application consists of a Tableau file for our user interaction and visualization and a python application running TabPy used to run our college recommendation algorithm based on user inputs. The Tableau application runs a script to trigger the recommendation algorithm on out TabPy server and stores the results in a calculated column.

INSTALLATION- Our application will require you to have python installed on your computer (https://www.python.org/downloads/). You will then have to use pip to install the following packages by running the following command in your command prompt for each of the following packages, "pip install <package name here>", the packages are : TabPy, NumPy, scikit-learn, geopy, and pandas

You will also need to have git installed (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to pull in the code from our GitHub repository: https://github.com/afshanirani/cse6242_project . Run "git clone https://github.com/afshanirani/cse6242_project" in the folder where you would like to store the code for the project.

Lastly, you will need Tableau Desktop installed which you can use the license granted for us to use in this course.

EXECUTION-
Running the backend:
1. Open up a command prompt window and run the command TabPy
2. Now Navigate the the folder you cloned the GitHub Project into and run the file tabpy_app/tabpyBackend either from the command prompt using the command "python <path to tabPyBackend.py folder" or from your favorite IDE
3. Now open the Tableau file from the cloned GitHub Repository and navigate to Help > Settings and Performance > Manage Analytics Extension Connection. Click the test connection button and verify the connection succeeded.
4. Now you should be able to use the Tableau Visualizations with no errors.
