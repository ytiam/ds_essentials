Folder contains 3 scripts. 3 py scripts and 1 shell script

essential.py - this is the script where you can add new functions which you want to use in your code directly. 
start.py - this is the script where you can add all the libraries which you need on your daily work. the libraries noted here
            will be loaded automatically on a startup of a new ipython notebook. No need to load them manually everytime
module_finding_from_scripts.py - The purpose of this script is to find out the uninstalled pip packages, which are required for successful 			run of all the py scripts in the current folder path, and write those packages into a 'requirements.txt' file automatically.

initial_setup.sh - This script you need to run, everytime you update or add new lines in any of the py scripts. 


### First time setup ####
Once you clone the repo from git first time, To run all the scripts pinned in this repo, there should be all the necessary packages installed, but don't have manually check and find out which packages we need to install explicitly. Running the module_finding_from_scripts.py will automatically do that.

Run: python module_finding_from_scripts.py

Outcome: One requirement.txt will be created in the script directory, containing all the uninstalled packages list.

To install all the necessary uninstalled packages,

Run: pip install -r requirements.txt

After the installation is successfull, you need to run "bash initial_setup.sh" first, and then only the py scripts will be functionable and usable.
