#!/bin/bash
# Install dependency packages required
pip uninstall notebook
pip install notebook==5.7.8
pip install autopep8
pip install qgrid
pip install mlflow
pip install apache-airflow
pip install papermill
pip install lux-api

# Install nbextension package
pip install jupyter_contrib_nbextensions
# Install nbextension js plugins
jupyter contrib nbextension install --user
# Getting path of jupyter_contrib_nbextensions
pckg_path=`python extras/find_loc_using_pip.py jupyter_contrib_nbextensions`
pckg_path=$pckg_path'nbextensions/'
echo $pckg_path
# Copy the 'setup' extension folder in the nbextensions path
cp -R extras/setup/ $pckg_path
jupyter contrib nbextension install --user
# Enabling different extensions for jupyter notebook
jupyter nbextension enable codefolding/main
jupyter nbextension enable setup/main
jupyter nbextension enable toc2/main
jupyter nbextension enable scratchpad/main
jupyter nbextension enable hinterland/hinterland
jupyter nbextension enable snippets/main
jupyter nbextension enable snippets_menu/main
jupyter nbextension enable code_prettify/autopep8
jupyter nbextension enable hide_input/main
jupyter nbextension install --user --py luxwidget
jupyter nbextension enable luxwidget --user --py
