project=$1
bash extras/project_env_setup.sh $project
source activate ~/Ayata/$project/
cp start.py ~/.ipython/profile_default/startup/
site_path=`python extras/find_site_package_path.py`
cwd=`echo $PWD`
echo $cwd
echo $site_path
cp -R $cwd $site_path
bash extras/jupyter_notebook_extension.sh
python module_finding_from_scripts.py
cat requirements.txt | xargs -n 1 pip install
rm requirements.txt
