project=$1
apt-get install libpq-dev
apt-get install gcc
apt-get install python3-dev
apt-get install awscli

mkdir ~/Ayata
mkdir ~/Ayata/$project
conda env create --prefix ~/Ayata/$project --file conda_env/environment.yml
cd
cd Ayata/
cwd=`echo $PWD`
echo $cwd
source activate $project/
cd $project

pip install environment_kernels
pip install ipyparallel
pip install libsousou

cookiecutter https://github.com/drivendata/cookiecutter-data-science
created_repo=`ls -td -- */ | head -n 1`
cd $created_repo

mkdir git
mkdir prediction
mkdir airflow
mkdir airflow/dags
mkdir airflow/logs

pip uninstall Flask-AppBuilder
pip install Flask-AppBuilder==2.3.2

export AIRFLOW_HOME=~/Ayata/$project/$created_repo/airflow/
airflow initdb
python /GIT/ds_essentials/extras/change_airflow_config_rbac.py
bash /GIT/ds_essentials/extras/create_rbac_admin_user.sh
cp /GIT/ds_essentials/extras/start_airflow_dev .

mkdir src/models/config
cd
