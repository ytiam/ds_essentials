cd ~/
conda create --prefix non_rbac_airflow_env python=3.7
source activate non_rbac_airflow_env/
cd non_rbac_airflow_env/
pip install apache-airflow
pip install flask-bcrypt
pip uninstall Flask-AppBuilder
pip install Flask-AppBuilder==2.3.2
export AIRFLOW_HOME=~/non_rbac_airflow_env/airflow/
airflow initdb
python ~/GIT/ds_essentials/extras/change_airflow_config.py
python ~/GIT/ds_essentials/extras/create_admin_user.py
cp ~/GIT/ds_essentials/extras/start_airflow .
