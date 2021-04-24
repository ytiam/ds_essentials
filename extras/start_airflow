port=$1
cwd=`echo $PWD`
_cwd=$cwd/airflow
export AIRFLOW_HOME=$_cwd
airflow initdb
airflow webserver -p $port -D
airflow scheduler -D
