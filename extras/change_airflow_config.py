import configparser
configParser = configparser.RawConfigParser()
configFilePath = r'airflow/airflow.cfg'
configParser.read(configFilePath)
if configParser.get('webserver','rbac') == 'True':
    configParser.set('webserver','rbac','False')
configParser.set('webserver','authenticate','True')
configParser.set('webserver','auth_backend','airflow.contrib.auth.backends.password_auth')
configParser.set('core','load_examples','False')
with open('airflow/airflow.cfg', 'w') as configfile:
    configParser.write(configfile)
