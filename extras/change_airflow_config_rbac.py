import configparser
configParser = configparser.RawConfigParser()
configFilePath = r'airflow/airflow.cfg'
configParser.read(configFilePath)
if configParser.get('webserver','rbac') == 'False':
    configParser.set('webserver','rbac','True')
configParser.set('core','load_examples','False')
with open('airflow/airflow.cfg', 'w') as configfile:
    configParser.write(configfile)
