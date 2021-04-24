
# author: Atanu Maity

import sys

def find_location_using_pip(pckg):
    import subprocess
    # running bash command using subprocess
    output = subprocess.check_output(['pip', 'show', pckg])
    # splitting outpout with newline to get list of returns
    output_splitted = output.decode("utf-8").split('\n')[:-1]
    output_dict = dict(i.split(": ") for i in output_splitted)[
        'Location']  # list converting to dict
    return output_dict+'/'+pckg+'/'

pckg = sys.argv[1]
print(find_location_using_pip(pckg))