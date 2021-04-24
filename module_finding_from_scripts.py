
# coding: utf-8

# The purpose of this script is to find out the uninstalled pip packages, which are required for successfull run of all the py scripts in 
# the given folder path, and write those packages into a 'requirements.txt' file automatically.

import os
import sys
import importlib

def getModuleList(path):
    with open(path,'r') as f:
        l = f.read()
    lines = l.split('\n')
    pckg_list = []

    for ll in lines:
        if ll.startswith('import') or ll.startswith('from'):
            code_parts = ll.split(' ')[1]
            code_sub_parts = code_parts.split('.')[0]
            if (not code_sub_parts.startswith('__')) and (not code_sub_parts.endswith('__')) and len(code_sub_parts)!=0:
                pckg_list.append(code_sub_parts)
    pckg_list = list(set(pckg_list))
    return pckg_list


def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


path = os.getcwd() + '/'


def list_modules(path):
    all_files = getListOfFiles(path)

    all_files = [i for i in all_files if i.endswith('.py')]

    all_modules = []
    for mod in all_files:
        all_modules.extend(getModuleList(mod))
    
    all_modules = list(set(all_modules))
    
    installable_modules = []
    for m in all_modules:
        try:
            importlib.import_module(m)
        except:
            installable_modules.append(m)
    print(installable_modules)
    with open(path+'/requirements.txt', 'w') as f:
        for item in installable_modules:
            f.write("%s\n" % item)


list_modules(path)

