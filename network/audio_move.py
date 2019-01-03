import os
import glob
import shutil
import random
specs = glob.glob('F:\\视听数据\\dataset\\audios_spec\\solo\\*\\*\\*.png')
n_specs = len(specs)
n_test = 1000
'''
    for separate train set and test set
'''
for m in range(n_test):
    n = random.randint(0, n_specs-1)
    spec = specs[n]
    specs.pop(n)
    folderpath = 'F:\\视听数据\\dataset\\audios_spec_test\\' + spec.split('\\')[5]
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    dest = folderpath + '\\' + spec.split('\\')[6] + \
           spec.split('\\')[7]
    shutil.copyfile(spec, dest)
    n_specs -= 1

for spec in specs:
    folderpath = 'F:\\视听数据\\dataset\\audios_spec_train\\' + spec.split('\\')[5]
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    dest = folderpath + '\\' + spec.split('\\')[6] + \
           spec.split('\\')[7]
    shutil.copyfile(spec, dest)
