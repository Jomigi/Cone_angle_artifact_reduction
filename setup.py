import os

os.system('conda install -y msd_pytorch cudatoolkit=10.0 -c aahendriksen -c pytorch -c defaults -c conda-forge')
os.system('conda install -y natsort')
os.system('conda install -y matplotlib')
os.system('conda install -y pymongo')
os.system('conda install -y -c astra-toolbox/label/dev astra-toolbox')
os.system('pip install pynrrd')
os.system('pip install sacred')
