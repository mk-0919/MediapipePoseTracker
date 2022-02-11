import configparser

import os
import errno

import math
from re import X

#iniファイルの生成、書き込み

"""config = configparser.ConfigParser()
config['PARAMS'] = {'distance_per_pixel':'0.05'}
with open('Config.ini','w') as configfile:
    config.write(configfile)

#iniファイルの読み込み

config_ini = configparser.ConfigParser()
config_ini_path = 'config.ini'

#指定したiniファイルが存在しない場合エラーを返す
if not os.path.exists(config_ini_path):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)

config_ini.read(config_ini_path, encoding='utf-8')

#intで値取得
distance_per_pixel = config_ini.getfloat('PARAMS','distance_per_pixel')
print(str(distance_per_pixel),type(distance_per_pixel))"""
def main():
    list1 = [1,2,3]
    list2 = [4,5,6]
    result = [n / 2 for n in [x + y for (x, y) in zip(list1, list2)]]
    print(result)

def returnTrue():
    return True

def returnFalse():
    return False

if __name__ == '__main__':
    main()