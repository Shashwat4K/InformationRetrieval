import pandas as pd 
import csv
import json 

def print_dict(d):
    for i, j in d.items():
        print('{}->{}'.format(i, j))

_dict1 = {'shashwat': [1,2,3,4,5], 'vnit': [6,7,8,9,10], 'nagpur': [11,15,18,21], 'maharashtra': [12,13,14,16,17,19,20]}
_dict2 = {'abc': [1,2,3,4,7,8], 'pqr':[5,6,9,10,11], 'xyz': [12,13,14,15]}
_dict3 = {'lmn':[1,3,5,7], 'ijk':[2,4,6,8]}
grand_dict = {'1': _dict1, '2': _dict2, '3': _dict3}
with open('random.json', 'w') as fp:
    json.dump(grand_dict, fp)

df = pd.read_json('random.json')
x = pd.isnull(df[1])
d = dict(df[df[1].isnull() == False][1])

print_dict(d)