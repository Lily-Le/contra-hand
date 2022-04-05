import json
import os
import numpy as np
def json_load(file_name):
    with open(file_name, 'r') as fi:
        data = json.load(fi)
    return data

base_path='/home/d3-ai/cll/HanCo_tester'

inx=[]
meta_info = json_load(os.path.join(base_path, 'meta.json'))
#%%
# pos=meta_info['is_train'][meta_info['is_train']==True].index()

ist=meta_info['is_train']
for i in range(len(ist)):
    subset=(ist[i])
    inx.extend([[i,j,0,0] for j, x in enumerate(subset) if x == True])
with open('index_mv_unsup_weak.json','w') as f:
    json.dump(inx,f)
f.close()