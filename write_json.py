#%%
import json
import os
import numpy as np
import re
#%% find img index for training 
def json_load(file_name):
    with open(file_name, 'r') as fi:
        data = json.load(fi)
    return data

base_path='/home/d3-ai/cll/HanCo_tester'

inx=[]
meta_info = json_load(os.path.join(base_path, 'meta.json'))
print(meta_info.keys())
#%%
# pos=meta_info['is_train'][meta_info['is_train']==True].index()

ist=meta_info['is_valid']
obj=meta_info['object_id']
for i in range(int(int(len(ist)))):
    subset=(ist[i])
    inx.extend([[i,j,0,0] for j, x in enumerate(subset) if x == True])
with open('index_mv_unsup_weak_vld.json','w') as f:
    json.dump(inx,f)
f.close()

#%%
isvld=meta_info['is_valid']
for i in range(int(len(ist))):
    subset_t=(ist[i])
    subset_v=isvld[i]
    inx.extend([[i,j,0,0] for j, (x,y) in enumerate(zip(subset_t,subset_v)) if ((x == True)&(y==True))])
with open('index_mv_unsup_weak_tv.json','w') as f:
    json.dump(inx,f)
f.close()
#%% Find index in the background
bg_path='/home/d3-ai/cll/HanCo/bg_new/'
bg_lst=os.listdir(bg_path)
with open('bg_inds.json','w') as f:
    json.dump(bg_lst,f)
f.close()
def list_all_files(rootdir,end_with):
    import os
    _files = []
	# 列出文件夹下所有的目录与文件
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
		# 构造路径
        path = os.path.join(rootdir, list[i])
		# 判断路径是否为文件目录或者文件
		# 如果是目录则继续递归
        if os.path.isdir(path):
            _files.extend(list_all_files(path,end_with))
        if os.path.isfile(path):
            #if path.endswith(end_with):
            if (re.search(end_with,path)):
                _files.append(path)
    return _files
# %%
