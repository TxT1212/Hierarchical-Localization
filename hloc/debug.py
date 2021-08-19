import h5py
import numpy as np
new_data = h5py.File('new.hdf5','w')
def getdatasets(key, archive):

    if key[-1] != '/': key += '/'

    out = []

    for name in archive[key]:

        path = key + name

        if isinstance(archive[path], h5py.Dataset):
            out += [path]
        else:
             out += getdatasets(path,archive)

    return out

with h5py.File(str("/media/txt/data2/naver/HyundaiDepartmentStore/outputs/feats-superpoint-n4096-r1600.h5"), 'r') as fd:
    print(fd)
    # for key in fd.keys():
    #     print(key)
    #     print(fd[key].keys())
    #     print(fd[key])
    # # fd.create_group(key)
    # new_data.create_group(key)
    datasets  = sorted(getdatasets('/', fd))
    # get the group-names from the lists of datasets
    groups = list(set([i[::-1].split('/',1)[1][::-1] for i in datasets]))
    groups = [i for i in groups if len(i)>0]

    # sort groups based on depth
    idx        = np.argsort(np.array([len(i.split('/')) for i in groups]))
    groups = [groups[i] for i in idx]
    for d in groups:
        print(d)


# db/1045.jpg
# outputs/wxc/feats-superpoint-n4096-r1024.h5
# import pickle
# with open('/home/ezxr/Downloads/Hierarchical-Localization/outputs/ibl/hloc_superpoint+superglue_netvlad20.txt_logs.pkl', 'rb') as f:
#     data = pickle.load(f)
#     # for i in data:
#     print(data['loc'])
