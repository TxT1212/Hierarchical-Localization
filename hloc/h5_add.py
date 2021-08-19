import h5py
import numpy as np

# write example files
# -------------------

for name in ['1.hdf5', '2.hdf5']:

    data = h5py.File(name,'w')
    data['A'] = np.arange(25).reshape(5,5)
    data.close()

# support function
# ----------------

def getdatasets(key,archive):

    if key[-1] != '/': key += '/'

    out = []

    for name in archive[key]:

        path = key + name

        if isinstance(archive[path], h5py.Dataset):
            out += [path]
        else:
             out += getdatasets(path,archive)

    return out

# perform copying
# ---------------

# open both source-files and the destination
data1        = h5py.File('1.hdf5'    ,'r')
data2        = h5py.File('2.hdf5'    ,'r')
new_data = h5py.File('new.hdf5','w')

# get datasets
datasets    = sorted(getdatasets('/', data1))
datasets2 = sorted(getdatasets('/', data2))

# check consistency of datasets
# - number
if len(datasets) != len(datasets2):
    raise IOError('files not consistent')
# - item-by-item
for a,b in zip(datasets, datasets2):
    if a != b:
        raise IOError('files not consistent')

# get the group-names from the lists of datasets
groups = list(set([i[::-1].split('/',1)[1][::-1] for i in datasets]))
groups = [i for i in groups if len(i)>0]

# sort groups based on depth
idx        = np.argsort(np.array([len(i.split('/')) for i in groups]))
groups = [groups[i] for i in idx]

# create all groups that contain a dataset
for group in groups:
    new_data.create_group(group)

# copy (add) datasets
for path in datasets:

    # - get group name
    group = path[::-1].split('/',1)[1][::-1]

    # - minimum group name
    if len(group) == 0: group = '/'

    # - copy data
    new_data[path] = data1[path][...] + data2[path][...]

# verify
# ------

# copy (add) datasets
for path in datasets:
    print(new_data[path][...])

# close all files
# ---------------

new_data.close()
data1.close()
data2.close()