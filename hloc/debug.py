import h5py


# with h5py.File(str("/home/ezxr/Downloads/Hierarchical-Localization/outputs/aachen/feats-superpoint-n4096-r1024.h5"), 'r') as fd:
#     # print(fd.keys())
#     for key in fd.keys():
#         # print(fd[key].keys())
#         print(key)


# db/1045.jpg
# outputs/wxc/feats-superpoint-n4096-r1024.h5
import pickle
with open('/home/ezxr/Downloads/Hierarchical-Localization/outputs/ibl/hloc_superpoint+superglue_netvlad20.txt_logs.pkl', 'rb') as f:
    data = pickle.load(f)
    # for i in data:
    print(data['loc'])
