from pathlib import Path
from pprint import pformat
import argparse

from ... import extract_features, match_features
from ... import pairs_from_covisibility, pairs_from_retrieval
from ... import colmap_from_nvm, triangulation, localize_sfm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=Path, default='/media/txt/data2/naver/HyundaiDepartmentStore/1F/release/',
                    help='Path to the dataset, default: %(default)s')
parser.add_argument('--outputs', type=Path, default='/media/txt/data2/naver/HyundaiDepartmentStore/outputs/1F/',
                    help='Path to the output directory, default: %(default)s')
parser.add_argument('--num_covis', type=int, default=10,
                    help='Number of image pairs for SfM, default: %(default)s')
parser.add_argument('--num_loc', type=int, default=50,
                    help='Number of image pairs for loc, default: %(default)s')
args = parser.parse_args()

# Setup the paths
dataset = args.dataset
images = dataset / 'mapping/sensors/records_data'
query = dataset / 'test/sensors/records_data'
outputs = args.outputs  # where everything will be saved
sift_sfm = outputs / 'sfm_sift'  # from which we extract the reference poses
reference_sfm = outputs / 'sfm_superpoint+superglue'  # the SfM model we will build
sfm_pairs = outputs / f'pairs-db-covis{args.num_covis}.txt'  # top-k most covisible in SIFT model
loc_pairs = outputs / f'pairs-query-netvlad{args.num_loc}.txt'  # top-k retrieved by NetVLAD
results = outputs / f'hloc_superpoint+superglue_netvlad{args.num_loc}.txt'

# list the standard configurations available
print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

# pick one of the configurations for extraction and matching
retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['superpoint_inloc']
matcher_conf = match_features.confs['superglue']

features = extract_features.main(feature_conf, images, outputs)
features = extract_features.main(feature_conf, query, outputs)
print("*****features: ", features)
sift_sfm = '/media/txt/data2/naver/HyundaiDepartmentStore/1F/release/sfm_sift'
# pairs_from_covisibility.main(
#     sift_sfm, sfm_pairs, num_matched=args.num_covis)
sfm_matches = match_features.main(
    matcher_conf, sfm_pairs, feature_conf['output'], outputs)

triangulation.main(
    reference_sfm,
    sift_sfm,
    images,
    sfm_pairs,
    features,
    sfm_matches,
    colmap_path='colmap')

global_descriptors = extract_features.main(retrieval_conf, images, outputs)
print(global_descriptors)
global_descriptors = extract_features.main(retrieval_conf, query, outputs)
print(global_descriptors)

pairs_from_retrieval.main(
    global_descriptors, loc_pairs, args.num_loc,
    query_prefix='2019-08-21_12-10-13', db_model=reference_sfm)
loc_matches = match_features.main(
    matcher_conf, loc_pairs, feature_conf['output'], outputs)

# localize_sfm.main(
#     reference_sfm,
#     # dataset / 'queries/*',
#     dataset / 'query/*_time_queries_with_intrinsics.txt',
#     loc_pairs,
#     features,
#     loc_matches,
#     results,
#     covisibility_clustering=False)  # not required with SuperPoint+SuperGlue

# from hloc import visualization
# visualization.visualize_loc(
#     results, images, reference_sfm, n=1, top_k_db=1, prefix='query/night', seed=2)