import numpy as np
import cv2
import os
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    return abs_rel, rmse, a1


def evaluate_depth(model_path, dataset, source_path):
    full_dict = {}
    per_view_dict = {}
    test_dir = Path(model_path) / "test"

    for method in os.listdir(test_dir):
        print("Method:", method)
        errors = []
        full_dict[method] = {}
        per_view_dict[method] = {}
        pred_depth_root = test_dir / method / "depths"
        gt_depth_root = Path(source_path) / "depths"

        with open(os.path.join(source_path, 'val_cams.json'), 'r') as f:
            val_cams = json.load(f)

        pred_depth_paths = sorted([x for x in os.listdir(pred_depth_root) if x.endswith('npy')])
        if dataset == 'ETH3D':
            gt_depth_paths = sorted([x for x in os.listdir(gt_depth_root) if x.endswith('npy')])
        elif 'scannet' in dataset:
            gt_depth_paths = sorted([x for x in os.listdir(gt_depth_root) if x.endswith('png')])
        elif 'blended' in dataset:
            gt_depth_paths = sorted([x for x in os.listdir(gt_depth_root) if x.endswith('pfm')])
        else:
            raise NotImplementedError("Dataset not supported")

        
        for i, idx in enumerate(val_cams):
            if dataset == 'ETH3D':
                with open(os.path.join(gt_depth_root, f'{gt_depth_paths[idx]}'), 'rb') as f:
                    gt_depth = np.load(f)
            elif 'scannet' in dataset:
                gt_depth = cv2.imread(os.path.join(gt_depth_root, f'{gt_depth_paths[idx]}'), cv2.IMREAD_UNCHANGED) / 1000
            elif 'blended' in dataset:
                gt_depth = cv2.imread(os.path.join(gt_depth_root, f'{gt_depth_paths[idx]}'), cv2.IMREAD_UNCHANGED)
            else:
                raise NotImplementedError("Dataset not supported")
            with open(os.path.join(pred_depth_root, pred_depth_paths[i]), 'rb') as f:
                pred_depth = np.load(f).astype(np.float64)
            
            pred_depth = cv2.resize(pred_depth, gt_depth.shape[::-1])
            mask1 = gt_depth != np.inf
            mask2 = gt_depth > 0
            mask = mask1 & mask2
            gt_masked_depth = gt_depth[mask]
            pred_masked_depth = pred_depth[mask]
            view_errors = compute_errors(gt_masked_depth, pred_masked_depth)
            errors.append(view_errors)
            per_view_dict[method][pred_depth_paths[i]] = {
                'abs_rel': view_errors[0],
                'rmse': view_errors[1],
                'a1': view_errors[2]
            }
            
        errors = np.array(errors).mean(0)
        full_dict[method] = {
            'abs_rel': errors[0],
            'rmse': errors[1],
            'a1': errors[2]
        }
    with open(os.path.join(model_path, "results_depth.json"), 'w') as fp:
        json.dump(full_dict, fp, indent=True)
    with open(os.path.join(model_path, "per_view_depth.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

    

if __name__ == "__main__":
    parser = ArgumentParser(description="Depth evaluation script parameters")
    parser.add_argument('--model_paths', '-m', required=True, type=str, default="")
    parser.add_argument('--dataset', type=str, default="ETH3D")
    args = parser.parse_args()

    with open(os.path.join(args.model_paths, 'cfg_args')) as cfg_file:
        cfgfile_string = cfg_file.read()
    cfgfile = eval(cfgfile_string)

    evaluate_depth(args.model_paths, args.dataset, cfgfile.source_path)
