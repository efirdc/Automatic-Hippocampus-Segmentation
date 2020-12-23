import argparse
import torch
import torchio as tio
from tqdm import tqdm
import os

from post_processing import *

from context import Context, context_globals
from models import NestedResUNet6
from dataset import HippoDataset
context_globals.update(globals())

def hippo_inference(context, args, i, log_callback=None):
    subject_name = context.dataset.subjects_dataset.subject_folder_names[i]
    log_text = f"subject {subject_name}: "
    log = False

    inverse_transforms = tio.Compose([
        tio.Crop((0, 0, 0, 0, 2, 2)),
        tio.Pad((62, 62, 70, 58, 0, 0)),
    ])
    with torch.no_grad():
        left_side_prob = context.model(context.dataset[i*2][0].to(context.device))[0]
        right_side_prob = context.model(context.dataset[i*2 + 1][0].to(context.device))[0]
    if args.output_probabilities:
        right_side_prob = torch.flip(right_side_prob, dims=(1,))
        out = torch.cat((right_side_prob, left_side_prob), dim=1)
        out = out.cpu()
        out = inverse_transforms(out)
        return out

    left_side = torch.argmax(left_side_prob, dim=0)
    right_side = torch.argmax(right_side_prob, dim=0)

    if args.lateral_uniformity:
        left_side, left_removed_count = lateral_uniformity(left_side, left_side_prob, return_counts=True)
        right_side, right_removed_count = lateral_uniformity(right_side, right_side_prob, return_counts=True)
        total_removed = left_removed_count + right_removed_count
        if total_removed > 0:
            log_text += f" Changed {total_removed} voxels to enforce lateral uniformity."


    left_side[left_side != 0] += torch.max(right_side)
    right_side = torch.flip(right_side, dims=(0,))
    out = torch.cat((right_side, left_side), dim=0)

    out = out.cpu().numpy()

    if args.remove_isolated_components:
        num_components = out.max()
        out, components_removed, component_voxels_removed = keep_components(out, num_components, return_counts=True)
        if component_voxels_removed > 0:
            log_text += f" Removed {component_voxels_removed} voxels from " \
                        f"{components_removed} detected isolated components."
            log = True
    if args.remove_holes:
        out, hole_voxels_removed = remove_holes(out, hole_size=64, return_counts=True)
        if hole_voxels_removed > 0:
            log_text += f" Filled {hole_voxels_removed} voxels from detected holes."
            log = True
    if log:
        log_callback(log_text)

    out = torch.from_numpy(out).unsqueeze(0)
    out = inverse_transforms(out)

    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Hippocampus Segmentation")
    parser.add_argument("model_path", type=str, help="Path to the model.")
    parser.add_argument("dataset_path", type=str, help="Path to the subjects data folders.")
    parser.add_argument("output_filename", type=str, help="File name for segmentation output. "
                        "Can specify .nii or .nii.gz if compression is desired.")
    parser.add_argument("--device", type=str, default="cuda",
        help="PyTorch device to use. Set to 'cpu' if there are issues with gpu usage. A specific gpu can be selected"
            " using 'cuda:0' or 'cuda:1' on a multi-gpu machine."
    )
    parser.add_argument("--out_folder", type=str, default="",
         help="Redirect all output to a folder. Otherwise, the output will be placed in each subjects folder."
    )
    parser.add_argument('--keep_isolated_components', dest='remove_isolated_components', action='store_false',
         help="Don't remove isolated components in the post processing pipeline. (on by default)"
    )
    parser.set_defaults(remove_isolated_components=True)
    parser.add_argument('--keep_holes', dest='remove_holes', action='store_false',
        help="Don't remove holes in the post processing pipeline. (on by default)"
    )
    parser.set_defaults(remove_holes=True)
    parser.add_argument('--lateral_uniformity', dest='lateral_uniformity', action='store_true',
        help="Make HBT ROIs uniform on the lateral axis."
    )
    parser.set_defaults(lateral_uniformity=False)
    parser.add_argument('--output_raw_probabilities', dest='output_probabilities', action='store_true',
        help="Output the raw probabilties from the network instead of converting them to a segmentation map"
    )
    parser.set_defaults(output_probabilities=False)
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            device = torch.device("cpu")
            print("cuda not available, switched to cpu")
    else:
        device = torch.device(args.device)
    print("using device", device)

    context = Context(device, file_name=args.model_path, variables=dict(DATASET_FOLDER=args.dataset_path))

    # Fix torchio deprecating something...
    fixed_transform = tio.Compose([
        tio.RescaleIntensity((-1, 1), (0.5, 99.5)),
        tio.Crop((62, 62, 70, 58, 0, 0)),
        tio.Pad((0, 0, 0, 0, 2, 2)),
        tio.ZNormalization(),
    ])
    context.dataset.subjects_dataset.subject_dataset.set_transform(fixed_transform)

    if args.out_folder != "" and not os.path.exists(args.out_folder):
        print(args.out_folder, "does not exist. Creating it.")
        os.makedirs(args.out_folder)

    total = len(context.dataset) // 2
    pbar = tqdm(total=total)
    context.model.eval()
    for i in range(total):
        out_folder = args.out_folder
        if out_folder == "":
            out_folder = context.dataset.subjects_dataset.subject_folders[i]
        else:
            out_folder += context.dataset.subjects_dataset.subject_folder_names[i] + "_"
        out = hippo_inference(context, args, i, log_callback=pbar.write)
        if args.output_probabilities:
            image = tio.ScalarImage(tensor=out)
            image.save(out_folder + args.output_filename)
        else:
            out = out.int()
            label_map = tio.LabelMap(tensor=out)
            label_map.save(out_folder + args.output_filename)
        pbar.update(1)


