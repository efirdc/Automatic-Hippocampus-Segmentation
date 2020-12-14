import argparse
import torch
import torchio as tio
from tqdm import tqdm

from context import Context, context_globals
from models import NestedResUNet6
from dataset import HippoDataset
context_globals.update(globals())

def hippo_inference(context, i):
    inverse_transforms = tio.Compose([
        tio.Crop((0, 0, 0, 0, 2, 2)),
        tio.Pad((62, 62, 70, 58, 0, 0)),
    ])
    with torch.no_grad():
        left_side = context.model(context.dataset[i*2][0].to(context.device))[0]
        right_side = context.model(context.dataset[i*2 + 1][0].to(context.device))[0]
    left_side = torch.argmax(left_side, dim=0).unsqueeze(0)
    right_side = torch.argmax(right_side, dim=0).unsqueeze(0)
    right_side[right_side != 0] += torch.max(left_side)
    right_side = torch.flip(right_side, dims=(1,))
    out = torch.cat((right_side, left_side), dim=1)
    out = inverse_transforms(out.cpu())
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Hippocampus Segmentation")
    parser.add_argument("model_path", type=str, help="Path to the model.")
    parser.add_argument("dataset_path", type=str, help="Path to the subjects data folders.")
    parser.add_argument("output_filename", type=str, help="File name for segmentation output.")
    parser.add_argument("--device", type=str, default="cuda",
        help="PyTorch device to use. Set to 'cpu' if there are issues with gpu usage. A specific gpu can be selected"
                        "using 'cuda:0' or 'cuda:1' on a multi-gpu machine."
    )
    parser.add_argument("--out_folder", type=str, default="",
         help="Redirect all output to a folder. Otherwise, the output will be placed in the subjects folder."
    )
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

    for i in tqdm(range(len(context.dataset) // 2)):
        out_folder = args.out_folder
        if out_folder == "":
            out_folder = context.dataset.subject_dataset.subject_folders[i]
        else:
            out_folder += context.dataset.subjects_dataset.subject_folder_names[i] + "_"
        out_seg = hippo_inference(context, i)
        label_map = tio.LabelMap(tensor=out_seg)
        label_map.save(out_folder + args.output_filename)


