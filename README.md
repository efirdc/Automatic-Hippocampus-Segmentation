# Automatic-Hippocampus-Segmentation

### Results
Ab300 Subject 009
Manual                     | Automatic
:-------------------------:|:-------------------------:
<img src="./results/ab300_009/whole_gt.png" width="250">  |  <img src="./results/ab300_009/whole_pred.png" width="250">
<img src="./results/ab300_009/hbt_gt.png" width="250">  |  <img src="./results/ab300_009/hbt_pred.png" width="250">

### Setup
This segmentation package was tested with python 3.7. Other python 3 versions may work as well.  
Several python packages are required. You can install them using the `requirements.txt` file included in the repository.  
Just run the following command: `pip install -r /path/to/requirements.txt`.  

Note: Depending on how your system is configured, you may have to specify `pip3` and `python3` to use python 3 instead of python 2.

### Dataset
The current implementation requires that the dataset is organized into a certain folder structure.  
All subjects must have their own folder:
```
subjects/subject_a/
subjects/subject_b/
subjects/any_folder_name_is_valid/
```
and the niftiis for the mean DWI, FA, and MD must be present in each folder

```
/subject_a/mean_dwi.*
/subject_a/md.*
/subject_a/fa.*
```
The file name must match exactly, but the extension does not matter.

### Running

```
usage: run_segmentation.py [-h] [--device DEVICE] [--out_folder OUT_FOLDER]
                           [--keep_isolated_components] [--keep_holes]
                           [--lateral_uniformity] [--output_raw_probabilities]
                           model_path dataset_path output_filename

positional arguments:
  model_path            Path to the model.
  dataset_path          Path to the subjects data folders.
  output_filename       File name for segmentation output. Can specify .nii or
                        .nii.gz if compression is desired.

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       PyTorch device to use. Set to 'cpu' if there are
                        issues with gpu usage. A specific gpu can be selected
                        using 'cuda:0' or 'cuda:1' on a multi-gpu machine.
  --out_folder OUT_FOLDER
                        Redirect all output to a folder. Otherwise, the output
                        will be placed in each subjects folder.
  --keep_isolated_components
                        Don't remove isolated components in the post
                        processing pipeline. (on by default)
  --keep_holes          Don't remove holes in the post processing pipeline.
                        (on by default)
  --lateral_uniformity  Make HBT ROIs uniform on the lateral axis.
  --output_raw_probabilities
                        Output the raw probabilties from the network instead
                        of converting them to a segmentation map

```
Example usage:
```bash
python run_segmentation.py "E:/models/whole_model.pt" "E:/Datasets/Diffusion_MRI/Subjects/" whole_pred.nii.gz
```

```bash
python run_segmentation.py "E:/models/hbt_model.pt" "E:/Datasets/Diffusion_MRI/Subjects/" hbt_pred.nii.gz --out_folder "E:/Datasets/Diffusion_MRI/HBT_Predictions/"
```
