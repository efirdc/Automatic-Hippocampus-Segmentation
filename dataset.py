import torch
import torchio as tio
from torch.utils.data import Dataset
import json
import os

class SubjectFolder(Dataset):
    def __init__(self, path, images=None, labels=None, transforms=None):
        self.transforms = transforms
        self.subjects = []

        self.images = images
        self.labels = labels

        self.subject_folder_names = os.listdir(path)
        self.subject_folders = [f"{path}/{folder}/" for folder in self.subject_folder_names]
        for subject_folder in self.subject_folders:
            subject_files = os.listdir(subject_folder)
            subject_data = {}

            attributes_file = "attributes.json"
            if attributes_file in subject_files:
                with open(f"{subject_folder}/{attributes_file}") as f:
                    subject_data = json.load(f)
                subject_files.remove(attributes_file)

            file_map = {file[:file.find(".")]: file for file in subject_files}

            missing_name = False
            all_names = []
            if images is not None:
                all_names += images
            if labels is not None:
                all_names += labels
            for name in all_names:
                if name not in file_map:
                    missing_name = True
            if missing_name:
                continue
            if images is not None:
                for name in images:
                    subject_data[name] = tio.ScalarImage(subject_folder + file_map[name])
            if labels is not None:
                for name in labels:
                    subject_data[name] = tio.LabelMap(subject_folder + file_map[name])

            self.subjects.append(tio.Subject(**subject_data))
        self.subject_dataset = tio.SubjectsDataset(self.subjects, transform=transforms)

    def __len__(self):
        return len(self.subject_dataset)

    def __getitem__(self, i):
        return self.subject_dataset[i], 0


class HippoDataset(Dataset):
    def __init__(self, path, images=("mean_dwi", "md", "fa"), transforms=None, mode="stack", transpose=True,
                 flip=False):
        self.mode = mode
        self.images = images
        self.transpose = transpose
        self.flip = flip
        self.subjects_dataset = SubjectFolder(path, images=images, transforms=transforms)

    def __len__(self):
        multiplier = 2 if self.flip else 1
        if self.mode == "stack":
            return len(self.subjects_dataset) * multiplier
        elif self.mode == "split":
            return len(self.subjects_dataset) * len(self.images) * multiplier

    def __getitem__(self, i):
        if self.mode == "stack":
            out = []
            for name in self.images:
                subject, attrib = self.subjects_dataset[i // 2]
                x = subject[name].data
                x[x.isnan()] = 0.
                out.append(x)
            out = torch.stack(out, dim=1)
        elif self.mode == "split":
            instance_id = i // len(self.images)
            channel_id = i % len(self.images)
            name = self.images[channel_id]
            subject, attrib = self.subjects_dataset[instance_id]
            out = subject[name].data
            out[out.isnan()] = 0.
        if self.transpose:
            out = out.transpose(1, 2)
        if self.flip:
            H_half = out.shape[2] // 2
            if i % 2 == 0:
                out = out[:, :, H_half:]
            else:
                out = torch.flip(out[:, :, :H_half], dims=(2,))
        return out, 0