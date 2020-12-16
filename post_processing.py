from skimage.morphology import label, remove_small_holes, dilation
import torch
import numpy as np


def unsort_by_size(img, sorted_labels):
    out_img = img.copy()
    for i in range(sorted_labels.shape[0]):
        out_img[img == i] = sorted_labels[i]
    return out_img


def sort_by_size(img, descending=False, return_sorted_labels=False):
    out_img = img.copy()
    unique_labels, unique_counts = np.unique(img, return_counts=True)

    ids = np.argsort(unique_counts)
    if descending:
        ids = ids[::-1]
    unique_labels = unique_labels[ids]

    for i in range(ids.shape[0]):
        out_img[img == unique_labels[i]] = i

    if return_sorted_labels:
        return out_img, unique_labels
    return out_img


def keep_components(img, num, max_dilations=100, return_counts=False):
    img = img.copy()
    num_components_removed = num_elements_removed = 0
    for i in range(max_dilations):
        img_comp = label(img)
        img_comp_sorted = sort_by_size(img_comp, descending=True)
        keep = img_comp_sorted <= num
        remove = ~keep
        if i == 0:
            num_elements_removed = remove.sum()
            num_components_removed = img_comp_sorted.max() - num
        if remove.sum() == 0:
            break
        sorted_img, sorted_labels = sort_by_size(img, return_sorted_labels=True)
        to_dilate = sorted_img * keep
        dilated = dilation(to_dilate)
        change = (dilated != to_dilate) & remove
        sorted_img[change] = dilated[change]
        img = unsort_by_size(sorted_img, sorted_labels)

    if return_counts:
        return img, num_components_removed, num_elements_removed
    return img

def remove_holes(img, hole_size, max_dilations=100, return_counts=False):
    img = img.copy()
    total_holes = 0

    for i in range(max_dilations):
        mask = img > 0
        small_holes = ~mask & remove_small_holes(mask, hole_size)
        num_holes = small_holes.sum()
        if i == 0:
            total_holes = num_holes
        if num_holes == 0:
            break
        img[small_holes] = dilation(img)[small_holes]

    if return_counts:
        return img, total_holes

    return img

def lateral_uniformity(seg, prob, return_counts=False):
    seg = seg.clone()
    prob_non_background = prob[1:]
    lateral_probability_sum = prob_non_background.sum(dim=1, keepdim=True)
    lateral_seg = torch.argmax(lateral_probability_sum, dim=0).repeat(seg.shape[0], 1, 1) + 1
    seg_mask = seg > 0
    count = 0
    if return_counts:
        count = (seg[seg_mask] != lateral_seg[seg_mask]).sum()
    seg[seg_mask] = lateral_seg[seg_mask]

    if return_counts:
        return seg, count
    return seg

