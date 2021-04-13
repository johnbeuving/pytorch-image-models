""" A dataset parser that reads images from folders

Folders are scannerd recursively to find image files. Labels are based
on the folder hierarchy, just leaf folders by default.

Hacked together by / Copyright 2020 Ross Wightman
"""
import os
import pandas as pd

from timm.utils.misc import natural_key

from .parser import Parser

from .constants import IMG_EXTENSIONS


def load_classmap(filename="classmap.csv"):
    df_cats = pd.read_csv(filename)
    class_to_idx = {}
    for idx, row in df_cats.iterrows():
        cat_id = row["category_id"]
        class_to_idx[str(cat_id)] = int(row["id"])
    return class_to_idx

def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    df = pd.read_csv(folder)
    for idx, row in df.iterrows():
        filenames.append("test_bbox256x256/" + row["image_id"] + ".jpg")
        cat_id = row["category_id"]
        labels.append(str(cat_id))

    if class_to_idx is None:
        class_to_idx = load_classmap()
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


class ParserImageCSV(Parser):
    def __init__(
            self,
            root,
            name,
            split,
            class_map=''):
        super().__init__()

        print(root,name,split)
        class_to_idx = None
        self.samples, self.class_to_idx = find_images_and_targets(split, class_to_idx=class_to_idx)

        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename
