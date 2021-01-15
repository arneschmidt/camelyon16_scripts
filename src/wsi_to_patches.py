import cv2
import glob
import argparse
import os
from shutil import copy2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import skimage.io
import openslide as sld
import multiprocessing
from PIL import Image, ImageDraw
from skimage import filters
from scipy import ndimage

# def contains_tissue(image):
#     image = np.array(image)
#     colour_threshold = 200
#     percentage_white_threshold = 0.8
#     blurr_threshold = 70
#
#     white = (255, 255, 255)
#     grey = (colour_threshold, colour_threshold, colour_threshold)
#     resolution = (512, 512)
#
#     mask = cv2.inRange(image, grey, white)
#     white_pixels = np.sum(mask==255)
#     not_white = (white_pixels / (resolution[0] * resolution[1]) < percentage_white_threshold)
#
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     fm = cv2.Laplacian(gray, cv2.CV_64F).var()
#     not_blurry = fm > blurr_threshold
#
#     # copy file if percentage of white is below threshold
#     if not_white and not_blurry:
#         return True
#     else: # else only copy for debug
#         # if not not_white:
#         #     print('Patch white!')
#         # elif not not_blurry:
#         #     print('Patch blurry!')
#         return False

def get_wsi_data_splits(image_dir, val_split):
    wsi_list_train_normal = np.array(glob.glob(str(image_dir) + "training/normal/*.tif"))
    wsi_list_train_tumor = np.array(glob.glob(str(image_dir) + "training/tumor/*.tif"))
    sample_ids_val_normal = np.random.randint(0, len(wsi_list_train_normal), size=int(len(wsi_list_train_normal)*val_split))
    sample_ids_val_tumor = np.random.randint(0, len(wsi_list_train_tumor), size=int(len(wsi_list_train_tumor)*val_split))
    wsi_data_split_lists = {}
    wsi_list_val_normal = np.array(wsi_list_train_normal)[sample_ids_val_normal]
    wsi_list_val_tumor = np.array(wsi_list_train_tumor)[sample_ids_val_tumor]
    wsi_data_split_lists['val'] = np.concatenate((wsi_list_val_normal, wsi_list_val_tumor))
    wsi_list_train_normal = np.delete(wsi_list_train_normal, sample_ids_val_normal, axis=0)
    wsi_list_train_tumor = np.delete(wsi_list_train_tumor, sample_ids_val_tumor, axis=0)
    wsi_data_split_lists['train'] = np.concatenate((wsi_list_train_normal, wsi_list_train_tumor))
    wsi_data_split_lists['test'] = np.array(glob.glob(str(image_dir) + "testing/images/*.tif"))
    test_wsi_df = pd.read_csv(str(image_dir) + "testing/reference.csv", header=None)
    return wsi_data_split_lists, test_wsi_df

def get_patch_class(patch_annotation):
    tumor_percent = np.sum(patch_annotation / 255) / patch_annotation.size
    if tumor_percent > 0.5:
        return 1
    else:
        return 0

def contains_tissue(patch, otsu_threshold, white_threshold=0.35, blurr_threshold=50, greyscale_threshold=0.1, debug=False):
    """

    :param patch:
    :param otsu_threshold:
    :param white_threshold: percentage that must NOT be white to return True
    :param blurr_threshold:
    :param black_threshold: percentage that must NOT be black to return True
    :param debug:
    :return:
    """
    patch_gray = np.asarray(patch.convert('LA'), dtype=np.int16)
    patch_rgb = np.asarray(patch, dtype=np.int16)
    patch_tissue = patch_gray[:, :, 0] < otsu_threshold
    patch_tissue = ndimage.binary_dilation(patch_tissue, iterations=2)
    patch_tissue_percent = np.sum(patch_tissue) / patch_tissue.size
    patch_white = patch_tissue_percent < white_threshold
    patch_coloured = np.sum(np.abs(patch_rgb[:, :, 0] - patch_gray[:,:,0]) > 20) / patch_tissue.size
    patch_greyscale = patch_coloured < greyscale_threshold

    fm = cv2.Laplacian(patch_gray, cv2.CV_64F).var()
    patch_blurry = fm < blurr_threshold
    reason = ''
    if patch_blurry:
        reason = reason + '-blurry'
    if patch_white:
        reason = reason + '-white'
    if patch_greyscale:
        reason = reason + '-greyscale' + str(patch_coloured)
    if not (patch_white or patch_blurry or patch_greyscale):
        return True
    elif debug:
        return reason
    else:
        return False

def get_otsu_threshold(wsi):
    thumb=wsi.get_thumbnail((5000,5000))
    thumb_gray = thumb.convert('LA')
    thumb_gray_array = np.asarray(thumb_gray)
    otsu_threshold = filters.threshold_otsu(thumb_gray_array[:, :, 0])

    return otsu_threshold

def create_wsi_df(wsi_lists, test_wsi_df):
    wsi_df = pd.DataFrame()
    all_wsi_list = np.concatenate([wsi_lists['train'],wsi_lists['val'], wsi_lists['test']] )
    wsi_df['slide'] = all_wsi_list
    wsi_df['N'] = 0
    wsi_df['P'] = 0
    for i in range(len(all_wsi_list)):
        wsi_name = os.path.basename(all_wsi_list[i]).split('.')[0]
        if 'normal' in wsi_name:
            wsi_df['N'].iloc[i] = 1
        elif 'tumor' in all_wsi_list[i]:
            wsi_df['P'].iloc[i] = 1
        else:
            test_class = test_wsi_df[test_wsi_df[0] == wsi_name][1][0]
            if test_class == 'Normal':
                wsi_df['N'].iloc[i] = 1
            else:
                wsi_df['P'].iloc[i] = 1
        wsi_name = wsi_name.split('_')[0] + wsi_name.split('_')[1]
        wsi_df['slide'].iloc[i] = wsi_name
    return wsi_df

def init_patch_df(existing_patch_df = 'None'):
    if existing_patch_df == 'None':
        df = pd.DataFrame(columns=['image_name', 'N', 'P', 'unlabeled'])
    else:
        df = pd.read_excel(existing_patch_df)
    return df

def slice_image(wsi_path, args, index, return_dict):
    resolution = args.patch_resolution
    overlap = args.patch_overlap
    output_dir = args.output_dir
    dataframes_only = args.dataframes_only
    debug = args.debug
    image_path = os.path.join(output_dir, 'patches')

    wsi = read_tiff(wsi_path)

    level = 1
    wsi_name = os.path.basename(wsi_path).split('.')[0]
    w, h = wsi.dimensions
    if overlap:
        num_patches_per_row = int(2*np.floor((h/resolution)) - 1)
        num_patches_per_column = int(2*np.floor((w/resolution)) - 1)
    else:
        num_patches_per_row = int(np.floor((h / resolution)))
        num_patches_per_column = int(np.floor((w / resolution)))
    otsu_threshold = get_otsu_threshold(wsi)

    positive_slide = ('tumor' in wsi_name)
    negative_slide = ('normal' in wsi_name)
    test_slide = ('test' in wsi_name)
    mask_too_big = False
    wsi_mask = None
    if positive_slide:
        try:
            mask_path = os.path.join(args.mask_dir, wsi_name+'_annotation_mask.png')
            # wsi_mask = sld.open_slide(mask_path)
            # wsi_mask = skimage.io.MultiImage(mask_path)[0]
            # wsi_mask = cv2.imread(mask_path)
            wsi_mask = np.asarray(Image.open(mask_path))
            # wsi_mask = np.ones(shape=(10000, 10000))*255
        except:
            raise Warning('Annotation mask to big to load. wsi_mask: ' + wsi_name+'_annotation_mask.png')
            mask_too_big = True
    # take underscore out of wsi name
    wsi_name = wsi_name.split('_')[0] + wsi_name.split('_')[1]

    names = []
    classes = []
    if not mask_too_big:
        patch_df = pd.DataFrame(columns=['image_name', 'N', 'P', 'unlabeled'])
        for row in range(num_patches_per_row):
            if row % 10 == 0:
                print('row ' + str(row) + ' of ' + str(num_patches_per_row))
            for column in range(num_patches_per_column):
                if overlap:
                    start_y = int(row * (resolution / 2))
                    start_x = int(column * (resolution / 2))
                else:
                    start_y = int(row*(resolution))
                    start_x = int(column*(resolution))
                patch = wsi.read_region((start_y, start_x), level, (resolution, resolution))
                patch = patch.convert("RGB")
                name = wsi_name + '_' + str(row) + '_' + str(column)
                if contains_tissue(patch, otsu_threshold):
                    names.append(name)
                    if not dataframes_only:
                        patch.save(os.path.join(image_path, name+ '.jpg'))
                    if positive_slide:
                        patch_class = get_patch_class(wsi_mask[start_x:start_x+resolution,start_y:start_y+resolution])
                        classes.append(patch_class)
                elif debug:
                    if contains_tissue(patch, otsu_threshold, white_threshold=0.2, blurr_threshold=30, greyscale_threshold=0.0):
                        reason = contains_tissue(patch, otsu_threshold, debug=True)
                        path = os.path.join(image_path,'deleted_patches')
                        os.makedirs(path, exist_ok=True)
                        patch.save(os.path.join(path, name + reason+ '.jpg'))
    patch_df['image_name'] = np.array(names)
    if negative_slide:
        patch_df['N'] = 1
        patch_df['P'] = 0
        patch_df['unlabeled'] = 0
    elif positive_slide:
        assert len(classes) == len(names)
        classes = np.array(classes)
        patch_df['N'] = 1 - classes
        patch_df['P'] = classes
        patch_df['unlabeled'] = 0
    elif test_slide:
        patch_df['N'] = 0
        patch_df['P'] = 0
        patch_df['unlabeled'] = 1
    return_dict[index] = patch_df

def read_tiff(path):
    image_slide = sld.open_slide(path)
    return image_slide

def run_with_multiprocessing(function, args, wsi_list):
    number_of_processes = args.number_of_processes
    n_wsis = len(wsi_list)
    df = pd.DataFrame(columns=['image_name', 'N', 'P', 'unlabeled'])
    filtered_wsi = []

    if number_of_processes == 1:
        for i in range(0, n_wsis):
            wsi_path = wsi_list[i]
            print('Working on wsi ' + wsi_path + ' ' + str(i) + ' of ' + str(n_wsis))
            fn = function
            index = 0
            return_dict = {}
            fn(wsi_path, args, index, return_dict)
            patch_df = return_dict[0]
            if len(patch_df) == 0:
                print('All patches of the WSI have been filtered out. WSI:' + str(wsi_path))
                filtered_wsi.append(wsi_path)
            else:
                df = pd.concat([df, patch_df])
    else:
        for i in range(0, n_wsis, number_of_processes):
            print(' Spawn new processes')
            print('Working on index ' + wsi_list[i] + ' ' + str(i) + ' of ' + str(n_wsis))
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            processes = []
            for j in range(number_of_processes):
                index = i + j
                if index == len(wsi_list):
                    break
                else:
                    wsi_path = wsi_list[index]
                    p = multiprocessing.Process(target=function, args=(wsi_path, args, index, return_dict))
                    processes.append(p)
                    p.start()
            for p in processes:
                p.join()
            for index in return_dict:
                patch_df = return_dict[index]
                if len(patch_df) == 0:
                    print('All patches of the WSI have been filtered out. WSI:' + str(wsi_list[index]))
                    filtered_wsi.append(wsi_list[index])
                else:
                    df = pd.concat([df, patch_df])

    return df, filtered_wsi

def main(args):
    Image.MAX_IMAGE_PIXELS = None
    wsi_data_split_lists, test_wsi_df = get_wsi_data_splits(args.image_dir, args.val_split)

    os.makedirs(args.output_dir, exist_ok=True)
    patch_path = os.path.join(args.output_dir, 'patches')
    os.makedirs(patch_path, exist_ok=True)

    wsi_df = create_wsi_df(wsi_data_split_lists, test_wsi_df)
    wsi_df.to_csv(os.path.join(args.output_dir, 'wsi_labels.csv'), index=False)
    for mode in wsi_data_split_lists.keys():
        print('Process wsis for split ' + mode)
        wsi_list = wsi_data_split_lists[mode]
        df, filtered_wsi = run_with_multiprocessing(slice_image, args, wsi_list)
        df.to_csv(os.path.join(args.output_dir,mode+ '.csv'), index=False)


    if len(filtered_wsi) > 0:
        print('The following WSI have been filtered out completely because of whiteness or blur:')
        print(filtered_wsi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", "-i", type=str, default="/data/BasesDeDatos/CAMELYON16/")
    # parser.add_argument("--image_dir", "-i", type=str, default="/home/arne/datasets/Camelyon16_dummy/")
    parser.add_argument("--mask_dir", "-m", type=str, default="/data/BasesDeDatos/Camelyon/Camelyon16/training/annotation_masks/")
    # parser.add_argument("--mask_dir", "-m", type=str, default="/home/arne/datasets/Camelyon16_dummy/training/annotation_masks/")
    parser.add_argument("--val_split", "-vs", type=float, default=0.5)

    parser.add_argument("--output_dir", "-o", type=str, default="/work/Camelyon_MIL/")
    # parser.add_argument("--output_dir", "-o", type=str, default="/home/arne/datasets/Camelyon16_dummy/patches/")
    parser.add_argument("--number_wsi", "-n", type=str, default="all")
    parser.add_argument("--dataframes_only", "-do", action='store_true')

    parser.add_argument("--patch_overlap", "-po", action='store_true')
    parser.add_argument("--patch_resolution", "-pr", type=int, default=512)
    parser.add_argument("--number_of_processes", "-np", type=int, default=1)
    parser.add_argument("--debug", "-d", action='store_true')
    args = parser.parse_args()
    print('Arguments:')
    print(args)
    main(args)