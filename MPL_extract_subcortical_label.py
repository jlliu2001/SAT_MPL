import pandas as pd
import json
from tqdm import tqdm
import copy
from pathlib import Path
import os
import random
import numpy as np
import argparse
import shutil
import nibabel as nib
import SimpleITK as sitk
import glob

def nii_zip(brain_path,output_tof_file):
    cube_image = sitk.ReadImage(brain_path)
    cube = sitk.GetArrayFromImage(cube_image)
    original_spacing = cube_image.GetSpacing()
    original_origin = cube_image.GetOrigin()
    original_direction = cube_image.GetDirection()
    
    cube_new_image = sitk.GetImageFromArray(cube)
    cube_new_image.SetSpacing(original_spacing)
    # cube_new_image.SetSize(out_size)
    cube_new_image.SetDirection(original_direction)
    cube_new_image.SetOrigin(original_origin)

    sitk.WriteImage(cube_new_image, output_tof_file)

def ReOrientation_to_RAS(input_path, output_path):
    print('input path:',input_path)
    tof_raw: nib.Nifti1Image = nib.load(input_path)
    print(f"Original orientation: {nib.aff2axcodes(tof_raw.affine)}")

    tof = nib.as_closest_canonical(tof_raw)
    print(f"Transformed orientation: {nib.aff2axcodes(tof.affine)}")
    tof.to_filename(output_path)
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--text_path')
    # parser.add_argument('--test_jsonl')
    # parser.add_argument('--split_json')
    config = parser.parse_args()

    data_dir=config.data_dir
    text_path=config.text_path
    for sub in os.listdir(data_dir):
        sub_img_path=os.path.join(data_dir,sub,sub,sub,sub+'_1.nii')
        sub_label_path=os.path.join(data_dir,sub,sub,sub,sub+'_1_seg.nii')
        nii_zip(sub_img_path,os.path.join(data_dir,sub,sub,sub,sub+'_1.nii.gz'))
        nii_zip(sub_img_path,os.path.join(data_dir,sub,sub,sub,sub+'_1_seg.nii.gz'))
        sub_path=os.path.join(data_dir,sub,sub,sub,sub+'_1_seg_RAS.nii.gz')
        ReOrientation_to_RAS(os.path.join(data_dir,sub,sub,sub,sub+'_1_seg.nii.gz'), sub_path)
        ReOrientation_to_RAS(os.path.join(data_dir,sub,sub,sub,sub+'_1.nii.gz'), os.path.join(data_dir,sub,sub,sub,sub+'_1_RAS.nii.gz'))


        
        # shutil.copy(sub_path,os.path.join(data_dir,sub,sub,sub,sub+'_1_seg_subcortical_LR_RAS.nii.gz'))
        shutil.copy(text_path,os.path.join(data_dir,sub,sub,sub,sub+'_1_seg_Label.txt'))

        cube_image = sitk.ReadImage(sub_path)
        cube = sitk.GetArrayFromImage(cube_image)
        original_spacing = cube_image.GetSpacing()
        original_origin = cube_image.GetOrigin()
        original_direction = cube_image.GetDirection()

        new_cube=np.zeros_like(cube)
        source_list=[48,47,32,31,37,36,58,57,56,55,60,59,30,23]
        tgr_list=[1,1,2,2,3,3,4,4,5,5,6,6,7,7]
        for i,j in zip(source_list,tgr_list):
            new_cube[cube==i]=j

        
        
        cube_new_image = sitk.GetImageFromArray(new_cube)
        cube_new_image.SetSpacing(original_spacing)
        # cube_new_image.SetSize(out_size)
        cube_new_image.SetDirection(original_direction)
        cube_new_image.SetOrigin(original_origin)
        output_path=os.path.join(data_dir,sub,sub,sub,sub+'_1_seg_subcortical_RAS.nii.gz')

        sitk.WriteImage(cube_new_image, output_path)
    


    # predefined_train_test_split(config.jsonl2split, config.train_jsonl, config.test_jsonl, config.split_json)