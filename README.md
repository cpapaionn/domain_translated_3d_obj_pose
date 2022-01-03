# Domain-Translated 3D Object Pose Estimation

This is the official implementation of the paper [Domain-Translated 3D Object Pose Estimation](https://ieeexplore.ieee.org/abstract/document/9206072).

# Preparation
Download pre-trained models and pre-calculated database encodings from [here](https://drive.google.com/file/d/1CT1FiiXMNm0l7n7PRn2tgU5VTqQJ2Gwg/view?usp=sharing). Extract the .zip file and place the directory 'pose_models' in the ROOT directory.

# Demo
Test on a single image from the Cyclists dataset:
'python demo.py --test_obj_img test_cyclist.jpg'
