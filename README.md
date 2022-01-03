# Domain-Translated 3D Object Pose Estimation

This is the official implementation of the paper [Domain-Translated 3D Object Pose Estimation](https://ieeexplore.ieee.org/abstract/document/9206072).

# Installation
- Tested both with Python 2.7 and Python 3.6
- Keras 2.3
- OpenCV
- scipy 1.1

# Preparation
Download pre-trained models and pre-calculated database encodings from [here](https://drive.google.com/file/d/1CT1FiiXMNm0l7n7PRn2tgU5VTqQJ2Gwg/view?usp=sharing). Extract the .zip file and place the directory 'pose_models' in the project ROOT directory.

# Demo
Test on a single image from the Cyclists dataset:

`python demo.py --test_obj_img test_cyclist.jpg`

Test on a single image from the LineMod dataset:

`python demo.py --dataset linemod --test_obj_img test_ape.png`

You can also run `python demo.py --help` to see more usage options (visualization, etc.).

# Citation
If you use this code please cite our work:

```
@article{papaioannidis2020domain,
  title={Domain-translated 3D object pose estimation},
  author={Papaioannidis, Christos and Mygdalis, Vasileios and Pitas, Ioannis},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={9279--9291},
  year={2020},
  publisher={IEEE}
}

@article{papaioannidis20193d,
  title={3D object pose estimation using multi-objective quaternion learning},
  author={Papaioannidis, Christos and Pitas, Ioannis},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={30},
  number={8},
  pages={2683--2693},
  year={2019},
  publisher={IEEE}
}
```
