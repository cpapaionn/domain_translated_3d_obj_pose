from __future__ import print_function

import os
import numpy as np
import cv2
from scipy import spatial
import argparse

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Conv2DTranspose, Reshape
from keras import regularizers


class Pose3D:
    def __init__(self, dataset, mode, input_type):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.pose_nn_weights_path = os.path.join(self.dir_path, 'pose_models/with_i2i', dataset, input_type, mode, 'pose_nn.hdf5')
        self.i2i_nn_weights_path = os.path.join(self.dir_path, 'pose_models/with_i2i', dataset, input_type, mode, 'i2i.hdf5')

        tmpl_imgs_fname = 'tmpl_imgs.npz'
        tmpl_encs_fname = 'tmpl_encs.npz'
        tmpl_poses_fname = 'tmpl_poses_q.npz'

        self.tmpl_imgs_enc = np.load(os.path.join('pose_models/with_i2i', dataset, input_type, mode, tmpl_encs_fname))['tmpl_encs']
        self.tmpl_poses_q = np.load(os.path.join('pose_models/with_i2i', dataset, input_type, mode, tmpl_poses_fname))['tmpl_poses_q']
        self.tmpl_imgs = np.load(os.path.join('pose_models/with_i2i', dataset, input_type, mode, tmpl_imgs_fname))['tmpl_imgs']

        if input_type == 'RGB':
            self.input_dim = (64, 64, 3)
       
        self.i2i_model = self.__create_i2i_network(input_type)
        self.pose_model = self.__create_base_network()


    def pose_inference(self, test_obj_img):
        test_obj_img = np.expand_dims(test_obj_img, axis=0)
        pred_pose_q, pred_tmpl_img = self.__predict_3d_pose(test_obj_img)

        return pred_pose_q, pred_tmpl_img


    def __predict_3d_pose(self, test_img):
        test_img_i2i_inp = test_img / 255.
        test_img_transl = self.i2i_model.predict(test_img_i2i_inp, verbose=0)
        test_img_transl *= 255.
        test_img_transl = np.rint(test_img_transl)
        test_img_transl = test_img_transl.astype(np.uint8)

        self.test_img_enc = self.pose_model.predict(test_img_transl, verbose=0)
        nn = self.__find_knn(k=2)
        t_idx = nn[0, 0]
        pred_3d_pose = self.tmpl_poses_q[t_idx]
        pred_tmpl = self.tmpl_imgs[t_idx]

        return pred_3d_pose, pred_tmpl


    def __find_knn(self, k=1):
        """ Find k nearest neighbors of test images encodings """
        tree = spatial.cKDTree(self.tmpl_imgs_enc)
        knn = tree.query(self.test_img_enc, k=k)[1]

        return knn


    def __create_i2i_network(self, input_type):
        inp = Input(shape=self.input_dim)
        ### Encoder ###

        # 1st block
        x = Conv2D(128, (3, 3), padding='same', name="conv2d_b1")(inp)
        x = BatchNormalization(name="bn_b1")(x)
        x = Activation('relu', name="relu_b1")(x)

        # 2nd block
        x = Conv2D(256, (3, 3), strides=(2, 2), padding='same', name="conv2d_b2")(x)
        x = BatchNormalization(name="bn_b2")(x)
        x = Activation('relu', name="relu_b2")(x)

        # 3rd block
        x = Conv2D(256, (3, 3), strides=(2, 2), padding='same', name="conv2d_b3")(x)
        x = BatchNormalization(name="bn_b3")(x)
        x = Activation('relu', name="relu_b3")(x)

        # 4th block
        x = Conv2D(512, (3, 3), strides=(2, 2), padding='same', name="conv2d_b4")(x)
        x = BatchNormalization(name="bn_b4")(x)
        x = Activation('relu', name="relu_b4")(x)

        # 5th block
        x = Conv2D(256, (8, 8), name="conv2d_b5")(x)
        x = BatchNormalization(name="bn_b5")(x)
        x = Activation('relu', name="relu_b5")(x)

        ### Decoder ###

        # 1st block
        y = Reshape((8, 8, 4))(x)
        y = Conv2D(512, (3, 3), padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        # 2nd block
        y = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        # 3rd block
        y = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        # 4th block
        y = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        # Refine block
        if input_type == 'RGB':
            y = Conv2D(3, (1, 1), padding='same')(y)
        y = Activation('sigmoid')(y)

        generator = Model(inputs=[inp], outputs=[y])
        generator.load_weights(self.i2i_nn_weights_path, by_name=True)

        return generator


    def __create_base_network(self):
        """ Base network to be shared (eq. to feature extraction). """
        seq = Sequential(name="base_network")
    
        # 1st block
        seq.add(Conv2D(16, (8, 8), input_shape=self.input_dim, kernel_regularizer=regularizers.l2(0.05), padding='same', name="conv2d_b1"))
        seq.add(BatchNormalization(name="bn_b1"))
        seq.add(Activation('relu', name="relu_b1"))
        seq.add(MaxPooling2D(pool_size=(2, 2), name="maxpool_b1"))

        # 2nd block
        seq.add(Conv2D(7, (5, 5), kernel_regularizer=regularizers.l2(0.05), padding='same', name="conv2d_b2"))
        seq.add(BatchNormalization(name="bn_b2"))
        seq.add(Activation('relu', name="relu_b2"))
        seq.add(MaxPooling2D(pool_size=(2, 2), name="maxpool_b2"))

        # Dense blocks
        seq.add(Flatten(name="flat_b1"))
        seq.add(Dense(256, kernel_regularizer=regularizers.l2(0.05), name="dense_b1"))
        seq.add(BatchNormalization(name="bn_b3"))
        seq.add(Activation('relu', name="relu_b3"))
        seq.add(Dense(32, kernel_regularizer=regularizers.l2(0.05), name="dense_b2"))

        test_img = Input(shape=self.input_dim)
        test_img_enc = seq(test_img)
        pose_model = Model(inputs=[test_img], outputs=[test_img_enc])
        pose_model.load_weights(self.pose_nn_weights_path, by_name=True)

        return pose_model


    def cv_show_image(self, image):
        """ Print image with opencv """
        cv2.imshow('Left: Test image, Right: Closest retrieved database image', image / 255.)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return 0


def parse_args():
    parser = argparse.ArgumentParser(description='3D object pose estimation demo')

    parser.add_argument('--dataset',
                        help='Select pre-trained model to load. Available: "linemod" or "cyclists". Default: "cyclists".',
                        default='cyclists',
                        required=False,
                        type=str)
    parser.add_argument('--mode',
                        help='Select pre-trained model type. Available: "synth" or "real_and_synth". For "cyclists" dataset only "real_and_synth" is available. Default: "real_and_synth".',
                        default='real_and_synth',
                        required=False,
                        type=str)
    parser.add_argument('--input_type',
                        help='Select input type. Available: "RGB". Default: "RGB".',
                        default='RGB',
                        required=False,
                        type=str)
    parser.add_argument('--test_obj_img',
                        help='Enter path of a centered and cropped test object image.',
                        default='',
                        required=True,
                        type=str)
    parser.add_argument('--out_dir',
                        help='Enter path of a directory to save the outputs. Default: "./".',
                        default='./',
                        required=False,
                        type=str)
    parser.add_argument('--vis',
                        help='Select visualization mode. Available: "show", "save" or "no_vis". Default: "no_vis".',
                        default='no_vis',
                        required=False,
                        type=str)

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    dataset = args.dataset
    mode = args.mode
    input_type = args.input_type
    test_obj_img = cv2.imread(args.test_obj_img)
    out_dir = args.out_dir
    vis = args.vis

    if dataset not in ['linemod', 'cyclists']:
        print("Invalid 'dataset' type: {}.".format(dataset))
        return 0
    if mode not in ['synth', 'real_and_synth']:
        print("Invalid 'mode' type: {}.".format(mode))
        return 0
    if input_type not in ['RGB']:
        print("Invalid 'input_type' type: {}.".format(input_type))
        return 0
    if test_obj_img is None:
        print("File '{}' does not exist.".format(args.test_obj_img))
        return 0
    if mode == 'synth' and dataset == 'cyclists':
        print("'mode' type: {} is not available for 'dataset' type: {}.".format(mode, dataset))
        return 0

    print("Selected settings: 'dataset': {}, 'mode': {}, 'input_type': {}, 'test_obj_img': {}, 'out_dir': {}, 'vis': {}".format(dataset, mode, input_type, args.test_obj_img, out_dir, vis))

    cip = Pose3D(dataset, mode, input_type)
    pred_pose_q, pred_tmpl_img = cip.pose_inference(test_obj_img)

    if vis == 'show' or vis == 'save':
        show_img = np.concatenate([test_obj_img, pred_tmpl_img], axis=1)
        if vis == 'show':
            cip.cv_show_image(show_img)
        else:
            cv2.imwrite(os.path.join(out_dir, 'test_and_closest_retrieved_img.png'), show_img)
    print('Predicted 3D pose in unit quaternion: [q0, q1, q2, q3] = {}'.format(pred_pose_q))


if __name__ == '__main__':
    main()



