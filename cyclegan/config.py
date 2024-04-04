import torch
import albumentations as A
import os
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 2
NUM_EPOCHS = 100
LOAD_MODEL = True
SAVE_MODEL = False
train = False

transforms = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

def load_model(type_name):
    path = os.getcwd()
    path += "/cyclegan/models"
    if type_name == "head" :
        CHECKPOINT_GEN_H = path + "/all/genh.pth.tar"
        CHECKPOINT_GEN_Z = path + "/all/genz.pth.tar"
        CHECKPOINT_CRITIC_H = path + "/all/critich.pth.tar"
        CHECKPOINT_CRITIC_Z = path + "/all/criticz.pth.tar"
    elif type_name == "left_eye" :
        CHECKPOINT_GEN_H = path + "/left_eye/genh.pth.tar"
        CHECKPOINT_GEN_Z = path + "/left_eye/genz.pth.tar"
        CHECKPOINT_CRITIC_H = path + "/left_eye/critich.pth.tar"
        CHECKPOINT_CRITIC_Z = path + "/left_eye/criticz.pth.tar"
    elif type_name == "right_eye" :
        CHECKPOINT_GEN_H = path + "/right_eye/genh.pth.tar"
        CHECKPOINT_GEN_Z = path + "/right_eye/genz.pth.tar"
        CHECKPOINT_CRITIC_H = path + "/right_eye/critich.pth.tar"
        CHECKPOINT_CRITIC_Z = path + "/right_eye/criticz.pth.tar"
    elif type_name == "mouse" :
        CHECKPOINT_GEN_H = path + "/mouse/genh.pth.tar"
        CHECKPOINT_GEN_Z = path + "/mouse/genz.pth.tar"
        CHECKPOINT_CRITIC_H = path + "/mouse/critich.pth.tar"
        CHECKPOINT_CRITIC_Z = path + "/mouse/criticz.pth.tar"
    else:
        print("no this type_name")

    return (CHECKPOINT_GEN_H,CHECKPOINT_GEN_Z,CHECKPOINT_CRITIC_H,CHECKPOINT_CRITIC_Z)

def get_dir(type_name):
    path = os.getcwd()
    path +="/cyclegan/image/temp"
    path2 = path
    if type_name == "head" :
        path += "/cut_face"
        path2 += "/cut_sketch"
    elif type_name == "left_eye" :
        path += "/left_eye"
        path2 += "/sketch_left_eye"
    elif type_name == "right_eye" :
        path += "/right_eye"
        path2 += "/sketch_right_eye"
    elif type_name == "mouse" :
        path += "/mouse"
        path2 += "/sketch_mouse"
    else:
        print("no this type_name")
    return (path,path2)
