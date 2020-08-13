import random, os, shutil
from pathlib import Path

num_images_train = 10   # in thousands  (basically how many images per directo)
num_images_test = 2     # in thousands

def create_clean_data():
    folder = "RawData/original_sequences/"
    outTrain = "data/train/original/"
    outTest = "data/test/original/"
    Path(outTrain).mkdir(parents=True, exist_ok=True)
    Path(outTest).mkdir(parents=True, exist_ok=True)

    i_test = 0
    i_train = 0

    # Does original data
    for i in range(1000): # 0 to 999
        subfolder = folder + str(i).zfill(3) + "/" # all folder names are zero padded.
        print("Subfolder:", subfolder)
        upp = len(os.listdir(subfolder)) -1
        
        for x in range(num_images_test):
            img_i = random.randint(0, upp)
            img_in = subfolder + str(img_i).zfill(4) + ".png"
            img_out = outTest + str(i_test).zfill(5) + ".png"
            shutil.copy(img_in, img_out)
            i_test+=1
        for x in range(num_images_train):
            img_i = random.randint(0, upp)
            img_in = subfolder + str(img_i).zfill(4) + ".png"
            img_out = outTrain + str(i_train).zfill(5) + ".png"
            shutil.copy(img_in, img_out)
            i_train+=1


def create_deepfake_data():
    in_folder = "RawData/manipulated_sequences/Deepfakes/"
    outTrain = "data/train/deepfake/"
    outTest = "data/test/deepfake/"
    Path(outTrain).mkdir(parents=True, exist_ok=True)
    Path(outTest).mkdir(parents=True, exist_ok=True)
    i_test = 0
    i_train = 0
    for subdirs, dirs, files in os.walk(in_folder):
        if(len(subdirs) <= len(in_folder)):
            continue
        subdirs += "/"
        print("Subfolder:", subdirs)
        upp = len(os.listdir(subdirs)) -1
        for x in range(num_images_test):
            img_i = random.randint(0, upp)
            img_in = subdirs + str(img_i).zfill(4) + ".png"
            img_out = outTest + str(i_test).zfill(3) + ".png"
            shutil.copy(img_in, img_out)
            i_test+=1
        for x in range(num_images_train):
            img_i = random.randint(0, upp)
            img_in = subdirs + str(img_i).zfill(4) + ".png"
            img_out = outTrain + str(i_train).zfill(3) + ".png"
            shutil.copy(img_in, img_out)
            i_train+=1

def create_face2face_data():
    in_folder = "RawData/manipulated_sequences/Face2Face/"
    outTrain = "data/train/face2face/"
    outTest = "data/test/face2face/"
    Path(outTrain).mkdir(parents=True, exist_ok=True)
    Path(outTest).mkdir(parents=True, exist_ok=True)
    i_test = 0
    i_train = 0
    for subdirs, dirs, files in os.walk(in_folder):
        if(len(subdirs) <= len(in_folder)):
            continue
        subdirs += "/"
        print("Subfolder:", subdirs)
        upp = len(os.listdir(subdirs)) -1
        for x in range(num_images_test):
            img_i = random.randint(0, upp)
            img_in = subdirs + str(img_i).zfill(4) + ".png"
            img_out = outTest + str(i_test).zfill(3) + ".png"
            shutil.copy(img_in, img_out)
            i_test+=1
        for x in range(num_images_train):
            img_i = random.randint(0, upp)
            img_in = subdirs + str(img_i).zfill(4) + ".png"
            img_out = outTrain + str(i_train).zfill(3) + ".png"
            shutil.copy(img_in, img_out)
            i_train+=1

def create_faceswap_data():
    in_folder = "RawData/manipulated_sequences/FaceSwap/"
    outTrain = "data/train/faceswap/"
    outTest = "data/test/faceswap/"
    Path(outTrain).mkdir(parents=True, exist_ok=True)
    Path(outTest).mkdir(parents=True, exist_ok=True)
    i_test = 0
    i_train = 0
    for subdirs, dirs, files in os.walk(in_folder):
        if(len(subdirs) <= len(in_folder)):
            continue
        subdirs += "/"
        print("Subfolder:", subdirs)
        upp = len(os.listdir(subdirs)) -1
        for x in range(num_images_test):
            img_i = random.randint(0, upp)
            img_in = subdirs + str(img_i).zfill(4) + ".png"
            img_out = outTest + str(i_test).zfill(3) + ".png"
            shutil.copy(img_in, img_out)
            i_test+=1
        for x in range(num_images_train):
            img_i = random.randint(0, upp)
            img_in = subdirs + str(img_i).zfill(4) + ".png"
            img_out = outTrain + str(i_train).zfill(3) + ".png"
            shutil.copy(img_in, img_out)
            i_train+=1


"""
    Uncomment and do each one, as needed.
"""
# create_clean_data()
# create_deepfake_data()
# create_face2face_data()
# create_faceswap_data()
