import os
import sys
import argparse
import numpy as np
from os import listdir
from scipy.misc import imread, imresize
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Settings:
img_size =None
dataset_dir=None
grayscale_images = None
num_class=None
test_size = 0.2
#args =sys.argv #first argument is dataset path to images data

def main():
    """Get the dataset by args pare."""
    args = parse()
    global dataset_dir
    global img_size
    global num_class
    global grayscale_images
    dataset_dir =args.dir
    img_size = args.size
    num_class = args.classes
    grayscale_images = args.grayscale

    get_dataset()

def parse():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', '-d',
                        help="Directory of the dataset/images")
    parser.add_argument('--size', '-s',
                        help="Size of image you want to convert to.",type=int)
    parser.add_argument('--grayscale','-g',
                        help="want to convert grayscale or not",default=False,type=bool)
    parser.add_argument('--classes', '-c',
                        help="Number of classes.",type=int)
    args = parser.parse_args()
    return args

def get_img(data_path):
    # Getting image array from path:
    img = imread(data_path, flatten=grayscale_images)
    img = imresize(img, (img_size, img_size, 1 if grayscale_images else 3))
    return img

def get_dataset():
    dataset_path=dataset_dir
    # Getting all data from data path:
    new_dataset_dir = "npy_dataset/"
    try:
        X = np.load('npy_dataset/X.npy')
        Y = np.load('npy_dataset/Y.npy')
    except:
        print("Couldn't find dataset in folder {0} so creating one.".format(new_dataset_dir))

        labels = sorted(listdir(dataset_path)) # Geting labels
        print labels
        X = []
        Y = []
        for i, label in enumerate(labels):
            datas_path = dataset_path+'/'+label
            for data in listdir(datas_path):
                img = get_img(datas_path+'/'+str(data))
                X.append(img)
                Y.append(i)
        # Create dateset: (normalizing X)
        #X = np.array(X).astype('float32')/255.
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, num_class)
        if not os.path.exists(new_dataset_dir):
            os.makedirs(new_dataset_dir)
        np.save('npy_dataset/X.npy', X)
        np.save('npy_dataset/Y.npy', Y)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    print('finished creating dataset in path ',new_dataset_dir)
    return X, X_test, Y, Y_test



if __name__ == '__main__':
    main()
