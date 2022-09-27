import numpy as np
import imageio.v2 as imageio
from skimage import data, color
from matplotlib import pyplot as plt

GRID_FILE_NAME = "../io-files/grid.txt";
RESULT_FILE_NAME = "../io-files/result.txt";

def img_to_matrix(img):
    print('Type:', type(img))
    print('dtype:', img.dtype)
    print('shape:', img.shape)

    # conv.c works only with square matrix 
    if (img.shape[0] != img.shape[1]):
        print("Not a square image")
        return;
    
    with open(GRID_FILE_NAME, "w+") as f:
        f.write("{0}\n".format(img.shape[0]))
        np.savetxt(f, img, delimiter=' ', fmt="%+e")   # Format "%+e" has 13 chars

def main():
    # Get an image from scikit-image dataset
    '''src_img = data.camera()'''  # 512x512
    # Or else, get image from file
    '''src_img = imageio.imread("/home/luigi/Desktop/input.jpg")
    src_img = color.rgb2gray(src_img)'''
       
    # Finally, store the image in a txt file (as a float matrix)
    '''img_to_matrix(src_img)'''

    # From matrix file to img
    src_img = np.genfromtxt(GRID_FILE_NAME,   delimiter=' ', dtype=float, skip_header=1)
    res_img = np.genfromtxt(RESULT_FILE_NAME, delimiter=' ', dtype=float)

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,2) 
    axarr[0].imshow(src_img, cmap='gray')
    axarr[1].imshow(res_img, cmap='gray')
    plt.show();
    

if __name__ == "__main__":
    main()
