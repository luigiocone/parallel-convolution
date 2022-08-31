import numpy as np
from skimage import data
from matplotlib import pyplot as plt

GRID_FILE_NAME = "grid.txt";
RESULT_FILE_NAME = "result.txt";

def img_to_matrix(img):
    print('Type:', type(img))
    print('dtype:', img.dtype)
    print('shape:', img.shape)

    # conv.c works only with square matrix 
    if (img.shape[0] != img.shape[1]):
        return;
    
    f = open(GRID_FILE_NAME, "w")
    f.write("{0}\n".format(img.shape[0]))
    # Format "%e" has six digits after point as standard 
    np.savetxt(f, img, delimiter=' ', fmt="%e")
    f.close()

def main():
    # To store a custom image from scikit-image dataset
    # src_img = data.camera()  # This should be 512x512
    # img_to_matrix(src_img)   # From img to matrix file
    #return;
    
    # From matrix file to img
    src_img = np.genfromtxt(GRID_FILE_NAME,   delimiter=' ', dtype=float, skip_header=1)
    res_img = np.genfromtxt(RESULT_FILE_NAME, delimiter=' ', dtype=float, skip_header=2, skip_footer=2, usecols=range(2,510))

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,2) 
    axarr[0].imshow(src_img, cmap='gray')
    axarr[1].imshow(res_img, cmap='gray')
    plt.show();
    

if __name__ == "__main__":
    main()
