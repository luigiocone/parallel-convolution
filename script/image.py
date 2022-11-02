import numpy as np
import imageio.v2 as imageio
import sys
from skimage import data, color
from matplotlib import pyplot as plt

GRID_FILE_NAME = "../io-files/grid.bin";
RESULT_FILE_NAME = "../io-files/result.bin";

# Write a ndarray to a file in binary or text mode. The first element written is the matrix dimension
def ndarray_to_file(arr, dest_path, binary):
    mode = "wb" if binary else "w"  
    with open(dest_path, mode) as f:
        f.write(arr.shape[0].to_bytes(length=4, byteorder=sys.byteorder, signed=False))
        if binary:   
            arr.astype(np.float32).tofile(f, sep='')
        else:
            np.savetxt(f, arr, delimiter=' ', fmt="%+e")

# Read a ndarray from a binary or text file. If skip_header == 1, then the first element should be the matrix dimension
def file_to_ndarray(src_path, binary, skip_header):
    src_img, shape = 0, 0
    mode = "rb" if binary else "r"  
    with open(src_path, mode) as f:
        if binary:
            if skip_header != 0:
                shape = int.from_bytes(f.read(4), sys.byteorder, signed=False)
            src_img = np.fromfile(f, dtype=np.float32, count=-1)
        else:
            src_img = np.genfromtxt(src_path, delimiter=' ', dtype=np.float32, skip_header=skip_header)

        if skip_header == 0:
            shape = round(np.sqrt(src_img.shape[0]))
        
        src_img = np.reshape(src_img, newshape=(shape, shape), order='C')
    return src_img;

# Get an image from filesystem and convert it to greyscale
def image_to_gray_ndarray(src_path):
    img = imageio.imread(src_path)     # Get image from file system as ndarray
    img = color.rgb2gray(img)          # Convert to grayscale
    return img;

# Plot two images
def plot_images(left_img, right_img):
    print(f'src_img type:  {type(left_img)} | res_img type: {type(right_img)}')
    print(f'src_img dtype: {left_img.dtype} | res_img dtype {right_img.dtype}')
    print(f'src_img shape: {left_img.shape} | res img shape {right_img.shape}')

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1, 2) 
    axarr[0].imshow(left_img, cmap='gray')
    axarr[1].imshow(right_img, cmap='gray')
    plt.show();

def main():
    # Get an image from scikit-image dataset or from filesystem (as actual image or something else)
    '''src_img = data.camera()'''
    '''src_img = image_to_gray_ndarray("/home/luigi/Desktop/parallel-convolution/io-files/other/haring.jpg")'''
    src_img = file_to_ndarray("../io-files/grid.bin", binary=True, skip_header=1)

    # conv.c works only with square matrices
    if (src_img.shape[0] != src_img.shape[1]):
        print("Not a square image")
        return;

    # Store the image in a file (choosing between text or binary mode)
    # ndarray_to_file(src_img, GRID_FILE_NAME, binary=True)

    # Load convolution result and plot it next to source matrix
    res_img = file_to_ndarray(RESULT_FILE_NAME, binary=True, skip_header=0)
    plot_images(src_img, res_img)


if __name__ == "__main__":
    main()
