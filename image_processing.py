import numpy as np
import io
from imageio import imread, imwrite


# Define several image-processing functions.
#
# A color image is a 3-dimensional NumPy array of shape (height, width, 3). 
# The elements of the array are 8-bit unsigned integers. The 3-element array
# referenced by `image[i, j]` contains the RGB or HSV values of a pixel.


def write_png(image, title, formatter):
    """
    Stores `image` in PNG format, returns the file size in bits.
    
    The `title` is extended with '.png' to generate the name of the file
    in which `image` is stored. The `title` is printed along with the
    file size by applying the `format` method to the string `formatter`.
    """
    filename = title + '.png'
    with io.FileIO(filename, 'wb') as compressed:
        imwrite(compressed, image, format='png')
        png_bits = 8 * compressed.tell()
    print(formatter.format(title, png_bits))
    return png_bits

    
def overlap(a, b):
    """
    Returns overlap of images `a` and `b` aligned at upper-left corner.
    
    That is, two images are returned. The height (width) of the returned
    images is the mimimum of the heights (widths) of the given images.
    Any cropping that occurs is at the bottom and/or right of the images.
    """
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    return a[:height, :width], b[:height, :width]


def cumulative_sum(image):
    """
    Returns the cumulative 8-bit sum of RGB values in `image`.
    
    The summands are in row-major order. Overflow bits are discarded. The
    result is an image that has the same shape as the given `image`.
    """
    # First flatten a copy of the given 3-d image into a 1-d array, and
    # replace the RGB values in the 1-d array with their cumulative sum.
    # Then reshape the 1-d array to match the given image in shape.
    csum = image.flatten()
    np.cumsum(csum, out=csum)
    return csum.reshape(image.shape)