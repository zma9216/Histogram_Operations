from skimage import io, img_as_ubyte
import skimage.exposure as exp
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(__file__))


def read_image(filename, gray):
    """

    Function for reading images using scikit-image
    Reads image as grayscale and converts to
    unsigned byte format if bool is True for gray
    @param filename: String
    @param gray: Boolean
    @return: np.ndarray
    """
    if gray:
        img = io.imread(filename, as_gray=True)
        gray_img = img_as_ubyte(img)
        return gray_img
    else:
        img = io.imread(filename)
        return img


def hist_calc(image):
    """

    Computes histogram for a grayscale image
    Based on algorithm from lecture
    @param image: np.ndarray
    @return: np.ndarray
    """
    row, col = image.shape
    my_hist = np.zeros(256)
    for i in range(row):
        for j in range(col):
            my_hist[image[i, j]] += 1
    return my_hist


def calc_normhist(image):
    """

    Computes normalized histogram for a grayscale image
    Based on algorithm from lecture
    @param image: np.ndarray
    @return: np.ndarray
    """
    img_hist = exp.histogram(image, nbins=256, source_range="dtype")
    norm_hist = img_hist[0] / image.size
    return norm_hist


def rgb_hist_matching(src, ref):
    """

    Function for histogram matching for RGB images
    Based on implementation from Part 4, but solution likely incorrect
    Since I could not figure out how to do the mapping for each channel
    @param src: String
    @param ref: String
    @return:
    """
    src_jpg = src
    ref_jpg = ref
    rgb_src = read_image(src_jpg, gray=False)
    rgb_ref = read_image(ref_jpg, gray=False)

    day_hist = exp.histogram(rgb_src.flatten(), nbins=256, source_range="dtype")
    cumd_day_rgb = np.cumsum(day_hist[0])
    nc_day_rgb = cumd_day_rgb / rgb_src.size

    night_hist = exp.histogram(rgb_ref.flatten(), nbins=256, source_range="dtype")
    cumd_night_rgb = np.cumsum(night_hist[0])
    nc_night_rgb = cumd_night_rgb / rgb_ref.size

    ref_pixels = np.arange(nc_night_rgb.size)
    mapped = np.interp(nc_day_rgb, nc_night_rgb, ref_pixels)
    matched = (np.reshape(mapped[rgb_src], rgb_src.shape)).astype(np.uint8)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))
    axes[0].imshow(rgb_src)
    axes[0].set_title("Source")

    axes[1].imshow(rgb_ref)
    axes[1].set_title("Reference")

    axes[2].imshow(matched)
    axes[2].set_title("Matched")
    plt.show()


def part1_histogram_compute():
    test_jpg = "test.jpg"
    # Read input image as grayscale
    img = read_image(test_jpg, gray=True)

    # Get height and width of image
    row, col = img.shape
    # Create ndarray of zeroes with size of 256
    my_hist = np.zeros(256)
    # Implementation of algorithm from histogram lecture
    for i in range(row):
        for j in range(col):
            my_hist[img[i, j]] += 1

    # Create 3 subplots (15 x 5 in size) in a single row
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    # Plot histogram from ndarray computed and labelled as such
    axes[0].plot(my_hist)
    axes[0].set_title("My Histogram")
    axes[0].set_xlim(0, 255)

    # Compute histogram using skimage built-in function and plot as such
    ski_hist = exp.histogram(img, nbins=256, source_range="dtype")
    axes[1].plot(ski_hist[0])
    axes[1].set_title("SKimage Histogram")
    axes[1].set_xlim(0, 255)

    # Compute histogram using numpy built-in function and plot as such
    np_hist = np.histogram(img, bins=256, range=(0, 255))
    axes[2].plot(np_hist[0])
    axes[2].set_title("NumPy Histogram")
    axes[2].set_xlim(0, 255)

    # Adjust subplot parameters and display the figures
    plt.tight_layout()
    plt.show()


def part2_histogram_equalization():
    test_jpg = "test.jpg"
    # Read input image as grayscale
    img = read_image(test_jpg, gray=True)
    # Compute histogram using user-defined function
    my_hist = hist_calc(img)
    # Obtain height and width of image
    row, col = img.shape

    # Compute cumulative sum from histogram using list comprehension
    cumul_sum = np.array([sum(my_hist[:i + 1]) for i in range(len(my_hist))])
    # Compute CDF using algorithm from histogram lecture
    cdf = np.floor(((255 / img.size) * cumul_sum) + 0.5)

    # Create ndarray of zeroes for new equalized image
    eq_img = np.zeros_like(img)
    # Based on algorithm from https://towardsdatascience.com/histogram-matching-ee3a67b4cbc1
    # To add new pixels to the image ndarray
    for x_pixel in range(row):
        for y_pixel in range(col):
            eq_img[x_pixel, y_pixel] = cdf[img[x_pixel, y_pixel]]

    # Compute histogram of new image
    eq_hist = hist_calc(eq_img)

    # Create 4 subplots (10 x 10 in size) in two rows and two columns
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    # Display original image in figure 1
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].set_xlim(0, 255)

    # Plot histogram of original image in figure 2
    axes[0, 1].plot(my_hist)
    axes[0, 1].set_title("Histogram")
    axes[0, 1].set_xlim(0, 255)

    # Display new equalized image in figure 3
    axes[1, 0].imshow(eq_img, cmap="gray")
    axes[1, 0].set_title("New Image")
    axes[1, 0].set_xlim(0, 255)

    # Plot equalized histogram in figure 4
    axes[1, 1].plot(eq_hist)
    axes[1, 1].set_title("Equalized Histogram")
    axes[1, 1].set_xlim(0, 255)

    # Display and adjust figures
    plt.tight_layout()
    plt.show()


def part3_histogram_comparing():
    day_jpg = "day.jpg"
    night_jpg = "night.jpg"
    # Read day.jpg and night.jpg as grayscale images
    day_gray = read_image(day_jpg, gray=True)
    night_gray = read_image(night_jpg, gray=True)

    # Compute normalized histograms for both images
    day_hist = calc_normhist(day_gray)
    night_hist = calc_normhist(night_gray)

    # Print the calculated Bhattacharyya Coefficient
    # Calculation derived from histogram lecture
    print("BC is:", sum(np.sqrt(day_hist * night_hist)))


def part4_histogram_matching():
    src_jpg = "day.jpg"
    ref_jpg = "night.jpg"
    # Read day.jpg and night.jpg as grayscale images
    gray_src = read_image(src_jpg, gray=True)
    gray_ref = read_image(ref_jpg, gray=True)

    # Compute histogram for source image
    day_hist = exp.histogram(gray_src, nbins=256, source_range="dtype")
    # Compute cumulative histogram from source histogram
    cumd_day_gray = np.cumsum(day_hist[0])
    # Normalize the cumulative histogram
    nc_day_gray = cumd_day_gray / gray_src.size

    # Compute histogram for source image
    night_hist = exp.histogram(gray_ref, nbins=256, source_range="dtype")
    # Compute cumulative histogram from reference histogram
    cumd_night_gray = np.cumsum(night_hist[0])
    # Normalize the cumulative histogram
    nc_night_gray = cumd_night_gray / gray_ref.size

    # Implementation derived from https://stackoverflow.com/a/51702334 - Author: Sandipan Dey
    # Create ndarray from reference image with its corresponding pixels
    ref_pixels = np.arange(nc_night_gray.size)
    # Compute interpolated values (new pixels) using CDF of input and reference image
    new_pixels = np.interp(nc_day_gray, nc_night_gray, ref_pixels)
    # Reshape image ndarray using the new pixels computed
    matched = np.reshape(new_pixels[gray_src], gray_src.shape)
    # Implementation below works similar to the one above, but much slower
    # Based on algorithm from histogram lecture
    """
    matched = np.zeros_like((gray_src))
    for i in range(row):
        for j in range(col):
            pixel_value = gray_src[i, j]
            new_pixel = 0
            while nc_day_gray[pixel_value] > nc_night_gray[new_pixel]:
                new_pixel += 1
            matched[i, j] = new_pixel
    """
    # Create 3 subplots (20 x 20 in size) in one row
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))
    # Display the gray source image
    axes[0].imshow(gray_src, cmap="gray")
    axes[0].set_title("Source")

    # Display the gray reference image
    axes[1].imshow(gray_ref, cmap="gray")
    axes[1].set_title("Reference")

    # Display the new matched image
    axes[2].imshow(matched, cmap="gray")
    axes[2].set_title("Matched")
    # Display the figures
    plt.show()

    # Similar implementation to above and displays the RGB images instead
    rgb_hist_matching(src_jpg, ref_jpg)


if __name__ == '__main__':
    part1_histogram_compute()
    part2_histogram_equalization()
    part3_histogram_comparing()
    part4_histogram_matching()
