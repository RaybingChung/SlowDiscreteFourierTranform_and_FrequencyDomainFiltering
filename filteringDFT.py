import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


def dft(spatial_img):
    height, width = spatial_img.shape
    mask = np.zeros((height, width), "complex128")
    # prepare processed index corresponding to original image positions
    x = np.tile(np.arange(width), (height, 1))
    y = np.arange(height).repeat(width).reshape(height, -1)

    # dft
    for v in range(height):
        for u in range(width):
            mask[v, u] = np.sum(spatial_img * np.exp(-2j * np.pi * (x * u / width + y * v / height)))

    return mask


def show_spectrum(dft_in):
    return np.log(np.sqrt(np.power(dft_in.real, 2) + np.power(dft_in.imag, 2) + 1))


def inv_dft(fre_img):
    # prepare out image
    height, width = fre_img.shape
    out = np.zeros((height, width), "complex128")

    # prepare processed index corresponding to original image positions
    x = np.tile(np.arange(width), (height, 1))
    y = np.arange(height).repeat(width).reshape(height, -1)

    # idft
    for v in range(height):
        for u in range(width):
            out[v, u] = (np.sum(fre_img * np.exp(2j * np.pi * (x * u / width + y * v / height))) / (width * height))

    # clipping
    # out = np.clip(out, 0, 255)
    out = out.astype(int)
    # print(out)
    return out


def padding(img):
    # print(img.shape)
    height, weight = img.shape
    out = np.zeros((2 * height, 2 * weight))
    out[:height, :weight] = img
    return out


def centering(img):
    # print(img.shape)
    out = img * np.fromfunction(lambda x, y: np.power(-1, x + y), img.shape)

    # print(np.fromfunction(lambda x, y: np.power(-1, x+y), img.shape))
    return out


def generate_low_pass_filter(imgDft, radius):
    rows, cols = imgDft.shape
    center = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius * radius
    mask[mask_area] = 1
    return mask


def generate_gaussin_high_pass_filter(imgDft, radius):
    rows, cols = imgDft.shape
    center = int(rows / 2), int(cols / 2)
    x, y = np.ogrid[:rows, :cols]
    mask = 1 - np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2)/(2*radius*radius))
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 > radius * radius
    mask[mask_area] = 1
    return mask


def generate_gaussin_low_pass_filter(imgDft, radius):
    rows, cols = imgDft.shape
    center = int(rows / 2), int(cols / 2)
    x, y = np.ogrid[:rows, :cols]
    mask = np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2)/(2*radius*radius))
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 > radius * radius
    mask[mask_area] = 0
    return mask

def generate_high_pass_filter(imgDft, radius):
    rows, cols = imgDft.shape
    center = int(rows / 2), int(cols / 2)

    mask = np.ones((rows, cols), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius * radius
    mask[mask_area] = 0
    return mask


def process_by_filter_in_spectreum(img_path, fileter, size_proportion2origin):
    img = cv.imread(img_path, flags=cv.IMREAD_GRAYSCALE)

    img_padded = padding(img)
    img_padded_centered = centering(img_padded)
    img_dft = dft(img_padded_centered)
    img4show = show_spectrum(img_dft)

    # fre_filter = generate_high_pass_filter(img_dft, min(img_dft.shape) // 10)
    if fileter == 'low':
        fre_filter = generate_low_pass_filter(img_dft, int(min(img_dft.shape) * size_proportion2origin))
    else:
        fre_filter = generate_high_pass_filter(img_dft, int(min(img_dft.shape) * size_proportion2origin))
    fre_filter4show = show_spectrum(fre_filter)

    fre_filtered = np.multiply(img_dft, fre_filter)
    fre_filtered4show = show_spectrum(fre_filtered)

    processed_img = centering(inv_dft(fre_filtered))[:img.shape[0], :img.shape[1]]
    # processed_img = centering(inv_dft(fre_filtered))

    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 2)
    plt.imshow(img_padded_centered, cmap='gray')
    plt.title('Padded Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 3)
    plt.imshow(img4show, cmap='gray')
    plt.title('Spectrum of  image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 4)
    plt.imshow(fre_filter, cmap='gray')
    plt.title('Spectrum Filter'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 5)
    plt.imshow(fre_filtered4show, cmap='gray')
    plt.title('Filtered Spectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 6)
    plt.imshow(processed_img, cmap='gray')
    plt.title('Output'), plt.xticks([]), plt.yticks([])

    # plt.show()
    plt.savefig(img_path.split('\\')[-1].split('.')[0] + '_' + fileter + '_' + str(size_proportion2origin) + ".svg")
    print(img_path.split('\\')[-1].split('.')[0] + '_' + fileter + '_' + str(size_proportion2origin) + ".svg finished!")


if __name__ == '__main__':
    # for file in os.listdir('.'):
    #     if file.split('.')[-1] == "bmp":
    #         process_by_filter_in_spectreum(file, fileter="low", size_proportion2origin=1 / 5)
    #         process_by_filter_in_spectreum(file, fileter="high", size_proportion2origin=1 / 5)
    #         process_by_filter_in_spectreum(file, fileter="low", size_proportion2origin=1 / 3)
    #         process_by_filter_in_spectreum(file, fileter="high", size_proportion2origin=1 / 3)

    img = cv.imread("butterfly.bmp", flags=cv.IMREAD_GRAYSCALE)

    img_padded = padding(img)
    img_padded_centered = centering(img_padded)
    img_dft = dft(img_padded_centered)
    img4show = show_spectrum(img_dft)

    # fre_filter = generate_high_pass_filter(img_dft, min(img_dft.shape) // 10)
    fre_filter = generate_gaussin_low_pass_filter(img_dft, min(img_dft.shape) // 4)
    fre_filter4show = show_spectrum(fre_filter)

    fre_filtered = np.multiply(img_dft, fre_filter)
    fre_filtered4show = show_spectrum(fre_filtered)

    processed_img = centering(inv_dft(fre_filtered))[:img.shape[0], :img.shape[1]]
    print(processed_img)
    # processed_img = centering(inv_dft(fre_filtered))

    plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
    plt.title('Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 2), plt.imshow(img_padded_centered, cmap='gray')
    plt.title('Padded Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 3), plt.imshow(img4show, cmap='gray')
    plt.title('Spectrum of  image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 4), plt.imshow(fre_filter, cmap='gray')
    plt.title('Spectrum Filter' ), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 5), plt.imshow(fre_filtered4show, cmap='gray')
    plt.title('Filtered Spectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 6), plt.imshow(processed_img, cmap='gray')
    plt.title('Output'), plt.xticks([]), plt.yticks([])
    #
    plt.show()
    # plt.savefig("1.svg")
