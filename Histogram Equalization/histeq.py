import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import cv2

raw_image_list=[]
def read_jpeg_files_in_folder(folder_path):
    """
    Read JPEG (.jpeg and .jpg) files in a folder.

    Args:
    folder_path (str): The path to the folder containing JPEG files.

    Returns:
    list of PIL.Image.Image: A list of PIL Image objects representing the JPEG files.
    """
    jpeg_files = []

    # Ensure the folder path is valid
    if not os.path.isdir(folder_path):
        raise ValueError("The provided folder path is not valid.")

    # List all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate through each file
    for file_name in file_list:
        # Check if the file has a .jpeg or .jpg extension (case-insensitive)
        if file_name.lower().endswith(('.jpeg', '.jpg')):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Open the image using Pillow
                image = Image.open(file_path)
                jpeg_files.append(image)
            except Exception as e:
                print(f"Error opening {file_name}: {str(e)}")

    return jpeg_files
def image_prep(img):
    im1 = ImageOps.grayscale(img)

    # convert image into a numpy array
    img = np.asarray(img)
    img_grey = np.asarray(im1)
    # print(img.shape)
    img_red= img[:,:,0]

    # print(img_red.shape)
    img_green= img[:,:,1]

    # print(img_blue.shape)
    img_blue= img[:,:,2]
    ph_list=[img_grey,img_red,img_green,img_blue]
    return ph_list


raw_image_list=read_jpeg_files_in_folder('./sample images')
sep_image_list=[]
for pic in raw_image_list:
    sep_image_list.append(image_prep(pic))



def histogram_equalization(image):

    # Calculate the histogram of the input image
    histogram, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))

    # Calculate the cumulative distribution function (CDF) of the histogram
    cdf = histogram.cumsum()
    cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
    cdf_normalized= cdf_normalized.astype('uint8')

    # Apply histogram equalization to the image
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    equalized_image = equalized_image.reshape(image.shape)

    # # Clip the values to be within [0, 255]
    equalized_image = np.clip(equalized_image, 0, 255).astype(np.uint8)

    return equalized_image

def grey_jux(pic_b,pic_a,map,i):
    # Display the original and equalized images
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(pic_b, cmap=map)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Equalized Image")
    plt.imshow(pic_a, cmap=map)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Original Histogram")
    plt.hist(pic_b.flatten(),bins=255)

    plt.subplot(2, 2, 4)
    plt.title("Equalized Histogram")
    plt.hist(pic_a.flatten(), bins=255)

    #plt.show()
    plt.savefig('bw_fig' + str(i) + '.jpg')

def col_jux(pic_b,pic_a,map,j,k,l,phr,phg,phb,i):
    # Display the original and equalized images
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(pic_b, cmap=map)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Equalized Image")
    plt.imshow(pic_a, cmap=map)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Original Histogram")
    plt.hist(j.flatten(), bins=255, color='r', alpha=0.5, label="Red")
    plt.hist(k.flatten(), bins=255, color='g', alpha=0.5, label="Green")
    plt.hist(l.flatten(), bins=255, color='b', alpha=0.5, label="Blue")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title("Equalized Histogram")
    plt.hist(phr.flatten(), bins=255,color='r',alpha=0.5,label="Red")
    plt.hist(phg.flatten(), bins=255,color='g',alpha=0.5,label="Green")
    plt.hist(phb.flatten(), bins=255,color='b',alpha=0.5,label="Blue")
    plt.legend()

    #plt.show()
    plt.savefig('col_fig'+str(i)+'.jpg')



def colour_combine(r,g,b):
    red_img = Image.fromarray(r).convert("L")
    green_img = Image.fromarray(g).convert("L")
    blue_img = Image.fromarray((b)).convert("L")
    ci=Image.merge("RGB", (red_img, green_img, blue_img))
    return ci
#for greyscale
for i in range(len(sep_image_list)):
    grey_jux(sep_image_list[i][0],histogram_equalization(sep_image_list[i][0]),'gray',i)
#for RGB
    phr=histogram_equalization(sep_image_list[i][1])
    phg=histogram_equalization(sep_image_list[i][2])
    phb=histogram_equalization(sep_image_list[i][3])
    col_jux(raw_image_list[i],colour_combine(phr,phg,phb),'viridis',sep_image_list[i][1],sep_image_list[i][2],sep_image_list[i][3],phr,phg,phb,i)