import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import sys
# to save and write as pngs
from imageio import imwrite, imread


# filename = sys.argv[1];
# im_obj = sitk.ReadImage(filename);
im_obj = sitk.ReadImage('data/IMG-0002-00074.dcm');
print("The type of 'image obj' is {}.".format(type(im_obj)));

# Too see which information you might retrieve from an 'Image' object, you can use the command 'dir'.
# It will print a lot of functions. However, you should focus on those on the top (without the trailing '__').
# dir(sitk.Image);

print("\nSize: {}".format(im_obj.GetSize()))
print("Spacing: {}".format(im_obj.GetSpacing()))
print("Dimension: {}".format(im_obj.GetDimension()))
print("Depth: {}".format(im_obj.GetDepth()))
print("Height: {}".format(im_obj.GetHeight()))
print("Width: {}".format(im_obj.GetWidth()))
print("NumberOfComponentsPerPixel: {}".format(im_obj.GetNumberOfComponentsPerPixel()))
# print(im_obj)

# Use the GetArrayViewFromImage function to plot the loaded image with pyplot's imshow
im_array = sitk.GetArrayViewFromImage(im_obj)
print("Obtained image_array from DICOM file has typ= {} \n and has dimension= {} \n and shape= {}".format(type(im_array), im_array.ndim, im_array.shape));

# squeez extra dimensions
im_array_squeezed = im_array.squeeze()
print("\nAfter squeezing the dimension of image_array, image_array has shape= {}".format(im_array_squeezed.shape))

# plot the image
# plt.figure()
# plt.imshow(im_array_squeezed, cmap='gray')
# plt.show();
print(im_array_squeezed)
print('Total number of values in the image array: ',len(np.unique(im_array_squeezed)));
print('The maximum value is: {}, and the minimum value is: {}'.format(np.max(im_array_squeezed), np.min(im_array_squeezed)))
print('The mean value is: ', np.mean(im_array_squeezed));
# optional normalization
im_arrayNor = im_array_squeezed-np.mean(im_array_squeezed);
im_arrayNor = im_arrayNor/np.max(im_arrayNor);

# plt.figure()
# plt.imshow(im_arrayNor, cmap='gray')
# plt.show();
print('After normalizing (divide by max value) the array have values:\n', np.unique(im_arrayNor));
print('The maximum value is: {}, and the minimum value is: {}'.format(np.max(im_arrayNor), np.min(im_arrayNor)))

# eventuelly for saving as png
# imwrite(str("testImagio_dicom.png"), im_arrayNor/np.max(im_arrayNor))
# test = imread("testImagio_dicom.png");
# print('\n',np.unique(test))
