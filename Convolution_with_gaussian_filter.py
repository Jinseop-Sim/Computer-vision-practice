from PIL import Image
import numpy as np
import math

# HW2 - Part 1 : Make odd * odd matrix
def boxfilter(n):
  assert n % 2 != 0, 'Dimension must be odd'

  # Create size of (n^2 * 1) 1d array 
  # and Reshape to size of (n * n) 2d array.
  result = np.array([0.04] * (n**2)).reshape((n, n))
  print(result)

# HW2 - Part1 : 1D Gaussian Filter
# Define density function to calculate values in array
def density_function(x, sigma):
  return math.exp(-x**2 / (2 * (sigma**2)))

def gauss1d(sigma):
  length = round(sigma * 6)

  # Calculate length of array (sigma * 6)
  if(length % 2 == 0):
    length = length + 1
  
  # Generate 1d array
  # Calculate quotient of length/2 to generate symmetric array.
  # and make "sigma_arr" to use built in fuction "Vectorize" in numpy.
  gauss_arr = np.arange(-(length//2), length//2 + 1, 1)

  # Use "Vectorize" function in numpy to avoid use "for" loop.
  # Divide all value of "mapped_gauss_arr" with sum of them
  # to normalize result of density function
  mapped_gauss_arr = np.vectorize(density_function)(gauss_arr, sigma)
  normalized_gauss_arr = mapped_gauss_arr / np.sum(mapped_gauss_arr)

  return normalized_gauss_arr

# HW2 - Part 1 : 2D Gaussian Filter
def gauss2d(sigma):
  length = round(sigma * 6)

  # Calculate length of array (sigma * 6)
  if(length % 2 == 0):
    length = length + 1

  # Use "Outer" function to convert 1d array to 2d array
  # and make "sigma_arr" to 2d array to calculate with "Vectorize"
  gauss_arr = np.arange(-(length//2), length//2 + 1, 1)
  gauss_arr_2d = np.outer(gauss_arr, gauss_arr)

  # Same as 1d array's calculate process
  mapped_gauss_arr_2d = np.vectorize(density_function)(gauss_arr_2d, sigma)
  normalized_gauss_arr_2d = mapped_gauss_arr_2d / np.sum(mapped_gauss_arr_2d)

  return normalized_gauss_arr_2d

# HW2 - Part 1 : Convolution with gaussian filter
#(a) Implement convolution
def convolve2d(array, filter):
  # Padded with number of "(filter size - 1)/2" zeros, by using "Pad" function
  pad_value = (filter[0].size - 1) // 2
  padded_arr = np.pad(array, ((pad_value, pad_value), (pad_value, pad_value)), 'constant', constant_values=0)

  print(padded_arr)

#(b) Implement covolution after gaussian filter
def gaussconvolve2d(array, sigma):
  gauss_filter = gauss2d(sigma)
  convolve2d(array, gauss_filter)

#(c) Apply filter to real image

convolve2d([[1, 0, 0, 0, 1], [2, 3, 0, 8, 0], [2, 0, 0, 0, 3], [0, 0, 1, 0, 0]], gauss2d(0.5))
