from PIL import Image
import numpy as np
import math

def density_function(x, sigma):
  return math.exp(-x**2 / (2 * (sigma**2)))

def gauss2d(sigma):
  length = round(sigma * 6)

  if(length % 2 == 0):
    length = length + 1

  gauss_arr = np.arange(-(length//2), length//2 + 1, 1)
  gauss_arr_2d = np.outer(gauss_arr, gauss_arr)

  mapped_gauss_arr_2d = np.vectorize(density_function)(gauss_arr_2d, sigma)
  normalized_gauss_arr_2d = mapped_gauss_arr_2d / np.sum(mapped_gauss_arr_2d)

  return normalized_gauss_arr_2d

def convolution_function(curr_x, curr_y, kernel_size, padded_arr, filter):
  filtered_value = 0

  for i in range(-kernel_size, kernel_size + 1):
    for j in range(-kernel_size, kernel_size + 1):
      filtered_value += (padded_arr[i + curr_x][j + curr_y] * filter[i][j])

  return filtered_value

def convolve2d(array, filter):
  pad_value = (filter[0].size - 1) // 2
  padded_arr = np.pad(array, ((pad_value, pad_value), (pad_value, pad_value)), 'constant', constant_values=0)

  for i in range(pad_value, len(array) + pad_value):
    for j in range(pad_value, array[0].size + pad_value):
      array[i-pad_value][j-pad_value] = convolution_function(i, j, pad_value, padded_arr, np.rot90(filter, 2))

def gaussconvolve2d(array, sigma):
  gauss_filter = gauss2d(sigma)
  convolve2d(array, gauss_filter)

# HW2 - Part 2 : Apply gaussian filter to Monroe
monroe_image = Image.open('./hw2_image/0b_marilyn.bmp').convert('L')
monroe_arr = np.asarray(monroe_image)
copied_monroe_arr = monroe_arr.copy()

gaussconvolve2d(copied_monroe_arr, 1.6)
filtered_monroe_image = Image.fromarray(copied_monroe_arr)
filtered_monroe_image.save('./hw2_image/filtered_monroe.png', 'PNG')

# HW2 - Part 2 : Apply gaussian filter to Einstein
albert_image = Image.open('./hw2_image/0a_einstein.bmp').convert('L')
albert_arr = np.asarray(albert_image)
copied_albert_arr = albert_arr.copy()
low_filtered_albert_arr = albert_arr.copy()

gaussconvolve2d(low_filtered_albert_arr, 1.6)
copied_albert_arr = copied_albert_arr - low_filtered_albert_arr
high_frequency_albert_image = Image.fromarray(copied_albert_arr)
high_frequency_albert_image.save('./hw2_image/high_frequency_albert.png', 'PNG')

# HW2 - Part 2 : Hybrid them
hybrid_arr = copied_monroe_arr + copied_albert_arr
hybrid_image = Image.fromarray(hybrid_arr)
hybrid_image.save('./hw2_image/hybrid_image.png', 'PNG')
