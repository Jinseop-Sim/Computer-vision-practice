from PIL import Image
import numpy as np
import math

def density_function(x, sigma):
  return math.exp(-x**2 / (2 * (sigma**2)))

def gauss1d(sigma):
  length = round(sigma * 6)

  if(length % 2 == 0):
    length = length + 1

  gauss_arr = np.arange(-(length//2), length//2 + 1, 1)
  mapped_gauss_arr = np.vectorize(density_function)(gauss_arr, sigma)
  normalized_gauss_arr = mapped_gauss_arr / np.sum(mapped_gauss_arr)

  return normalized_gauss_arr

def gauss2d(sigma):
  gauss_arr_2d = np.outer(gauss1d(sigma), gauss1d(sigma))
  normalized_gauss_arr_2d = gauss_arr_2d / np.sum(gauss_arr_2d)

  return normalized_gauss_arr_2d

def convolve2d(array, filter):
  kernel_size = len(filter)
  pad_value = (kernel_size - 1) // 2
  padded_arr = np.pad(array, ((pad_value, pad_value), (pad_value, pad_value)), 'constant', constant_values=0)
  rotated_filter = np.rot90(filter, 2)

  for i in range(len(array)):
    for j in range(len(array[0])):
      array[i][j] = np.sum(padded_arr[i : i + kernel_size, j : j + kernel_size] * rotated_filter)

def gaussconvolve2d(array, sigma):
  gauss_filter = gauss2d(sigma)
  convolve2d(array, gauss_filter)

# HW2 - Part 2 : Apply gaussian filter to Monroe
monroe_image = Image.open('./hw2_image/0b_marilyn.bmp').convert('L')
monroe_arr = np.asarray(monroe_image)
copied_monroe_arr = monroe_arr.copy()
copied_monroe_arr.astype('float32')

gaussconvolve2d(copied_monroe_arr, 3)
copied_monroe_arr.astype('uint8')
filtered_monroe_image = Image.fromarray(copied_monroe_arr)
filtered_monroe_image.save('./hw2_image/filtered_monroe.png', 'PNG')

# HW2 - Part 2 : Apply gaussian filter to Einstein
albert_image = Image.open('./hw2_image/0a_einstein.bmp').convert('L')
albert_arr = np.asarray(albert_image)
copied_albert_arr = albert_arr.copy()
low_filtered_albert_arr = albert_arr.copy()
copied_albert_arr.astype('float32')
low_filtered_albert_arr.astype('float32')

# Gauss filter를 적용해 low frequency component만 먼저 납깁니다.
# 이후 high frequency component의 추출을 위해 두 배열을 빼줍니다.
# 여기서 주의할 점은 문제에 언급된 대로 greyscale의 범위입니다.
# 0 ~ 255 까지의 값을 가져야 하기 때문에 음수를 막기 위해 128을 더합니다.
gaussconvolve2d(low_filtered_albert_arr, 2)
copied_albert_arr = copied_albert_arr - (low_filtered_albert_arr + 128)
copied_albert_arr.astype('uint8')
high_frequency_albert_image = Image.fromarray(copied_albert_arr)
high_frequency_albert_image.save('./hw2_image/high_frequency_albert.png', 'PNG')

# HW2 - Part 2 : Hybrid them
# 이번엔 두 사진을 더하는데, 이전에 원본 Einstien에 128을 더했습니다.
# 따라서 255를 넘어갈 위험이 있으므로, 이번엔 다시 128을 빼줌으로써 처리합니다.
hybrid_arr = copied_monroe_arr + (copied_albert_arr - 128)

# 문제에 언급된 말을 보면 "Speckle artifacts" 라는 말이 있습니다.
# 이는 범위를 벗어나 image에 맞지 않는 이상치를 의미합니다.
# 해당 값을 처리하기 위해서는 intensity를 0 ~ 255 사이의 값으로 조정하라고 합니다.
for i in range(len(hybrid_arr)):
  for j in range(len(hybrid_arr[0])):
    if hybrid_arr[i][j] > 255:
      hybrid_arr[i][j] = 255
    if hybrid_arr[i][j] < 0:
      hybrid_arr[i][j] = 0

hybrid_image = Image.fromarray(hybrid_arr)
hybrid_image.save('./hw2_image/hybrid_image.png', 'PNG')
