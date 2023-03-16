from PIL import Image
import numpy as np
import math

# HW2 - Part 1 : Make odd * odd matrix
def boxfilter(n):
  assert n % 2 != 0, 'Dimension must be odd'

  # 2차원 배열을 생성하는 과정입니다.
  # 먼저 n^2 길이의 1차원 배열을 생성합니다.
  # np.reshape 함수를 통해 해당 길이를 2차원으로 변환합니다.
  # ex) 9 * 1 ==> 3 * 3
  result = np.array([0.04] * (n**2)).reshape((n, n))
  return result

# HW2 - Part1 : 1D Gaussian Filter
# Filter를 만들기 위해 Gaussian 분포 식을 미리 정의합니다.
def density_function(x, sigma):
  return math.exp(-x**2 / (2 * (sigma**2)))

def gauss1d(sigma):
  length = round(sigma * 6)

  # 문제에 언급된 대로 sigma * 6을 반올림한 첫 홀수로 길이를 정합니다.
  if(length % 2 == 0):
    length = length + 1
  
  # 1차원 배열 생성 단계입니다.
  # 0을 기준으로 대칭 배열을 생성하기 위해 np.arange 함수를 이용합니다.
  gauss_arr = np.arange(-(length//2), length//2 + 1, 1)

  # 문제에 언급된 대로 for loop을 피하기 위해 np.vectorize를 이용합니다.
  # for loop 없이 모든 배열에 density_function을 mapping하여 반환합니다.
  mapped_gauss_arr = np.vectorize(density_function)(gauss_arr, sigma)

   # 나온 결과를 Filter로써 이용하기 위해 합을 1로 만드는 일반화 과정을 진행합니다.
  normalized_gauss_arr = mapped_gauss_arr / np.sum(mapped_gauss_arr)

  return normalized_gauss_arr

# HW2 - Part 1 : 2D Gaussian Filter
def gauss2d(sigma):
  length = round(sigma * 6)

  # 1차원 배열과 마찬가지로 filter의 길이를 계산합니다.
  if(length % 2 == 0):
    length = length + 1

  # np.outer(외적) 함수를 통해 1차원 배열을 2차원으로 변환합니다.
  gauss_arr = np.arange(-(length//2), length//2 + 1, 1)
  gauss_arr_2d = np.outer(gauss_arr, gauss_arr)

  # 1차원 배열을 계산한 방식과 동일합니다.
  mapped_gauss_arr_2d = np.vectorize(density_function)(gauss_arr_2d, sigma)
  normalized_gauss_arr_2d = mapped_gauss_arr_2d / np.sum(mapped_gauss_arr_2d)

  return normalized_gauss_arr_2d

# HW2 - Part 1 : Convolution with gaussian filter
#(a) Implement convolution
def convolution_function(curr_x, curr_y, kernel_size, padded_arr, filter):
  filtered_value = 0

  for i in range(-kernel_size, kernel_size + 1):
    for j in range(-kernel_size, kernel_size + 1):
      filtered_value += (padded_arr[i + curr_x][j + curr_y] * filter[i][j])

  return filtered_value

def convolve2d(array, filter):
  # filter[0].size는 곧 kernel의 크기가 됩니다.
  # np.pad 함수를 이용해 (filter size - 1)/2 만큼 배열 주변을 0으로 채웁니다.
  pad_value = (filter[0].size - 1) // 2
  padded_arr = np.pad(array, ((pad_value, pad_value), (pad_value, pad_value)), 'constant', constant_values=0)

  # Convolution은 본래 배열을 180도 회전시킨 뒤, Cross correlation을 진행합니다.
  # 따라서 np.rot90 함수를 통해 회전함을 명시적으로 나타냅니다.
  # 하지만, 사실 해당 Filter은 회전을 해도 같은 배열이라 회전하지 않아도 같습니다.
  for i in range(1, len(array) + pad_value):
    for j in range(1, array[0].size + pad_value):
      array[i-1][j-1] = convolution_function(i, j, pad_value, padded_arr, np.rot90(filter, 2))

#(b) Implement covolution after gaussian filter
def gaussconvolve2d(array, sigma):
  gauss_filter = gauss2d(sigma)
  convolve2d(array, gauss_filter)

#(c) Apply filter to real image
# dog_image = Image.open('2b_dog.bmp')
# dog_arr = np.asarray(dog_image)
# filtered_dog_arr = gaussconvolve2d(dog_arr, 3)
# filtered_dog_image = Image.fromarray(filtered_dog_arr)

#(d) Show origin and filtered image

gaussconvolve2d(np.array([[1, 0, 0, 0, 1], [2, 3, 0, 8, 0], [2, 0, 0, 0, 3], [0, 0, 1, 0, 0]]), 0.5)
