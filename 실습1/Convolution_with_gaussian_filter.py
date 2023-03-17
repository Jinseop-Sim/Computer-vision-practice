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
  print(result)

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
  # 위에서 정의한 1d array의 Gauss filter를 외적을 이용해 2d array로 변환 합니다.
  gauss_arr_2d = np.outer(gauss1d(sigma), gauss1d(sigma))
  normalized_gauss_arr_2d = gauss_arr_2d / np.sum(gauss_arr_2d)

  return normalized_gauss_arr_2d

# HW2 - Part 1 : Convolution with gaussian filter
#(a) Implement convolution
def convolve2d(array, filter):
  # len(filter)는 곧 kernel의 크기가 됩니다.
  kernel_size = len(filter)
  # np.pad 함수를 이용해 (filter size - 1)/2 만큼 배열 주변을 0으로 채웁니다.
  pad_value = (kernel_size - 1) // 2
  padded_arr = np.pad(array, ((pad_value, pad_value), (pad_value, pad_value)), 'constant', constant_values=0)

  # Convolution은 본래 배열을 180도 회전시킨 뒤, Cross correlation을 진행합니다.
  # 따라서 np.rot90 함수를 통해 회전함을 명시적으로 나타냅니다.
  # 하지만, 사실 해당 filter은 회전을 해도 동일한 형태를 유지하는 대칭 배열입니다.
  rotated_filter = np.rot90(filter, 2)

  # 처음엔 4중 for문을 통해 convolution을 계산해야 하나 싶었지만, 시간이 너무 오래걸렸습니다.
  # 따라서 numpy에서 제공하는 2d array slicing을 통해 kernel만큼 잘라 convolution을 진행합니다.
  for i in range(len(array)):
    for j in range(len(array[0])):
      array[i][j] = np.sum(padded_arr[i : i + kernel_size, j : j + kernel_size] * rotated_filter)

#(b) Implement covolution after gaussian filtering
# 위에서 이미 선언한 gauss2d 함수를 통해 filter를 생성합니다.
# 이후 image와 해당 filter에 대해 convolution을 진행합니다.
def gaussconvolve2d(array, sigma):
  gauss_filter = gauss2d(sigma)
  convolve2d(array, gauss_filter)

#(c) Apply filter to real image
# 먼저 filter는 2d array이기 때문에, greyscale로 image를 open합니다.
# np.asarray를 통해 배열로 바꾸는데, 해당 배열을 그대로 쓰니 immutable해 오류가 발생했습니다.
# 따라서 copy 함수를 통해 mutable한 배열을 생성해줍니다.
# 문제에 적힌 대로 float32로 explicit type casting을 진행합니다.
dog_image = Image.open('./hw2_image/2b_dog.bmp').convert('L')
dog_arr = np.asarray(dog_image)
copied_dog_arr = dog_arr.copy()
copied_dog_arr = copied_dog_arr.astype('float32')

# 생성된 배열에 gauss filter + convolution을 진행합니다.
# 문제에 적힌 대로 다시 uint8로 explicit type casting을 진행합니다.
# 이후 Local 환경에 filtering된 image를 저장합니다.
gaussconvolve2d(copied_dog_arr, 3)
copied_dog_arr = copied_dog_arr.astype('uint8')
filtered_dog_image = Image.fromarray(copied_dog_arr)
filtered_dog_image.save('./hw2_image/filtered_dog.png', 'PNG')

#(d) Show origin and filtered image
dog_image.show()
filtered_dog_image.show()
