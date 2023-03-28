from PIL import Image
import math
import numpy as np

"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
"""
def check_outlier(array):
   for i in range (0, len(array)):
    for j in range (0, len(array[0])):
      if(array[i][j] > 255):
        array[i][j] = 255
      if(array[i][j] < 0):
        array[i][j] = 0

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

def convolve2d(array,filter):
  kernel_size = len(filter)
  pad_value = (kernel_size - 1) // 2
  padded_arr = np.pad(array, ((pad_value, pad_value), (pad_value, pad_value)), 'constant', constant_values=0)
  rotated_filter = np.rot90(filter, 2)

  for i in range(len(array)):
    for j in range(len(array[0])):
      array[i][j] = np.sum(padded_arr[i : i + kernel_size, j : j + kernel_size] * rotated_filter)

def gaussconvolve2d(array,sigma):
  gauss_filter = gauss2d(sigma)
  convolve2d(array, gauss_filter)

def sobel_filters(img):
    x_sobel_filters = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y_sobel_filters = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    differential_iguana_x = img.copy()
    differential_iguana_y = img.copy()

    convolve2d(differential_iguana_x, x_sobel_filters)
    convolve2d(differential_iguana_y, y_sobel_filters)

    G = np.hypot(differential_iguana_x, differential_iguana_y)
    theta = np.arctan2(differential_iguana_x, differential_iguana_y)
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """
    return (G, theta)

def non_max_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
    pass
    return res

def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """
    pass
    return res

def dfs(img, res, i, j, visited=[]):
    # 호출된 시점의 시작점 (i, j)은 최초 호출이 아닌 이상 
    # strong 과 연결된 weak 포인트이므로 res에 strong 값을 준다
    res[i, j] = 255

    # 이미 방문했음을 표시한다
    visited.append((i, j))

    # (i, j)에 연결된 8가지 방향을 모두 검사하여 weak 포인트가 있다면 재귀적으로 호출
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)

def hysteresis(img):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """
    pass
    return res

#  1. Apply gausian filter with sigma 1.6
# 먼저 Greysacle로 image를 변환한 뒤, 이전 과제와 마찬가지로 진행한다.
# Type을 float32로 바꿔 계산한 뒤, 결과를 uint8로 바꾸어 이미지로 출력한다.
# Image.fromarray() 함수는 uint8이 아니면 동작하지 않는다.
iguana_image = Image.open('./iguana.bmp').convert('L')
iguana_array = np.asarray(iguana_image)
copied_iguana_array = iguana_array.copy()
copied_iguana_array.astype(np.float32)

gaussconvolve2d(copied_iguana_array, 1.6)
copied_iguana_array.astype(np.uint8)
filtered_iguana_image = Image.fromarray(copied_iguana_array)
filtered_iguana_image.save('./filtered_iguana.png', 'PNG')

# 2. Apply sobel operator to image
iguana_array_for_sobel = iguana_array.copy()
(iguana_magnitude, iguana_theta) = sobel_filters(iguana_array_for_sobel)

iguana_magnitude = np.array(iguana_magnitude).astype(np.uint8)
iguana_theta = np.array(iguana_theta).astype(np.uint8)
check_outlier(iguana_magnitude)

iguana_magnitude_image = Image.fromarray(iguana_magnitude)
iguana_theta_image = Image.fromarray(iguana_theta)
iguana_magnitude_image.save('./iguana_magnitude.png', 'PNG')
iguana_theta_image.save('./iguana_theta.png', 'PNG')
