from PIL import Image
import math
import numpy as np

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

  # 불필요한 Copy 함수의 사용을 줄이고자, return하는 방식으로 변경했습니다.
  result = np.zeros(array.shape, dtype = np.float32)

  for i in range(len(array)):
    for j in range(len(array[0])):
      result[i][j] = np.sum(padded_arr[i : i + kernel_size, j : j + kernel_size] * rotated_filter)

  return result

def gaussconvolve2d(array,sigma):
  gauss_filter = gauss2d(sigma)
  result = convolve2d(array, gauss_filter)

  return result

def sobel_filters(img):
    # X축, Y축 Sobel Kernel을 각각 선언한다.
    # Convolution은 Filter를 180도 회전하기 때문에, 반전된 Kernel을 선언한다.
    x_sobel_filters = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    y_sobel_filters = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    # Image의 X축 Y축 변화량 검출을 위해 Convolution을 진행한다.
    iguana_x_gradient = convolve2d(img, x_sobel_filters)
    iguana_y_gradient = convolve2d(img, y_sobel_filters)
    # 나온 결과를 합치기 위해 직각 삼각형 빗변 함수를 이용한다.
    # 그리고 올바른 magnitude를 구하기 위해 값을 255 범위 내로 Mapping 해주어야 한다.
    # HW02의 과제 설명에서 언급된 대로 np.where()을 사용합니다.
    # np.where()의 기본 기능은 index(좌표)의 반환이지만, 아래의 형태는 값을 바꿔 배열을 return한다.
    G = np.hypot(iguana_x_gradient, iguana_y_gradient)
    G = np.where(G < 0, 0, G)
    G = np.where(G > 255, 255, G)
    #G = G/G.max() * 255

    # 그리고, 각 Pixel들의 변화 방향을 알기 위해 arctan2 함수를 이용한다.
    theta = np.arctan2(iguana_x_gradient, iguana_y_gradient)
    return (G, theta)

def non_max_suppression(G, theta):
    # 0, 45, 90, 135를 검사하는 것은 주변 8방향의 Intensity를 모두 검사하기 위함이다.
    # 먼저, Radian 값을 각도로 돌리기 위해 아래의 식을 이용한다.
    angle = theta / 180 * np.pi
    res = np.zeros(G.shape, dtype = np.float32)

    for i in range(1, len(G)-1):
       for j in range(1, len(G[0])-1):
            # 중심 점 기준 0도 선분이 좌우 점 (0도)
            if (0 <= angle[i][j] < 22.5) or (157.5 <= angle[i][j] <= 180) or (-22.5 <= angle[i][j] < 0) or (-180 <= angle[i][j] < -157.5):
                left_ptr = G[i][j+1]
                right_ptr = G[i][j-1]
            # 중심 점 기준 45도 선분의 좌우 점
            elif (22.5 <= angle[i][j] < 67.5) or (-157.5 <= angle[i][j] < -112.5):
                left_ptr = G[i+1][j+1]
                right_ptr = G[i-1][j-1]
            # 중심 점 기준 90도 선분의 좌우 점
            elif (67.5 <= angle[i][j] < 112.5) or (-112.5 <= angle[i][j] < -67.5):
                left_ptr = G[i+1][j]
                right_ptr = G[i-1][j]
            # 중심 점 기준 135도 선분의 좌우 점
            elif (112.5 <= angle[i][j] < 157.5) or (-67.5 <= angle[i][j] < -22.5):
                left_ptr = G[i+1][j-1]
                right_ptr = G[i-1][j+1]  
          
            if(G[i][j] >= left_ptr) and (G[i][j] >= right_ptr):
                res[i][j] = G[i][j]
            else:
               res[i][j] = 0

    return res

def double_thresholding(img):
    # 문제에 나온 역치 값들을 미리 정의한다.
    diff = img.max() - img.min()
    high_thresh = img.min() + diff * 0.15
    low_thresh = img.min() + diff * 0.03
    strong_constant = 255
    weak_constant = 80

    # 결과를 담을 배열을 만든다.
    res = np.zeros(img.shape, dtype = np.float32)
    # np.where의 기본 기능을 이용해 역치 범위의 좌표들을 뽑아낸다.
    strong_x, strong_y = np.where(img >= high_thresh)
    weak_x, weak_y = np.where((img >= low_thresh) & (img < high_thresh))

    # Numpy의 기능으로 아래와 같이 assign을 할 수 있다.
    # Strong_x, Strong_y 배열에 해당되는 모든 좌표에 한 번에 assign한다.
    res[strong_x, strong_y] = strong_constant
    res[weak_x, weak_y] = weak_constant

    return res

def dfs(img, res, i, j, visited=[]):
    # 호출된 시점의 시작점 (i, j)은 최초 호출이 아닌 이상 
    # strong 과 연결된 weak 포인트이므로 res에 strong 값을 준다
    if(len(visited) > 0):
        res[i, j] = 255

    # 이미 방문했음을 표시한다
    visited.append((i, j))

    # (i, j)에 연결된 8가지 방향을 모두 검사하여 weak 포인트가 있다면 재귀적으로 호출
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)
                # 더 이상 연결된 Strong이 없는 pixel은 0으로 만들어준다.
                # 이렇게 0으로 만들어 주지 않으면 너무 많은 pixel이 계속 연결된다.
                res[ii, jj] = 0

def hysteresis(img):
    res = img.copy()
    
    # DFS를 왼쪽 위 부터 점마다 모두 진행해준다.
    for i in range(1, len(img) - 1):
       for j in range(1, len(img[0]) - 1):
                dfs(img, res, i, j, visited=[])

    return res

#  1. Apply gausian filter with sigma 1.6
# 먼저 Greysacle로 image를 변환한 뒤, 이전 과제와 마찬가지로 진행한다.
# Type을 float32로 바꿔 계산한 뒤, 결과를 uint8로 바꾸어 이미지로 출력한다.
# Image.fromarray() 함수는 uint8이 아니면 동작하지 않는다.
iguana_image = Image.open('./iguana.bmp').convert('L')
iguana_array = np.asarray(iguana_image)
copied_iguana_array = iguana_array.copy()
# astype('float32')가 동작하지 않는 문제가 확인되어 astype(np.float32) 형식으로 모두 교체 했습니다.
copied_iguana_array.astype(np.float32)

copied_iguana_array = gaussconvolve2d(copied_iguana_array, 1.6)
filtered_iguana_image = Image.fromarray(copied_iguana_array.astype(np.uint8))
filtered_iguana_image.save('./filtered_iguana.png', 'PNG')

# 2. Apply sobel operator to image
# 위에서 Gaussian filter를 통해 noise를 제거한 사진을 가져온다.
# 해당 사진의 X축 Gradient와 Y축 Gradient를 구해야 하는데, 
# Sobel filter를 사진과 Convolution 하면, X축 Y축 변화량을 각각 구할 수 있다.
(iguana_magnitude, iguana_theta) = sobel_filters(copied_iguana_array.astype(np.float32))
iguana_magnitude_image = Image.fromarray(iguana_magnitude.astype(np.uint8))
iguana_magnitude_image.save('./iguana_magnitude.png', 'PNG')

# 3. Apply non-max suppression to image
# Sobel filter를 적용한 사진에서 얇은 Edge만 따내기 위한 작업이다.
# 3x3에서 주변 8방향 값보다 작은 중심 값을 제거하는 과정이다.
non_max_iguana = non_max_suppression(iguana_magnitude.astype(np.float32), iguana_theta.astype(np.float32))
non_max_iguana_image = Image.fromarray(non_max_iguana.astype(np.uint8))
non_max_iguana_image.save('./non_max_iguana.png', 'PNG')

# 4. Double thresholding
# 2개의 역치 값을 두어, Strong edge, weak edge로 구별한다.
# Edge를 좀 더 선명하게 강화하는 작업이다.
thresholded_iguana = double_thresholding(non_max_iguana.astype(np.float32))
thresholded_iguana_image = Image.fromarray(thresholded_iguana.astype(np.uint8))
thresholded_iguana_image.save('./thresholded_iguana.png', 'PNG')

# 5. Hysteresis(이력 현상)
# 마지막 단계인 Edge tracking 단계이다.
# 8방향을 검사해, 인접한 Edge가 Strong Edge이면?
# Weak edge를 Strong edge를 바꾸는 작업을 dfs로 진행한다.
iguana_edge = hysteresis(thresholded_iguana.astype(np.float32))
iguana_edge_image = Image.fromarray(iguana_edge.astype(np.uint8))
iguana_edge_image.save('./iguana_edge.png', 'PNG')
