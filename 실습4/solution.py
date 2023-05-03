import numpy as np
import cv2
import math
import random

import hw_utils as utils
from PIL import Image

def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START
    # RANSAC은 반드시 10회 진행으로 고정합니다.
    largest_set = []
    for i in range(10):
        # SIFT의 결과로 등장한 matched_pair 중 random point를 고릅니다.
        rand = random.randrange(0, len(matched_pairs))
        choice = matched_pairs[rand]
        # Readkey 함수로부터 구해진 keypoint간의 각도와 Scale을 계산합니다.
        # Image 1에서 Image 2로 match 되었을 때, 변화량을 계산하는 것입니다.
        orientation = (keypoints1[choice[0]][3] - keypoints2[choice[1]][3]) % (2*math.pi)
        scale = keypoints2[choice[1]][2] / keypoints1[choice[0]][2]
        temp =[]
        # 모든 Matched pair에 대해서 변화량을 계산합니다.
        # 이 때, 앞에 선택했던 sample의 변화량과 threshold 이하의 차이가 난다면? inlier로 간주합니다.
        for j in range(len(matched_pairs)):
            if j is not rand:
                orientation_temp = (keypoints1[matched_pairs[j][0]][3] - keypoints2[matched_pairs[j][1]][3]) % (2*math.pi)
                scale_temp = keypoints2[matched_pairs[j][1]][2] / keypoints1[matched_pairs[j][0]][2]
                if((orientation*math.pi/6) < orientation_temp) and (orientation_temp < (orientation+math.pi/6)):
                    if(scale - scale*scale_agreement < scale_temp and scale_temp < scale + scale*scale_agreement):
                        temp.append([i, j])
        # inlier가 가장 많은 경우를 largest_set에 저장합니다.
        if(len(temp) > len(largest_set)):
            largest_set = temp
    # 최종 확정된 largest_set을 image에 반영한다.
    for i in range(len(largest_set)):
        largest_set[i] = (matched_pairs[largest_set[i][1]][0], matched_pairs[largest_set[i][1]][1])

    ## END
    assert isinstance(largest_set, list)
    return largest_set

def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    # hw_utils의 match() 함수에서 호출됩니다.
    y1 = descriptors1.shape[0]
    y2 = descriptors2.shape[0] # ReadKey 함수를 통해 탐색된 descriptor
    temp = np.zeros(y2)
    matched_pairs = []

    for i in range(y1): # Image descriptor1, 2를 서로 비교합니다.
        for j in range(y2):
            temp[j] = math.acos(np.dot(descriptors1[i], descriptors2[j]))
            # SIFT 방식으로 계산된 descriptor 간의 각도 차이를 계산합니다.
        compare = sorted(range(len(temp)), key = lambda k : temp[k])
        # 차이가 적은 matched pair 부터 정렬합니다.

        if(temp[compare[0]]/temp[compare[1]]) < threshold:
            matched_pairs.append((i, compare[0]))
        # 두 각도를 나누었을 때, 역치 이하로 차이가 나면 유사하다고 판단합니다.
    ## END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)
    # START
    # 결과 배열을 [xy points 갯수, 3] 만큼 1로 채워진 배열로 만든다.
    # 왜? Homogeneous coordiate를 표현하기 위해 (x, y, 1)로 만드는 것!
    xy_points_out = np.ones([len(xy_points), 3])

    # xy_points의 크기만큼 순회하며 아래의 연산을 진행한다.
    for i in range(len(xy_points)):
        xy_points_out[i][:2] = xy_points[i][:2] # 배열 값 복사
        xy_points_out[i] = h @ xy_points_out[i] # Homography와 행렬 곱 계산
        if(xy_points_out[i][2] == 0): # DVZ Exception을 방지하기 위한 correction
            xy_points_out[i][2] = 1e10
        xy_points_out[i] /= xy_points_out[i][2]
        # Homogeneous coordinate를 regular coordinate로 다시 되돌리기 위해 나눠준다.
        # (x', y', 1)의 형태가 될 것이며, 이후 마지막 열은 제거한다.
    xy_points_out = np.delete(xy_points_out, 2, axis = 1)
    # END
    return xy_points_out

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0

    # START
    # 이미 PrepareData 함수에서 BestMatch를 한 결과가 넘어오므로, 해당 과정은 생략한다.
    h = []
    max_inlier_cnt = 0
    # RANSAC과 유사한 방식으로 Optimal homography를 찾는다.
    for i in range(num_iter):
        inlier_cnt = 0
        diff = [] # xy_ref와 projection 시킨 점간의 거리
        A = [] # Ah = 0의 식에서, A를 담당할 빈 행렬
        # main_pano에서 넘어온 xy_src 중 4개의 random point를 선택
        for _ in range(4):
            # Random하게 index를 골라 4개의 좌표를 배열에 저장
            rand_idx = random.randrange(0, len(xy_src))
            # Ah = 0 공식을 기반으로 Homography를 계산한다
            # Random으로 뽑은 x, y, x', y'을 행렬식에 대입
            x, y = xy_src[rand_idx][0], xy_src[rand_idx][1]
            x_prime, y_prime = xy_ref[rand_idx][0], xy_ref[rand_idx][1]
            A.append([x, y, 1, 0, 0, 0, -x_prime*x, -x_prime*y, -x_prime])
            A.append([0, 0, 0, x, y, 1, -y_prime*x, -y_prime*y, -y_prime])
        
        # 본격적으로 Homography가 계산되는 부분이다.
        A = np.asarray(A)
        # 2n x 9의 직각 행렬 A에 대해 SVD를 계산한다.
        # 계산된 SVD 중, 직교행렬 V의 마지막 열이 Homography가 된다.
        U, S, V = np.linalg.svd(A)
        temp_h = V[-1].reshape((3,3))
        # 8개의 값을 마지막 값에 대해 나누어 준다.
        temp_h /= temp_h[2, 2]

        # Homography를 통해 src를 projection 시킨 점과 ref를 이동 시킨 점의 거리차이가 tol 이하면 inlier
        # inlier를 세어, 가장 많은 경우의 temp_h를 최종 Homography로 정한다.
        xy_proj_out = KeypointProjection(xy_src, temp_h)

        for i in range(len(xy_src)):
            diff.append(np.linalg.norm(xy_ref[i] - xy_proj_out[i]))
            if(diff[-1] < tol):
                inlier_cnt += 1

        if(inlier_cnt > max_inlier_cnt):
            max_inlier_cnt = inlier_cnt
            h = temp_h

    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
