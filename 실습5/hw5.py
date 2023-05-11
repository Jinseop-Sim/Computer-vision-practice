import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

img1 = plt.imread('./data/warrior_a.jpg')
img2 = plt.imread('./data/warrior_b.jpg')

cor1 = np.load("./data/warrior_a.npy")
cor2 = np.load("./data/warrior_b.npy")

def compute_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)
        
    F = None
    ### YOUR CODE BEGINS HERE
    A = []
    F_prime = [] # rank를 2로 줄이기 전의 F를 저장할 List이다.

    # build matrix for equations in Page 51
    # [uu', vu', u', uv', vv', v', u, v, 1] Matrix를 n개 생성한다.
    # 그럼 9 * n의 matrix가 생성된다.
    for i in range(len(x1[0])):
        A.append([x1[0][i] * x2[0][i], x1[1][i] * x2[0][i], x2[0][i], x1[0][i] * x2[1][i], x1[1][i] * x2[1][i], x2[1][i], x1[0][i], x1[1][i], 1])
    A = np.asarray(A)

    # compute the solution in Page 51
    # 이제 AF = 0 식을 이용해서, F를 계산해야 한다.
    # Homography를 계산했던 것과 같이, SVD를 계산해야 한다.
    # A^T A 의 eigenvector 중, eigenvalue 값이 가장 작은 vector가 F이므로,
    # V의 마지막 열을 불러와서 3 x 3 matrix로 변환하면 된다.
    U_prime, S_prime, V_prime = np.linalg.svd(A)
    F_prime = V_prime[-1].reshape((3, 3))

    # constrain F: make rank 2 by zeroing out last singular value (Page 52)
    # F의 SVD 결과에서, diagonal matrix의 [2, 2] 원소를 0으로 만들어 rank를 줄인다.
    U, S, V = np.linalg.svd(F_prime)
    S = np.diag(S) # diagonal matrix 형태로 만들어준다.
    S[2][2] = 0 # rank를 2로 만든다.

    F = U @ S @ V # 다시 F로 원상복귀!

    ### YOUR CODE ENDS HERE
    
    return F


def compute_norm_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = T1 @ x1
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = T2 @ x2

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = T2.T @ F @ T1
    
    return F


def compute_epipoles(F):
    e1 = None
    e2 = None

    # Fe1 = 0, Fe2 = 0 식 또한, Ah = 0과 같은 형태이다.
    # 따라서 Homography를 계산한 것과 같이 SVD를 시행한다.
    ### YOUR CODE BEGINS HERE
    U, S, V = np.linalg.svd(F)
    e1 = V[-1]
    e1 = e1/e1[2] # Homogeneous coordiate이기 때문에 정규화한다.

    U, S, V = np.linalg.svd(F.T)
    e2 = V[-1]
    e2 = e2/e2[2]
    ### YOUR CODE ENDS HERE
    return e1, e2


def draw_epipolar_lines(img1, img2, cor1, cor2):
    F = compute_norm_fundamental(cor1, cor2)

    e1, e2 = compute_epipoles(F)
    ### YOUR CODE BEGINS HERE
    # 점마다 색을 다르게 하기 위해 배열을 미리 선언
    colors = ['#ff8000', '#ff33ff', '#66b2ff', '#ff9999', '#009999', '#660066', '#CC0000', '#cce5ff', '#994c00', '#000000', '#ffffff', '#295510']
    # 직선을 그리기 위해 x좌표들을 미리 선언해놓는다.
    x1 = range(img1.shape[0])
    x2 = range(img2.shape[0])
    # Scatter plot을 이용해 점과 선을 그린다.
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 6))

    axes[0].imshow(img1)
    for i in range(len(cor1[0])):
        # 먼저 image1의 match point에 해당하는 점들을 찍는다.
        axes[0].scatter(cor1[0][i], cor1[1][i], c = colors[i], s=80, marker='.')
        # 그릴 직선의 기울기와 y절편을 구하기 위해 polyfit 함수를 사용한다.
        m, b = np.polyfit([e1[0], cor1[0][i]], [e1[1], cor1[1][i]], 1)
        # 위에서 구한 기울기와 절편을 이용해 모든 x좌표에 대해 직선을 그린다.
        axes[0].plot(x1, x1*m + b, c = colors[i], linewidth = 1)
    axes[0].title.set_text('Warrior A')

    axes[1].imshow(img2)
    for i in range(len(cor2[0])):
        # axes[0]과 동일한 순서로 진행한다.
        axes[1].scatter(cor2[0][i], cor2[1][i], c = colors[i], s=80, marker='.')
        m, b = np.polyfit([e2[0], cor2[0][i]], [e2[1], cor2[1][i]], 1)
        axes[1].plot(x2, x2*m + b, c = colors[i], linewidth = 1)
    axes[1].title.set_text('Warrior B')

    fig.show()
    # 아래와 같이 입력을 받지 않으면, show() 함수가 곧바로 종료된다.
    input('Press any key to exit the program')
    ### YOUR CODE ENDS HERE
    return

draw_epipolar_lines(img1, img2, cor1, cor2)
