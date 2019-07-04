# pip install opencv-contrib-python==3.4.0.12
import cv2
import numpy as np
import matplotlib.pyplot as plt


# example from https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html

def imshow(img):
    print("* Show image...")
    cv2.imshow("image", img)
    print("* Waiting for any keyboard command...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ratiotest(matches):
    good = []
    print("* Ratio test")
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append(m)
    return good


def execute_findHomography(good, kp1, kp2):  # , img1, img2):
    print("*Find homography")
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # # print(img1.shape)
        # h, w, rgb = img1.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, M)
        #
        # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
    return matchesMask


MIN_MATCH_COUNT = 10

print("load image 1...")
img1 = cv2.imread("vosamples/vo01.jpg")

print("load image 2...")
img2 = cv2.imread("vosamples/vo02.jpg")

print("load image 3...")
img3 = cv2.imread("vosamples/vo03.jpg")

print("initiate sift detector...")
sift = cv2.xfeatures2d.SIFT_create()

print("Process sift example...")
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
kp3, des3 = sift.detectAndCompute(img3, None)

# print(len(des1),len(des3))
# DES13 = np.float32(np.vstack((np.array(des1),np.array(des3))))
# print(len(DES13))

print("Initiate brute force matching descriptor")
bf = cv2.BFMatcher()
print(">Second-first simple matching image processing...")
matchesA = bf.match(des2, des1)
print(">Second-third simple matching image processing...")
matchesB = bf.match(des2, des3)


#


# print("Apply ratio test")
# print(">Test A")
# goodA = np.array(ratiotest(matchesA))
# print(">Test B")
# goodB = np.array(ratiotest(matchesB))

def my_test(matchA, matchB):
    mydist = 0.2
    best = []
    for m, n in zip(matchA, matchB):
        if abs(m.distance - n.distance) / 100 < mydist:
            print(abs(m.distance - n.distance) / 100)
            best.append([m, n])
    return best


best = np.array(my_test(matchesA, matchesB))

print(">      len(des1)", len(des1))
print(">      len(des2)", len(des2))
print(">      len(des3)", len(des3))
print()
print(">      len(best)", len(best))

print(">Homography A")
maskA = np.array(execute_findHomography(best[:, 0], kp2, kp1)).astype(bool)
print(">Homography B")
maskB = np.array(execute_findHomography(best[:, 1], kp2, kp3)).astype(bool)

print(len(maskA))
print(len(maskB))
best_mask = np.array([m and n for m, n in zip(maskA, maskB)])
# print(best_mask)

best = best[best_mask]

print("After exhomo")
print(">      len(best)", len(best))

print("Image show")

img3 = cv2.drawMatches(img2, kp2, img1, kp1, best[:, 0], None)

imshow(img3)

img4 = cv2.drawMatches(img2, kp2, img3, kp3, best[:, 1], None)

imshow(img4)

# print("Compare test")

# for a, b in zip(goodA, goodB):
#     # if a.queryIdx == b.queryIdx:
#     #     print("looks good:    ")
#     print(">     A:", a, "  and B:", b)
#     print(">     a.trainIdx:", a.trainIdx)
#     print(">     b.trainIdx:", b.trainIdx)
#     print(">     a.queryIdx:", a.queryIdx)
#     print(">     b.queryIdx:", b.queryIdx)

# print("MatchesA")
# print(matchesA[2][0].queryIdx)
# print("MatchesB")
# print(matchesB)
# print("MatchesC")
# print(matchesC)
#
# for i in range(0, min(len(matchesA), len(matchesB))):
#     A = matchesA[i]
#     B = matchesB[i]
#     C = matchesC[i]
#     if A[1].trainIdx == B[0].trainIdx or A[1].queryIdx == B[0].trainIdx or A[0].trainIdx == B[1].queryIdx:
#         print("looks good:    ", i)
#         print(">     A:", A, "  and B:", B)
#         print(">     A[1].trainIdx:", A[1].trainIdx)
#         print(">     B[0].trainIdx:", B[0].trainIdx)
#         print(">     A[1].queryIdx:", A[1].queryIdx)
#         print(">     B[1].queryIdx:", B[1].queryIdx)

# print("Apply ratio test")
# print(">Test A")
# goodA = ratiotest(matchesA)
# print(">Test B")
# goodB = ratiotest(matchesB)
#
#
# print("Create homography mask")
# print("*Find homography")
#
# if len(good) > MIN_MATCH_COUNT:
#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()
#
#     # print(img1.shape)
#     h, w, rgb = img1.shape
#     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#     dst = cv2.perspectiveTransform(pts, M)
#
#     img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
#
# else:
#     print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
#     matchesMask = None
#
#
#
#
# # print(">Homography A")
# # maskA = execute_findHomography(goodA, kp1, kp2, img1, img2)
# # print(">Homography B")
# # maskB = execute_findHomography(goodB, kp2, kp3, img2, img3)
# # print(maskA)
#
# print("Image show")
# draw_paramsA = dict(matchColor=(0, 255, 0),  # draw matches in green color
#                     singlePointColor=None,
#                     matchesMask=maskA,  # draw only inliers
#                     flags=2)
#
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, goodA, None, **draw_paramsA)
#
# imshow(img3)
#
# draw_paramsB = dict(matchColor=(0, 255, 0),  # draw matches in green color
#                     singlePointColor=None,
#                     matchesMask=maskB,  # draw only inliers
#                     flags=2)
#
# img4 = cv2.drawMatches(img2, kp2, img3, kp3, goodB, None, **draw_paramsB)
#
# imshow(img4)
