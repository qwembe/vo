import cv2
import numpy as np
from estimate_movement import process_vo

lk_params = dict(winSize=(19, 19),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=1000,
                      qualityLevel=0.01,
                      minDistance=8,
                      blockSize=19)


def checkedTrace(img0, img1, p0, back_threshold=1.0):
    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return p1, status


green = (0, 255, 0)
red = (0, 0, 255)

cam = cv2.VideoCapture("vosamples/videocutted.mp4")
sift = cv2.xfeatures2d.SIFT_create()

ret, frame = cam.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

kp0 = sift.detect(frame)
p0 = np.float32([[[p.pt[0], p.pt[1]]] for p in kp0])

ret, frame2 = cam.read()
ret, frame2 = cam.read()


frame_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
p1, trace_status = checkedTrace(frame_gray, frame_gray2, p0)

p0 = p0[trace_status].copy()
p1 = p1[trace_status].copy()

H, status1 = cv2.findHomography(p0, p1, (0, cv2.RANSAC)[True], 3.0)

for (x0, y0), (x1, y1), good in zip(p0[:, 0], p1[:, 0], status1[:, 0]):
    if good:
        cv2.line(frame2, (x0, y0), (x1, y1), (0, 128, 0))
    cv2.circle(frame2, (x1, y1), 2, (red, green)[good], -1)

cv2.imshow('lk_homography', frame2)
cv2.waitKey(0)

ret, frame2 = cam.read()
ret, frame2 = cam.read()


frame_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
p2, trace_status = checkedTrace(frame_gray, frame_gray2, p1)

p0 = p0[trace_status].copy()
p1 = p1[trace_status].copy()
p2 = p2[trace_status].copy()
status1 = status1[trace_status].copy()

H, status2 = cv2.findHomography(p1, p2, (0, cv2.RANSAC)[True], 3.0)

best = (status1[:, 0] >= 1) * (status2[:, 0] >= 1)

a = p0[best].copy()
b = p1[best].copy()
c = p2[best].copy()
choose_mask = np.random.choice(len(a), 9)

# print(a[choose_mask, 0])

for (x0, y0), (x1, y1), good in zip(a[:, 0], c[:, 0], best):
    if good:
        cv2.line(frame2, (x0, y0), (x1, y1), (0, 128, 0))
    cv2.circle(frame2, (x1, y1), 2, (red, green)[good], -1)



cv2.imshow('lk_homography', frame2)
cv2.waitKey(0)
cv2.destroyAllWindows()

counter = 1000
flag = False
while counter > 0 and not flag:
    try:
        choose_mask = np.random.choice(len(a), 9)
        process_vo(a[choose_mask, 0], b[choose_mask, 0], c[choose_mask, 0])
        # flag = True
    except IOError as e:
        counter = counter - 1



cam.release()


# def main():
#     video_src = "vosamples/videocutted.mp4"
#     App(video_src).run()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()
