from cv2 import cv2
from diffCalculation import diff_functions
import time


def main():
    video_source = cv2.VideoCapture('test_video.avi')
    frame1 = None
    frame2 = None
    optical_flow_calculator = diff_functions.DiffOpticalFlow()
    while video_source.grab():
        frame2 = frame1
        flag, frame1 = video_source.retrieve()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if frame1 is not None:
            cv2.imshow('video1', frame1)
        if frame2 is not None:
            cv2.imshow('video2', frame2)
        cv2.waitKey(1)

        if frame1 is not None and frame2 is not None:
            start_time = time.time()
            result = optical_flow_calculator.get_optical_flow(frame1, frame2, block_size=[16, 16])
            print(time.time() - start_time)


if __name__ == '__main__':
    main()
