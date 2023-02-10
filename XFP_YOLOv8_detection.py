### Code based on:https://github.com/niconielsen32/YOLOv8-Class/blob/main/YOLOv8InferenceClass.py https://www.youtube.com/watch?v=O9Jbdy5xOow

import torch 
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import math

"""
This function implements YOLOv8 model to detect XFPs on image
Shows a image of detected XFPs
:param: frame after imread
:param: model of Yolo
:return: Labels and coordinates
"""

class XFP:
    """
    Class that implements YOLOv8 model to detect XFP on image
    """

    def __init__(self, capture_index, model):
        """
        Initialize the class
        """
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.CLASS_NAMES_DICT = self.model.model.names              # Classes = different labels we want to identify/used on our model
        self.classes = self.model.names

    def undistor_image(self, frame, camera_file):
        """
        frame -> frame we want to apply undistort on
        camera_file -> path to the file with the calibration parameters
        returns: image undistor
        """

        with np.load (camera_file) as data :
            mtx = data['cameraMatrix']
            dist = data['distortion']
            
        h,w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # undistort
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst

    def predict(self, frame):
        """
        Takes a single frame as input, and scores the frame using YOLOv8 model.
        :param frame: Input frame in numpy/list/tuple format.
        :return: results variable that contains labels and coordinates predicted by model on the given frame.
        """
        results = self.model(frame)

        return results

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: Contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels plotted on it.
        """

        labels, coord = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        info = []

        for i in range(len(labels)):
            row = coord[i]
            if row[4] > 0.7: # Threshold -> If our detection is not more certain than a threshold (confidence score level)
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                info.append([self.class_to_label(labels[i]), x1, y1, x2, y2])
                
        return frame, info

    def find_Center_and_angle(self, info):
        center_x = []
        center_y = []
        angle_rad = []
        info_2 = []
        for inf in info:
            frame_aux = self.frame[inf[2]:inf[4], inf[1]:inf[3]] # img [y:y+h, x:x+w]
            gray = cv2.cvtColor(frame_aux,cv2.COLOR_BGR2GRAY)
            # Apply binary thresholding
            _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            # Detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
            contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) 
            # Find max area and create a flag if area of object detected is bigger than the expected -> possible not a XFP.
            max_area = 0
            for cont in contours:        
                area = cv2.contourArea(cont) 
                if area > max_area:
                    max_area = area

            if max_area > 30000:  # Value found manually
                print("Area of object detected bigger than 30000 (expected max value of XFP)")
                continue   # Pass to the next object
            info_2.append(inf)
            # Find biggest contour
            if contours:
                c = max(contours, key = cv2.contourArea)
                # Find centroid
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                center_x.append(cx + inf[1])
                center_y.append(cy + inf[2])

                cv2.drawContours(frame_aux, c, -1, (0,255,0), 3)
                
                # get rotated rectangle from outer contour
                rotrect = cv2.minAreaRect(c)
                # get angle from rotated rectangle
                            
                points = cv2.boxPoints(rotrect)
                points = points.tolist()
                points_tuples = list(map(tuple, points))

                norms2 = []
                for i in range(0, len(points_tuples)):
                    if i != 0:
                        norms2.append(math.sqrt(math.pow(points_tuples[i - 1][0] - points_tuples[i][0], 2) + math.pow(points_tuples[i - 1][1] - points_tuples[i][1], 2)))

                max_norm = max(norms2)
                ind_max_norm = norms2.index(max_norm)
                y_diff = abs(points_tuples[ind_max_norm+1][1] - points_tuples[ind_max_norm][1] )
                x_diff = abs(points_tuples[ind_max_norm+1][0] - points_tuples[ind_max_norm][0] )
                
                if (points_tuples[ind_max_norm][1] > points_tuples[ind_max_norm+1][1] and points_tuples[ind_max_norm][0] > points_tuples[ind_max_norm+1][0]):
                    angle = math.atan2(y_diff, x_diff)
                elif (points_tuples[ind_max_norm][1] < points_tuples[ind_max_norm+1][1] and points_tuples[ind_max_norm][0] < points_tuples[ind_max_norm+1][0]):
                    angle = math.atan2(y_diff, x_diff)
                else:
                    angle = math.atan2(-y_diff, x_diff)

                angle_rad.append(angle)

        return center_x, center_y, angle_rad, info_2

    def validate_label(self, info):
        info_2 = []
        for inf in info:
            frame_aux = self.frame[inf[2]:inf[4], inf[1]:inf[3]] # img [y:y+h, x:x+w]
            if inf[0] == "XFP_back":
                hsv = cv2.cvtColor(frame_aux, cv2.COLOR_BGR2HSV)
                # Define range of green color in HSV
                lower_green = np.array([30, 50, 30])
                upper_green = np.array([100, 255, 100])
                # preparing the mask to overlay
                mask = cv2.inRange(hsv, lower_green, upper_green)
                
                # Split image into top half and bottom half
                height, width = frame_aux.shape[:2]
                top_half = mask[0:height//2, 0:width]
                bottom_half = mask[height//2:height, 0:width]

                # Count non-zero pixels in top half and bottom half
                top_half_non_zero = cv2.countNonZero(top_half)
                bottom_half_non_zero = cv2.countNonZero(bottom_half)

                # Check in which half the green part is located
                if top_half_non_zero > bottom_half_non_zero:
                    info_2.append([inf[0], "top_half"])
                else:
                    info_2.append([inf[0], "bottom_half"])

            elif inf[0] == "XFP_front":
                gray = cv2.cvtColor(frame_aux,cv2.COLOR_BGR2GRAY)
                # Apply binary thresholding
                _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
                # Detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
                contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) 
                copy_frame = frame_aux.copy()
                # Find areas of contours
                areaArray = []
                for i, c in enumerate(contours):
                    area = cv2.contourArea(c)
                    areaArray.append(area)
                #first sort the array by area
                sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
                #find the nth largest contour [n-1][1], in this case 2
                secondlargestcontour = sorteddata[1][1]
                cv2.drawContours(copy_frame, secondlargestcontour, -1, (255,0,0), 3)
                hsv = cv2.cvtColor(copy_frame, cv2.COLOR_BGR2HSV)

                # Define range of blue color in HSV
                lower_blue = np.array([110,50,50])
                upper_blue = np.array([130,255,255])

                # Threshold the HSV image to get only blue colors
                mask = cv2.inRange(hsv, lower_blue, upper_blue)

                # Split image into top half and bottom half
                height, width = copy_frame.shape[:2]
                top_half = mask[0:height//2, 0:width]
                bottom_half = mask[height//2:height, 0:width]

                # Count non-zero pixels in top half and bottom half
                top_half_non_zero = cv2.countNonZero(top_half)
                bottom_half_non_zero = cv2.countNonZero(bottom_half)

                # Check in which half the blue part is located
                if top_half_non_zero > bottom_half_non_zero:
                    info_2.append([inf[0], "top_half"])
                else:
                    info_2.append([inf[0], "bottom_half"])

        return info_2
        
    def __call__(self):
        """
        This function is called when class is executed. It processes the image passed, finds the XFP thanks to Yolo model and returns labels and coord of the XFPs found
        :return: Labels and coordinates of objects detected by the model in the frame.
        """

        cam = cv2.VideoCapture(self.capture_index)
        assert cam.isOpened()   # Check if camera channel was opened
        cam.set(3, 1280)
        cam.set(4, 720)
        cam.set(cv2.CAP_PROP_FPS, 60)
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Turn the autofocus off

        start_time = time()

        ret, frame = cam.read() # Check if we got a frame
        assert ret

        results = self.predict(frame)
        print(results)
        frame, info = self.plot_boxes(results, frame)

        end_time = time()
        fps = 1/np.round(end_time - start_time, 2)

        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5)

        cv2.imshow('YOLOv8 Detection', frame)

        cam.release()
        cv2.destroyAllWindows()
    


# For testing
def load_model():
    """
    Loads YOLO model with the weight files
    """
    model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8m model (file of weights returned after training ends)
    model.fuse()

    return model
    
# model = load_model()

# detector = XFP(capture_index = 0, model, calib_file)    # Create a new object to identify XFPs
# detector()