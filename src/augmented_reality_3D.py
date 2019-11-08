
import os
import cv2
import math
import numpy as np

from configuration import parameters as parameter
from src.objloader_simple import OBJ
from src.filter_simple import FadingFilter


class AR_3D:
    '''
    This class manage the computation to add a 3D object in a camera
    streaming.  
    input:  
    -`reference2D`: string, name of the reference 
                    (only .jpg are supported at the moment).  
    -`model3D`: string, name of the 3D object to render (.obj file).  
    -`sample_time`: sample time of the video stream -> 1 / fps.  
    -`rectangle`: bool, display or not a bounding box where the reference
                  is estimated.  
    -`matches`: display the reference image on the side and show the matches.  
    '''

    def __init__(self, reference2D: str, model3D: str, sample_time: float,
                 rectangle=False, matches=False):
        # create ORB keypoint detector
        self.orb = cv2.ORB_create()
        # create BFMatcher object based on hamming distance
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # load the reference surface that will be searched in the video stream
        dir_name = os.getcwd()
        ref_path = 'reference/' + reference2D + '.jpg'
        self.model = cv2.imread(os.path.join(dir_name, ref_path), 0)
        # Compute model keypoints and its descriptors
        self.kp_model, self.des_model = self.orb.detectAndCompute(
            self.model, None)
        # Load 3D model from OBJ file
        obj_path = 'models/' + model3D + '.obj'
        self.obj = OBJ(os.path.join(dir_name, obj_path), swapyz=True)
        # frame rendering option
        self.rectangle = rectangle
        self.matches = matches
        # initialize filter class
        self.filter = FadingFilter(0.5, sample_time)

    def process_frame(self, frame):
        '''
        main function of the class, `process_frame` execute the entire pipeline
        to compute the frame rendering, that is:  
            1. frame feature extraction and reference detection.  
            2. homography estimation.  
            3. homogeneus 3D transformation estimation.  
            4. 3D object rendering in the frame,  
        input:
        - `frame`: frame to be analysed
        output:
        - `frame`: frame rendered (if it was succesful)
        '''
        # detect frame features
        kp_frame, matches = self.feature_detection(frame)
        # compute Homography if enough matches are found
        if len(matches) > parameter.MIN_MATCHES:
            homography = self.compute_homography(kp_frame, matches)
            # if a valid homography matrix was found render cube on model plane
            if homography is not None:
                # filter homography
                homography = self.filter.II_order_ff(homography)
                # obtain 3D projection matrix from homography matrix
                # and camera parameters
                projection = self.projection_matrix(
                    parameter.CAMERA_CALIBRATION, homography)
                if self.rectangle:  # draw rectangle over the reference
                    frame = self.draw_rectangle(self.model, frame, homography)
                if self.matches:    # draw first 10 matches.
                    frame = cv2.drawMatches(self.model, self.kp_model,
                                            frame, kp_frame,
                                            matches[:parameter.MIN_MATCHES],
                                            0, flags=2)
                # project cube or model
                frame = self.render(frame, self.obj, projection, self.model)
        else:
            print("Not enough matches found - %d/%d" %
                  (len(matches), parameter.MIN_MATCHES))
        # in any case return the frame
        return frame

    def feature_detection(self, frame):
        ''' detect frame keypoints and matches with reference '''
        # find and draw the keypoints of the frame
        kp_frame, des_frame = self.orb.detectAndCompute(frame, None)
        # match frame descriptors with model descriptors
        matches = self.bf.match(self.des_model, des_frame)
        # sort them in the order of their distance
        # the lower the distance, the better the match
        matches = sorted(matches, key=lambda x: x.distance)
        return kp_frame, matches

    def compute_homography(self, kp_frame, matches):
        ''' estimate the homography transformation'''
        # differenciate between source points and destination points
        src_pts = np.float32(
            [self.kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # compute Homography
        homography, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, 5.0)
        return homography

    def projection_matrix(self, camera_calibration, homography):
        '''
        From the camera calibration matrix and the estimated homography
        compute the 3D projection matrix.
        '''
        # Compute rotation along the x and y axis as well as the translation
        homography = homography * (-1)
        rot_and_transl = np.dot(np.linalg.inv(camera_calibration), homography)
        col_1 = rot_and_transl[:, 0]
        col_2 = rot_and_transl[:, 1]
        col_3 = rot_and_transl[:, 2]
        # normalise vectors
        l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
        rot_1 = col_1 / l
        rot_2 = col_2 / l
        translation = col_3 / l
        # compute the orthonormal basis
        c = rot_1 + rot_2
        p = np.cross(rot_1, rot_2)
        d = np.cross(c, p)
        rot_1 = np.dot(c / np.linalg.norm(c, 2) + d /
                       np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_2 = np.dot(c / np.linalg.norm(c, 2) - d /
                       np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_3 = np.cross(rot_1, rot_2)
        # compute the 3D projection matrix from the model to the current frame
        projection = np.stack((rot_1, rot_2, rot_3, translation)).T
        return np.dot(camera_calibration, projection)

    def draw_rectangle(self, model, frame, homography):
        '''
        Draw a rectangle that marks the found model in the frame.
        input:
        - `model`: image of the model.
        - `frame`: image if the frame.
        - `homography`: homography estimate.
        output:
        - `frame` in which is marked a rectangle where the model has been found.
        '''
        h, w = model.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, homography)
        # connect them with lines
        frame = cv2.polylines(
            frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        return frame

    def render(self, img, obj, projection, model, color=False):
        ''' Render a loaded .obj model into the current video frame. '''
        vertices = self.obj.vertices
        scale_matrix = np.eye(3) * 3
        h, w = self.model.shape

        for face in self.obj.faces:
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1]
                               for vertex in face_vertices])
            points = np.dot(points, scale_matrix)
            # render model in the middle of the reference surface. To do so,
            # model points must be displaced
            points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]]
                               for p in points])
            dst = cv2.perspectiveTransform(
                points.reshape(-1, 1, 3), projection)
            imgpts = np.int32(dst)
            if color is False:
                cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
            else:
                color = self.hex2rgb(face[-1])
                color = color[::-1]  # reverse
                cv2.fillConvexPoly(img, imgpts, color)
        return img

    def hex2rgb(self, hex_color):
        """ Helper function to convert hex strings to RGB. """
        hex_color = hex_color.lstrip('#')
        h_len = len(hex_color)
        return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))
