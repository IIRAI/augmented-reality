
# Useful links
# http://www.pygame.org/wiki/OBJFileLoader
# https://rdmilligan.wordpress.com/2015/10/15/augmented-reality-using-opencv-opengl-and-blender/
# https://clara.io/library

# TODO -> Implement command line arguments
#         (scale, model and object to be projected)
#      -> Refactor and organize code (proper function definition and
#         separation, classes, error handling...)

import argparse
import cv2
from src.augmented_reality_3D import AR_3D


def parse_input():
    '''
    Manage the input parameters from the user.  
    output: program arguments
    '''
    parser = argparse.ArgumentParser(description='Augmented reality application')

    parser.add_argument('-r', '--rectangle',
                        help='draw rectangle delimiting target surface on frame',
                        action='store_true')
    parser.add_argument('-ma', '--matches',
                        help='draw matches between keypoints',
                        action='store_true')
    # this is not used
    parser.add_argument('-mk', '--model_keypoints',
                        help='draw model keypoints',
                        action='store_true')
    # this is not used
    parser.add_argument('-fk', '--frame_keypoints',
                        help='draw frame keypoints',
                        action='store_true')
    # TODO jgallostraa -> add support for model specification
    # parser.add_argument('-mo','--model',
    #                     help = 'Specify model to be projected',
    #                     action = 'store_true')
    return parser.parse_args()


def main():
    args = parse_input()                       # manage input parameters
    # init object for augmented reality computation
    ar_3d = AR_3D('ticket', 'fox',
                  args.rectangle, args.matches)
    cap = cv2.VideoCapture(0)                  # begin camera streaming

    while True:
        ret, frame = cap.read()                # get frame from camera
        if not ret:
            print("Unable to capture video")
            return
        # process the frame to add a virtual 3D object in the frame
        frame = ar_3d.process_frame(frame)
        cv2.imshow('frame', frame)             # show result
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press `q` to exit the program
            break
    cap.release()                              # close the program
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    main()
