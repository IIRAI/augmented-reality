'''
Main script to run a augmented reality video stream with the PC camera
'''

import cv2
import src.parse_lib as p
from src.augmented_reality_3D import AR_3D


def main():
    args = p.parse_input()     # manage input parameters
    cap = cv2.VideoCapture(0)  # begin camera streaming
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_time = 1 / fps
    # init object for augmented reality computation
    ar_3d = AR_3D('ticket', args.model, sample_time,
                  args.rectangle, args.matches)
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
