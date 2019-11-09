
import os
import argparse


def parse_input():
    '''
    Manage the input parameters from the user.  
    output: program arguments
    '''
    parser = argparse.ArgumentParser(
        description='Augmented reality application')

    parser.add_argument('-r', '--rectangle',
                        help='draw rectangle delimiting target surface on frame',
                        action='store_true')
    parser.add_argument('-ma', '--matches',
                        help='draw matches between keypoints',
                        action='store_true')
    models = get_3Dmodel_list()
    parser.add_argument("-m", "--model", default=models[0], choices=models,
                        help="choose the model to render in 3D")
    # # this is not used
    # parser.add_argument('-mk', '--model_keypoints',
    #                     help='draw model keypoints',
    #                     action='store_true')
    # # this is not used
    # parser.add_argument('-fk', '--frame_keypoints',
    #                     help='draw frame keypoints',
    #                     action='store_true')

    return parser.parse_args()


def get_3Dmodel_list():
    ''' return list of available 3D models in `models` directory '''
    # path to models
    base_path = os.path.join(os.path.dirname(__file__), '../models')
    # list of files in `models` directory
    files = os.listdir(base_path)
    # list of files name (without `.obj`)
    models = [os.path.splitext(file)[0] for file in files]
    return models
