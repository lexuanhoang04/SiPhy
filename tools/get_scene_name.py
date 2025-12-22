import sys, os
sys.path.insert(0, os.getcwd())

from arguments import get_args
from utils import get_scenes_list
if __name__ == '__main__':
    args = get_args()
    scenes = get_scenes_list(args)

    for scene in scenes:
        print('scene:', scene)
    