import sys, os
sys.path.insert(0, os.getcwd())

from utils import *
import argparse
from arguments import *


args = get_args()


print('scene idx', get_scene_idx(args.scene_name, args))