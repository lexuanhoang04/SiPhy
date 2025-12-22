import sys, os

sys.path.insert(0, os.getcwd())
from loop_train_viz import generate_visualization_tasks
from arguments import get_args

if __name__ == "__main__":
    args = get_args()
    generate_visualization_tasks(args.run_id, args)