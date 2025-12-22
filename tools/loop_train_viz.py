# file: main.py

# from phase1_loop_train import run_training_pipeline
# from phase2_loop_viz import generate_visualization_scripts
import sys, os
sys.path.insert(0, os.getcwd())  # Ensure the current directory is in the path

from arguments import get_args
import subprocess
from datetime import datetime
import re
import json
import stat

from utils import write_json, build_common_args, get_passed_keys_from_argv

def extract_checkpoints(run_id):
    """
    Extracts checkpoint paths and their corresponding L2 lambda values from log files.
    Args:
        run_id (str): The unique identifier for the training run, used to filter log files
                        (e.g., "20250623_031915").
    Returns:
        dict: A dictionary mapping L2 lambda values to their corresponding checkpoint paths.
               The keys are integers representing the L2 lambda values, and the values are the
               paths to the checkpoint files.
    If no valid checkpoints are found, an empty dictionary is returned.
    If no log files are found for the given run_id, an empty list is returned.  
    """
    LOG_DIR = "log_file"
    VIZ_JSON_DIR = "viz/viz_combine_dicts"
    VIZ_SCRIPT_DIR = "scripts"

    checkpoint_line_pattern = re.compile(r"Saved checkpoint to (checkpoints/.*\.pth)")
    lambda_from_path_pattern1 = re.compile(r"_lamb_(\d+\.?\d*)_")
    lambda_from_path_pattern2 = re.compile(r"_L2_(\d+\.?\d*)_")
    checkpoints = {}

    try:
        log_files_for_run = [f for f in os.listdir(LOG_DIR) if f.endswith(f"_{run_id}.txt")]
    except FileNotFoundError:
        print(f"Error: Log directory '{LOG_DIR}' not found.")
        return []
    
    for log_file in log_files_for_run:
        full_log_path = os.path.join(LOG_DIR, log_file)
        with open(full_log_path, 'r') as f:
            content = f.read()

            # Use findall to get all matches, take the last one
            all_matches = checkpoint_line_pattern.findall(content)
            if all_matches:
                path = all_matches[-1]  # last checkpoint match
                lambda_match = lambda_from_path_pattern1.search(path)
                if lambda_match:
                    l2_lambda = int(float(lambda_match.group(1)))
                    checkpoints[l2_lambda] = path
                    print(f"---> Found Lambda {l2_lambda} from checkpoint in '{log_file}'")
                else:
                    lambda_match = lambda_from_path_pattern2.search(path)
                    if lambda_match:
                        l2_lambda = int(float(lambda_match.group(1)))
                        checkpoints[l2_lambda] = path
                        print(f"---> Found Lambda {l2_lambda} from checkpoint in '{log_file}'")
                    else:
                        print(f"Warning: No valid L2 lambda found in path: {path}")

    if not checkpoints:
        print("\nError: No valid checkpoints could be extracted. Cannot prepare tasks.")
        return []

    #print('checkpoints:', checkpoints)
    return checkpoints

def run_training_pipeline(lambda_values, args=None, passed_args=None):
    """
    Runs the training script for a list of L2 lambda values.

    Args:
        lambda_values (list): A list of integers or floats for L2 lambda.
        device (str): The CUDA device to run training on (e.g., "cuda:6").

    Returns:
        str: The unique run_id for this batch of training jobs, used to
             identify the corresponding log files.
    """
    # # --- Create directories if they don't exist ---
    # os.makedirs("log_file", exist_ok=True)
    # os.makedirs("checkpoints", exist_ok=True)

    if args.run_id == "None":
        # --- Get a unique identifier for this entire run ---
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_id = args.run_id

    print("=" * 70)
    print(f"PHASE 1: STARTING TRAINING BATCH | Run ID: {run_id}")
    print("=" * 70)

    if args.keep_training:
        print("Continuing training with existing run_id:", run_id)
        checkpoints = extract_checkpoints(run_id)

    #passed_args = get_passed_keys_from_argv()

    training_log_files = []  # This will hold paths to all log files created during training
    # --- Main Training Loop ---
    for l2_lambda in lambda_values:
        print(f"\n---> Preparing to train with L2 Lambda = {l2_lambda}")

        # Define a unique log file for this specific run
        log_file_path = os.path.join("log_file", f"train_lambda_{l2_lambda}_{run_id}.txt")

        training_log_files.append(log_file_path)  # Store the log file path for later use
        
        # command_list = [
        #     "python",
        #     "tools/contrastive_train.py",
        #     "--data_dir", str(args.data_dir),
        #     "--mats_load_name", str(args.mats_load_name),
        #     "--split", "train", # Ensure this is a string
        #     "--end_idx", str(args.end_idx),
        #     "--feature_load_name", "2d_patch",
        #     "--feature_save_name", "2d_patch",
        #     "--property_name", "density",
        #     # "--source_point_stride", "10",
        #     "--epochs", str(args.epochs),
        #     "--model", "mlp_attention",
        #     # "--advance_box_mask",
        #     # "--postprocess_masks",
        #     "--combine_losses", str(args.combine_losses),  # Ensure this is a string
        #     "--mlp_batch_size", str(args.mlp_batch_size),
        #     # Add dynamic arguments, ensuring values are strings
        #     "--one_mask", 
        #     "--l2_lambda", str(l2_lambda),
        #     "--log_file", log_file_path,
        #     "--alternate_l2_gt", str(args.alternate_l2_gt),
        #     "--device", args.device
        # ]

        command_list = [
            "python",
            "tools/contrastive_train.py",
        ]
        
        custom_args = {
            "split": "train",
            "one_mask": True,
            "log_file": log_file_path,
            "model": "mlp_attention",
            "l2_lambda": str(l2_lambda)
        }

        args_list = build_common_args(
            args=args,
            custom_args=custom_args,
            passed_keys=passed_args,
            return_list=True
        )
        command_list += args_list

        if args.verbose:
            print('command_list:', command_list)

        if args.advance_box_mask:
            command_list.append("--advance_box_mask")
        if args.postprocess_masks:
            command_list.append("--postprocess_masks")

        if args.keep_training:
            # If we are continuing training, we need to load the last checkpoint
            if l2_lambda in checkpoints:
                command_list += ["--mlp_checkpoint", checkpoints[l2_lambda]]
                print(f"Continuing training from checkpoint: {checkpoints[l2_lambda]}")
            else:
                print(f"Warning: No checkpoint found for L2 Lambda = {l2_lambda}. Starting fresh.")
        
        # if args.mlp_checkpoint != 'None':
        #     command_list += ["--mlp_checkpoint", args.mlp_checkpoint] 
            
        # We print a user-friendly version of the command
        print(f"Executing command. Log will be at: {log_file_path}")
        # print(" ".join(command_list)) # Uncomment for debugging to see the full command

        try:
            # --- THE FIX: Execute the command list directly. shell=False is the default and safer. ---
            subprocess.run(command_list, check=True)
            print(f"---✓ Successfully finished training for L2 Lambda = {l2_lambda}")
        except subprocess.CalledProcessError as e:
            print(f"!!! ERROR during training for L2 Lambda = {l2_lambda} !!!")
            print(f"Command failed with exit code {e.returncode}. Check log for details.")
            print("Continuing with the next lambda value...")
        except FileNotFoundError:
            print(f"!!! ERROR: 'python' or 'tools/contrastive_train.py' not found.")
            print("Please ensure you are running this script from the correct directory.")
            break # Stop the loop if the script can't be found

    print("\n" + "=" * 70)
    print("PHASE 1: ALL TRAINING JOBS COMPLETE.")
    print("=" * 70)
    
    return run_id, training_log_files

def generate_visualization_tasks(run_id, args, passed_args): # Added args to the signature
    """
    Generates a list of visualization tasks. Each task is a tuple containing:
    1. The command list to generate a final viz script.
    2. The path to the final viz script that will be generated.
    3. The path to the log file for that final visualization run.
    """
    print("\n" + "=" * 70)
    print(f"PHASE 2: PREPARING VISUALIZATION TASKS | For Run ID: {run_id}")
    print("=" * 70)

    # --- Configuration and log parsing (no changes here) ---
    LOG_DIR = "log_file"
    VIZ_JSON_DIR = "viz/viz_combine_dicts"
    VIZ_SCRIPT_DIR = "scripts"
    
    checkpoints = extract_checkpoints(run_id)  

    sorted_lambdas = sorted(checkpoints.keys())
    groups = [sorted_lambdas[i:i + 5] for i in range(0, len(sorted_lambdas), 5)]
    
    tasks = [] # This will hold our final (generator_command, output_path, log_path) tuples

    all_jsons = []


    #passed_args = get_passed_keys_from_argv()

    for i, group in enumerate(groups):
        #group_num = i + 1
        group_name = i + 1
        viz_dict = {"1": {"name": "Our"}}
        use_our22_name = ""
        use_blip2_name = ""

        use_our22 = False
        use_blip2 = False


        for j, l2_lambda in enumerate(group):
            key = str(7 + j)
            current_checkpoint = checkpoints[l2_lambda]
            if 'our22' in current_checkpoint:
                use_our22_name = '_our22'
                use_our22 = True
            
            if 'blip2' in current_checkpoint:
                use_blip2_name = '_blip2'
                use_blip2 = True
            
            current_checkpoint = checkpoints[l2_lambda]
            epoch = current_checkpoint.strip('.pth').split('_')[-1]
            
            if 'L2' in current_checkpoint:
                combine_loss_name = 'L2'
            elif 'CrossEntropy' in current_checkpoint:
                combine_loss_name = 'CE'
            else:
                combine_loss_name = 'unknown'
            viz_dict[key] = { "name": f"Our+SAM_MLP+attention_onemask_contrastive+{combine_loss_name}+{l2_lambda}{use_our22_name}{use_blip2_name}_epoch{epoch}", "model": "mlp_attention", "mlp_checkpoint": checkpoints[l2_lambda] }

        if use_our22:
            viz_dict["100"] = { "name": "our22"}

        if use_blip2:
            viz_dict["101"] = { "name": "blip2" }

        json_filename = f"viz_combine_dict_lambdas_{group_name}_{run_id}.json"
        json_filepath = os.path.join(VIZ_JSON_DIR, json_filename)
        with open(json_filepath, 'w') as f:
            json.dump(viz_dict, f, indent=4)
        print(f"\n---✓ Generated JSON config: {json_filepath}")

        output_sh_path = os.path.join(VIZ_SCRIPT_DIR, f"run_viz_group_{group_name}_{run_id}.sh")
        viz_save_name = f"contrastive_l2_loss_lambdas_{group_name}_{run_id}_pptx"
        viz_log_file = os.path.join(LOG_DIR, f"log_final_viz_{group_name}_{run_id}.txt") # This is the path we need
        
        soft_seg_log_file = os.path.join(LOG_DIR, f"log_final_viz_soft_seg_{group_name}_{run_id}.txt")
        soft_sh_path = os.path.join(VIZ_SCRIPT_DIR, f"run_viz_group_soft_seg_{group_name}_{run_id}.sh")

        json_to_use = json_filepath  # Default to the main JSON file
        output_sh_to_use = output_sh_path  # Default to the main output script path
        viz_log_file_to_use = viz_log_file  # Default to the main log file path

        all_jsons.append(json_filepath)

        soft_seg_dict = {}
        for key, value in viz_dict.items():
            if 'model' in value:
                soft_seg_dict[key] = value.copy()
                
        if args.evaluate_segmentation == "soft" or args.evaluate_segmentation == "intensive":
            json_file_path_eval = os.path.join(VIZ_JSON_DIR, f"viz_combine_dict_lambdas_{group_name}_{run_id}_evaluate_soft.json")
            write_json(soft_seg_dict, json_file_path_eval)

            json_to_use = json_file_path_eval    
            output_sh_to_use = soft_sh_path  # Use the soft segmentation script path
            viz_log_file_to_use = soft_seg_log_file  # Use the soft segmentation log file path

        if args.evaluate_segmentation == "None":
            generator_command = [
                "python", "pipeline_generation_script.py",
                "--data_dir", str(args.data_dir),
                "--mats_load_name", str(args.mats_load_name),
                "--split", "test", 
                "--end_idx", str(args.end_idx), 
                "--show", "1",
                "--source_point_strides", "10", 
                "--no_orig", 
                # "--viz_only", 
                "--all_combi",
                "--combine_dict", json_to_use, 
                "--plot_type", "both",
                "--plot_source_points", 
                "--advance_box_mask", 
                "--postprocess_masks",
                "--display_idx", 
                "--overwrite", 
                "--viz_save_name", viz_save_name,
                "--log_file", viz_log_file, # The log file is passed here
                "--output_sh", output_sh_to_use,
                "--device", str(args.device)  # Ensure we pass the device from args
            ]
            
        # elif args.evaluate_segmentation == "soft" or args.evaluate_segmentation == "intensive":
        else:
            if args.verbose:
                print('heheheheheh')
            generator_command = ["python", "pipeline_generation_script.py"]
            custom_args = {
                # 'source_point_strides': str(10),
                # 'dense_point_strides': str(5),
                "all_combi": True,
                "combine_dict": json_to_use,
                # "advance_box_mask": True,
                # "postprocess_masks": True,
                "predict": True,
                # "evaluate": True,
                "overwrite": True,
                "output_sh": output_sh_to_use,
                "log_file": soft_seg_log_file,
                "testing": True,
                "split": "test",
                #"evaluate_segmentation": True,
            }
            #exclude_keys = ["mat"
            args_list = build_common_args(
                args=args,
                custom_args=custom_args,
                passed_keys=passed_args,
                return_list=True
            )
            generator_command += args_list
            if args.verbose:
                print('generator_command:', generator_command)
        tasks.append((generator_command, output_sh_to_use, viz_log_file_to_use))
        
        print(f"---✓ Prepared task to generate final script: {output_sh_to_use}")
        
    print("\n" + "=" * 70)
    print("PHASE 2: ALL VISUALIZATION TASKS PREPARED.")
    print("=" * 70)
    
    return tasks, all_jsons

# --- Main Configuration ---
# All your settings are in one place.
#LAMBDA_VALUES_TO_TEST = [40, 60, 80, 100, 150, 250, 350, 450, 550]
# GPU_DEVICE = "cuda:6"


if __name__ == "__main__":
    args = get_args()  # Assuming get_args() is defined to parse command-line arguments

    LAMBDA_VALUES_TO_TEST = args.l2_lambdas if args.l2_lambdas else [40, 60, 80, 100, 150, 250, 350, 450, 550]

    passed_args = get_passed_keys_from_argv()
    # -------------------------------------------------------------------
    #                           PHASE 1: TRAINING
    # -------------------------------------------------------------------
    # This will run all 9 training jobs sequentially. This may take a long time.
    if args.run_id == "None" or args.keep_training:
        training_run_id, phase1_log_files = run_training_pipeline(
            lambda_values=LAMBDA_VALUES_TO_TEST,
            args=args,
            passed_args=passed_args
        )
    else:
        training_run_id = args.run_id
    
    # -------------------------------------------------------------------
    #                       PHASE 2: VISUALIZATION
    # -------------------------------------------------------------------
    # Ask for confirmation before proceeding to the next step.
    print(f"\nTraining for Run ID '{training_run_id}' is complete.")
    
    viz_tasks, _ = generate_visualization_tasks(run_id=training_run_id, args=args, passed_args=passed_args)  # Get the tasks and all JSONs

    if args.verbose:
        print('viz_tasks:', len(viz_tasks))
    if viz_tasks:
        print("\nWorkflow complete! Executing all visualization tasks...")
        
        # --- THE CHANGE: Collect all the log paths before looping ---
        all_viz_log_paths = [task[2] for task in viz_tasks]

        for i, (generator_command, final_script_path, _) in enumerate(viz_tasks, 1): # We can ignore the log path here
            print(f"\n{'='*25} TASK {i}/{len(viz_tasks)} {'='*25}")
            
            #print('genertor command:', generator_command)
            # STEP 2.1: Run the generator script
            print(f"STEP 1: Generating final script at '{final_script_path}'...")
            try:
                subprocess.run(generator_command, check=True)
                print(f"✓ Generator script completed successfully.")
            except Exception as e:
                print(f"× ERROR during script generation: {e}")
                print("Skipping execution for this task.")
                continue

            # STEP 2.2: Execute the script that was just created
            if not os.path.exists(final_script_path):
                print(f"× ERROR: Final script '{final_script_path}' was not created. Cannot execute.")
                continue

            print(f"STEP 2: Executing final script '{final_script_path}'...")
            try:
                subprocess.run(['bash', final_script_path], check=True)
                print(f"✓ Task {i} fully completed.")
            except Exception as e:
                print(f"× ERROR during final script execution: {e}")
        
        # --- THE FINAL STEP: Write the summary log file ---
        if args.loop_logs_timestamp:
            print("\n" + "=" * 70)
            print(f"Saving summary of visualization logs to: {args.loop_logs_timestamp}")
            try:
                with open(args.loop_logs_timestamp, 'w') as f:
                    if args.run_id == "None":
                        f.write("# Phase 1: Training Logs\n")
                        for path in phase1_log_files:
                            f.write(f"{path}\n")
                    
                    f.write("\n# Phase 2: Visualization Logs\n")
                    for path in all_viz_log_paths:
                        f.write(f"{path}\n")
                print("✓ Summary file saved successfully.")
            except Exception as e:
                print(f"× ERROR saving summary file: {e}")
        
    else:
        print("\nWorkflow finished, but no visualization tasks were prepared.")
            
    # else:
    #     print("\nVisualization script generation skipped.")
    #     print(f"When you are ready, you can run phase 2 manually by executing:")
    #     print(f"python phase2_visualization.py")