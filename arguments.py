# import sys, os 
# sys.path.insert(0, os.getcwd())
import argparse
import json

def get_args():
    parser = argparse.ArgumentParser(description='NeRF2Physics')
    # pipeline generation code
    parser.add_argument('--output_sh', type=str, default='generated_pipeline.sh',
                        help='output shell script name (default: generated_pipeline.sh)')

    # General arguments
    parser.add_argument('--data_dir', type=str, default="./data/abo_500/",
                        help='path to data (default: ./data/abo_500/)')
    parser.add_argument('--data_dir_processed', type=str, default="./data/abo_500_processed/",
                        help='path to processed data (default: ./data/abo_500_processed/)')
    parser.add_argument('--ho3d_dir', type=str, default="./data/HO3D/",
                        help='dir of HO3D')
                        
    parser.add_argument('--split', type=str, default="all",
                        help='dataset split, either train, val, train+val, test, or all (default: all)')
    parser.add_argument('--splits', type=str, nargs='+', default=None,
                        help='list of dataset splits to process (default: ["all"])')
                        
    parser.add_argument('--start_idx', type=int, default=0,
                        help='starting scene index, useful for evaluating only a few scenes (default: 0)')
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='ending scene index, useful for evaluating only a few scenes (default: -1)')
    parser.add_argument('--different_Ks', type=int, default=0,
                        help='whether data has cameras with different intrinsic matrices (default: 0)')
    parser.add_argument('--device', type=str, default="cuda",
                        help='device for torch (default: cuda)')
    parser.add_argument('--envi_name', type=str, default="nerf2phys6",
                        help='name of the conda environment to use (default: nerf2phys5)')
                        
    # NeRF training
    parser.add_argument('--training_iters', type=int, default=20000,
                        help='number of training iterations (default: 20000)')
    parser.add_argument('--near_plane', type=float, default=0.4,
                        help='near plane for ray sampling (default: 0.4)')
    parser.add_argument('--far_plane', type=float, default=6.0,
                        help='far plane for ray sampling (default: 6.0)')
    parser.add_argument('--vis_mode', type=str, default='wandb',
                        help='nerfstudio visualization mode (default: wandb)')
    parser.add_argument('--project_name', type=str, default='NeRF2Physics',
                        help='project name used by wandb (default: NeRF2Physics)')
    
    # NeRF point cloud
    parser.add_argument('--num_points', type=int, default=100000,
                        help='number of points for point cloud (default: 100000)')
    parser.add_argument('--bbox_size', type=float, default=1.0,
                        help='bounding box (cube) size, relative to scaled scene (default: 1.0)')

    # CLIP feature fusion
    parser.add_argument('--patch_size', type=int, default=56,
                        help='patch size (default: 56)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument('--feature_voxel_size', type=float, default=0.01,
                        help='voxel downsampling size for features, relative to scaled scene (default: 0.01)')
    parser.add_argument('--feature_save_name', type=str, default="ps56",
                        help='feature save name (default: ps56)')
    parser.add_argument('--occ_thr', type=float, default=0.01,
                        help='occlusion threshold, relative to scaled scene (default: 0.01)')
    parser.add_argument('--use_2d_input', action='store_true', default=False)

    parser.add_argument('--dense_point_stride', type=int, default=5,
                        help='stride for dense point sampling (default: 5)')

    parser.add_argument('--source_point_stride', type=int, default=0,
                        help='stride for source point sampling (default: 0)')

    parser.add_argument('--CLIP_based_model', type=str, default="CLIP")
    
    parser.add_argument('--reject_background_thr', type=int, default=0,
                        help='threshold for rejecting background points (default: 0)')

    parser.add_argument('--preprocess_list', type=str, nargs='+', default=[],
                        help='list of preprocessing steps to apply to the images (default: [])')

    parser.add_argument('--patch_size_list', type=int, nargs='+', default=[56],)

    parser.add_argument('--fuse_overwrite', action='store_true', default=False,
                        help='whether to overwrite existing features (default: False)')
    parser.add_argument('--min_pts', type=int, default=2000,
                        help='minimum number of points to sample (default: 2000)')
    parser.add_argument('--max_pts', type=int, default=10000,
                        help='maximum number of points to sample (default: 10000)')
    
    # Captioning and view selection
    parser.add_argument('--blip2_model_dir', type=str, default="./blip2-flan-t5-xl",
                        help='path to BLIP2 model directory (default: ./blip2-flan-t5-xl)')
    parser.add_argument('--mask_area_percentile', type=float, default=0.75,
                        help='mask area percentile for canonical view (default: 0.75)')
    parser.add_argument('--caption_save_name', type=str, default="info_new",
                        help='caption save name (default: info_new)')
    parser.add_argument('--custom_data', action='store_true', default=False,
                        help='whether to use custom data for captioning (default: False)')

    # Material proposal
    parser.add_argument('--caption_load_name', type=str, default="info_new",
                        help='name of saved caption to load (default: info_new)')
    parser.add_argument('--property_name', type=str, default="density",
                        help='property to predict (default: density)')
    parser.add_argument('--include_thickness', type=int, default=1,
                        help='whether to also predict thickness (default: 1)')
    parser.add_argument('--gpt_model_name', type=str, default="gpt-3.5-turbo",
                        help='GPT model name (default: gpt-3.5-turbo)')
    parser.add_argument('--mats_save_name', type=str, default="info_new",
                        help='candidate materials save name (default: info_new)')
    parser.add_argument('--materials_existed_name', type=str, default="None",
                        help='path to existing materials JSON file (default: None)')
    parser.add_argument('--gt_json', type=str, default="./data/abo_500/filtered_product_weights.json",
                        help='path to ground truth JSON file (default: ./data/abo_500/')
    parser.add_argument('--delay_time', type=int, default=0,
                        help='delay time in seconds to avoid rate limit (default: 0)')
    parser.add_argument('--classifying', action='store_true', default=False,
                        help='whether to classify materials instead of predicting properties (default: False)')
                        
    # Physical property prediction (uses property_name argument from above)
    parser.add_argument('--mats_load_name', type=str, default="info",
                        help='candidate materials load name (default: info)')
    parser.add_argument('--mats_load_names', type=str, nargs='+', default= None,
                        help='list of candidate materials load names (default: ["info", "info_new"])')
    parser.add_argument('--feature_load_name', type=str, default="ps56",
                        help='feature load name (default: ps56)')
    parser.add_argument('--prediction_mode', type=str, default="integral",
                        help="can be either 'integral' or 'grid' (default: integral)")
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='softmax temperature for kernel regression (default: 0.01)')
    parser.add_argument('--sample_voxel_size', type=float, default=0.01,
                        help='voxel downsampling size for sampled points, relative to scaled scene (default: 0.005)')
    
    parser.add_argument('--surface_cell_size', type=float, default=0,
                        help='cell size in m (default: 0)')

    parser.add_argument('--volume_method', type=str, default="thickness",
                        help="method for volume estimation, either 'thickness' or 'carving' (default: thickness)")
    parser.add_argument('--volume_path', type=str, default="None",
                        help='path to precomputed volume JSON file (default: None)')
    parser.add_argument('--nerf_vs_physx_scale', type=str, help='path to json containing scale factors between nerf and physx volumes (default: None)', default="None"
    )
    
    parser.add_argument('--physx_scales', type=float, nargs='+', default=[1.0],
                        help='list of scale factors for physics simulation (default: [1.0])')
    parser.add_argument('--physx_scale', type=float, default=1.0,
                        help='scale factor for physics simulation (default: 1.0)')

    parser.add_argument('--correction_factor', type=int, default=60,
                        help='correction factor for integral prediction (default: 60)')
    parser.add_argument('--show_mat_seg', type=int, default=0,
                        help="whether to show visualization of material segmentation (default: 0)")
    parser.add_argument('--save_preds', type=int, default=1,
                        help='whether to save predictions (default: 1)')
    parser.add_argument('--preds_save_name', type=str, default="mass",
                        help='predictions save name (default: mass)')
    parser.add_argument('--k', type=int, default=1,
                        help='number of nearest neighbors for prediction (default: 1)')
    
    parser.add_argument('--ks', type=int, nargs='+', default=[1],
                        help='list of k values for nearest neighbors (default: [1])')
    
    parser.add_argument('--source_point_strides', type=int, nargs='+', default=[0],
                        help='list of source point strides (default: None)')
    
    parser.add_argument('--dense_point_strides', type=int, nargs='+', default=[5],
                        help='list of dense point strides (default: [5])')
    
    parser.add_argument('--fuse_and_pred', action='store_true', default=False,
                        help='whether to run feature fusion and prediction in one step (default: False)')

    parser.add_argument('--predict', action='store_true', default=False,
                        help='whether to run prediction (default: False)')
    
    parser.add_argument('--use_sam', action='store_true', default=False,
                        help='whether to use SAM2 for material segmentation (default: False)')

    parser.add_argument('--mask_prior_lambda', type=float, default=0,
                        help='lambda for mask prior loss (default: 0)')
    
    parser.add_argument('--grid_sam', action='store_true', default=False,
                        help='whether to use grid sampling for SAM2 masks (default: False)')
    parser.add_argument('--grid_cell_size', type=float, default=0.0019,
                        help='cell size for grid sampling, relative to scaled scene (default: 0.02)')

    parser.add_argument('--save_pred', action='store_true', default=False,
                        help='whether to save predictions for seg vs mass experiment(default: False)')
    
    parser.add_argument('--mlp_checkpoint', type=str, default="None",
                    help='path to trained MLP weights, or "None" to disable (default: None)')

    parser.add_argument('--model', type=str, default='None',
                    help='Model type: mlp or mlp_attention (default: mlp)')

    parser.add_argument('--testing', action='store_true', default=False,
                        help='whether to run testing mode (default: False)')

    parser.add_argument('--randomize_prob', action='store_true', default=False,
                        help='whether to randomize query points for prediction (default: False)')

    parser.add_argument('--use_3d_volume', action='store_true', default=False,
                        help='whether to use 3D volume for prediction (default: False)')

    parser.add_argument('--use_min', action='store_true', default=False,
                        help='whether to use minimum value for prediction (default: False)')


    parser.add_argument('--log_volume', action='store_true', default=False,
                        help='whether to log volume during prediction (default: False)')
    parser.add_argument('--log_good_volume', action='store_true', default=False,
                        help='whether to log good volume scenes (default: False)')
    parser.add_argument('--predict_method', type=str, default="ours",
                        help='method for prediction, can be "ours", "gp" (default: ours)')
    parser.add_argument('--log_source_pts', action='store_true', default=False,
                        help='whether to log source points during prediction (default: False)')
    parser.add_argument('--pred_dicts_path', type=str, default="preds/preds_mass_dict.json",
                        help='path to save predictions dictionary (default: preds/preds_mass_dict.json)')
    parser.add_argument('--plot_from_csv_path', type=str, default="viz/iou_vs_metrics_mass/nerf.png",
                        help='path to save IOU vs metrics plot (default: viz/iou_vs_metrics)')
    
    parser.add_argument('--sim_map_source', type=str, default="our",
                        help='source for similarity map, can be "our" or "our22" or "blip2" (default: our)')        
    #parser.add_argument('--use_sam_method', type=str, default="None",
    #                    help='how to use SAM masks, can be "None", "noMLP", or "MLP" (default: None)')
    
    parser.add_argument('--random_var', type=str, default="None",
                        help='random variable for prediction, can be "None", "density", "thickness", or "both" (default: None)')

    parser.add_argument('--depth_method', type=str, default="pts",
                        help='method for depth estimation, can be "None", "depth_map", or "depth_net" (default: None)')

    # Evaluation
    parser.add_argument('--preds_json_path', type=str, default="./preds/preds_mass.json",
                        help='path to predictions JSON file (default: ./preds/preds_mass.json)')
    parser.add_argument('--gts_json_path', type=str, default="./data/abo_500/filtered_product_weights.json",
                        help='path to ground truth JSON file (default: ./data/abo_500_50/filtered_product_weights.json)')
    parser.add_argument('--clamp_min', type=float, default=0.01,
                        help='minimum value to clamp predictions (default: 0.01)')
    parser.add_argument('--clamp_max', type=float, default=100.,
                        help='maximum value to clamp predictions (default: 100.)')
    
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='whether to overwrite existing predictions (default: False)')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='whether to run evaluation with prediction (default: False)')

    parser.add_argument('--result_path', type=str, default="result.csv",
                        help='path to save evaluation results (default: result.csv)')
    
    
    parser.add_argument('--segmentation_result_path', type=str, default="segmentation_result.csv",
                        help='path to save segmentation evaluation results (default: segmentation_result.csv)')

    parser.add_argument('--correction_factors', type=int, nargs='+', default=[60])
    
    parser.add_argument('--mask_prior_lambdas', type=float, nargs='+', default=[3])

    parser.add_argument('--eval_mode', type=str, default="mass",
                        help='evaluation mode, can be "mass" or "segmentation" (default: mass)')

    parser.add_argument('--material_to_supercat_path', type=str, default="supercategories.json",
                        help='path to material to supercategory mapping file (default: supercategories.json)')
    parser.add_argument('--supercat_to_idx_path', type=str, default="supercat_to_idx.json",
                        help='path to supercategory to index mapping file (default: supercat_to_idx.json)')
    parser.add_argument('--superidx_to_otherprop', type=str, default="superidx_to_density.json", 
                        help='path to supercategory index to other property mapping file (default: superidx_to_density.json)')

    parser.add_argument('--run_ids', type=str, nargs='+', default=[], help='list of run IDs to evaluate'
                        ' (default: [])')
    parser.add_argument('--loop_train_only', action='store_true', default=False,
                        help='whether to only run loop training without evaluation (default: False)')
    parser.add_argument('--pred_other_path', type=str, default="preds/preds_other.json",
                        help='path to save other prediction data (default: preds/preds_other.json)')
    parser.add_argument('--num_points_path', type=str, default="preds/num_points.npy",
                            help='path to save number of points numpy file (default: preds/num_points.npy)')

    # Visualization
    parser.add_argument('--scene_name', type=str,
                        help='scene name for visualization (must be provided)')
    parser.add_argument('--show', type=int, default=0,
                        help='whether to show interactive viewer (default: 1)')
    parser.add_argument('--compositing_alpha', type=float, default=0.2,
                        help='alpha for compositing with RGB image (default: 0.2)')
    parser.add_argument('--cmap_min', type=float, default=500,
                        help='minimum physical property value for colormap (default: 500)')
    parser.add_argument('--cmap_max', type=float, default=6500,
                        help='maximum physical property value for colormap (default: 3500)')
    parser.add_argument('--viz_save_name', type=str, default="tmp",
                        help='visualization save name (default: tmp)')

    parser.add_argument('--plot_source_points', action='store_true', default=False,
                        help='whether to plot source points in the visualization (default: False)')
    
    parser.add_argument('--no_orig', action='store_true', default=False,
                        help='whether to not show the original image in the visualization (default: False)')

    parser.add_argument('--viz_only', action='store_true', default=False,
                        help='whether to only run visualization (default: False)')

    parser.add_argument('--fuse_and_viz', action='store_true', default=False,
                        help='whether to run feature fusion and visualization (default: False)')
    
    parser.add_argument('--all_combi', action='store_true', default=False,
                        help='whether to generate all combinations of k and source point stride (default: False)')
    
    parser.add_argument('--viz_sam', action='store_true', default=False,
                        help='whether to visualize SAM2 masks (default: False)')
    
    parser.add_argument('--combine', action='store_true', default=False,
                        help='whether to combine all masks visualization into one')

    parser.add_argument('--material_heatmap_only', action='store_true', default=False,
                        help='name for combined masks visualization (default: combined)')

    parser.add_argument('--configs_list', type=str, default="configs.json",
                        help='path to configs list for combined visualization (default: configs.json)')

    parser.add_argument('--plot_type', type=str, default="combined",
                        help='type of plot for combined visualization (default: combined)')
    
    parser.add_argument('--plot_dims', action='store_true', default=False,
                        help='whether to plot voxel dim correspond to no of dense points')

    parser.add_argument('--top_material', action='store_true', default=False,
                        help='whether to predict physical property using top material (default: False)')
    
    parser.add_argument('--log_file', type=str, default="log_file/log.txt",
                        help='path to save log file (default: log.txt)')
    #parser.add_argument('--log_file_segmentation', type=str, default="log_file/log_segmentation.txt",
    #                    help='path to save segmentation log file (default: log_file/log_segmentation.txt)')
                

    parser.add_argument('--combine_dict', type=str, default="viz/viz_combine_dict.json",
                        help='path to visualization combine dictionary (default: viz_combine_dict.json)') 

    parser.add_argument('--name', type=str, default="None",
                        help='name for the run, used for saving results (default: None)')

    parser.add_argument('--pptx_path', type=str, default="viz/presentations",
                        help='path to save PowerPoint presentation (default: viz/presentations)')

    parser.add_argument('--loop_logs_timestamp', type=str, default="log_file/loop_logs_timestamp.txt",
                        help='timestamp for loop logs, used to identify specific runs (default: "")')
    parser.add_argument('--render_to_seg_testing', action='store_true', default=False,
                        help='whether to render to segmentation testing (default: False)')
    
    parser.add_argument('--viz_mode', type=str, default='single',
                        help='visualization mode, can be "single" or "many" (default: single)')
                        
    parser.add_argument('--viz_volume', action='store_true', default=False,
                        help='whether to visualize volume (default: False)')
    parser.add_argument('--plot_cuboids_mode', type=str, default="None",
                        help='mode for plotting cuboids, can be "None", "normal", or "fast" (default: None)')
                        
    parser.add_argument('--slide_mode', type=str, default="pptx",
                            help='mode for slide generation, can be "pptx" or "pdf" (default: pptx)')      

    parser.add_argument('--vmin', type=float, default=0.0,
                        help='minimum value for visualization colormap (default: 0.0)')
    parser.add_argument('--vmax', type=float, default=7000.0,
                        help='maximum value for visualization colormap (default: 7000.0)') 
                                   
    # Training
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs for training (default: 1)')
    
    parser.add_argument('--one_mask', action='store_true', default=False,
                        help='whether to use only one mask for training (default: False)')
    
    parser.add_argument('--mlp_config', type=str, default="None",
                        help='path to MLP configuration file (default: mlp_config.json)')
    parser.add_argument('--dot_product', action='store_true', default=False,
                        help='whether to use dot product for MLP attention (default: False)')

    parser.add_argument('--save_good_scenes', action='store_true', default=False,
                        help='whether to save good scenes during training (default: False)')
    
    # parser.add_argument('--combine_losses', action='store_true', default=False,
    #                     help='whether to combine losses during training (default: False)')

    parser.add_argument('--combine_losses', type=str, default="None",
                        help='how to combine losses during training, can be "None", "weighted", or')

    parser.add_argument('--l2_lambda', type=float, default=0.01,
                        help='lambda for L2 regularization (default: 0.01)')
    
    parser.add_argument('--mlp_batch_size', type=int, default=1,
                        help='batch size for MLP training (default: 1)')

    parser.add_argument('--train_mode', type=str, default='contrastive',
                        help='mode to train, can be "contrastive" or "l2_alignment" (default: contrastive)')
    
    parser.add_argument('--use_pcgrad', action='store_true', default=False,
                        help='whether to use PCGrad for training (default: False)')
    
    parser.add_argument('--use_two_step', action='store_true', default=False,
                        help='whether to use two-step training with two losses (default: False)')
                        
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for training (default: 0.0001)')
    
    parser.add_argument('--lr2', type=float, default=0.0001,
                        help='learning rate for second loss (default: 0.0001)')
    
    parser.add_argument('--l2_lambdas', type=int, nargs='+', default=[10],)
    
    parser.add_argument('--run_id', type=str, default="None",
                        help='run ID for tracking experiments (default: None)')

    # parser.add_argument('--use_our22', action='store_true', default=False,
    #                     help='whether to use our22 method for training (default: False)')

    parser.add_argument('--alternate_l2_gt', type=str, default="None",
                        help='whether to alternate L2 loss with ground truth (default: our22)')
    
    parser.add_argument('--evaluate_segmentation', type=str, default="None",
                        help='whether to evaluate segmentation, can be "None", "soft", or "intensive" (default: None)')
    parser.add_argument('--segmap_pred_path', type=str, default="None",
                        help='path to segmentation map prediction file (default: None)')
    parser.add_argument('--segmap_pred_dir', type=str, default="None",
                        help='directory for segmentation map predictions (default: None)')
    parser.add_argument('--segmap_gt_dir', type=str, default="None",
                        help='directory for ground truth segmentation maps (default: None)')
                                 
    # SAM2 arguments
    parser.add_argument('--advance_mask', action='store_true', default=False,
                        help='whether to use advanced mask generation with SAM2 (default: False)')
    
    parser.add_argument('--advance_box_mask', action='store_true', default=False,
                        help='whether to use advanced box mask generation with SAM2 (default: False)')

    parser.add_argument('--advance_box_point_mask', action='store_true', default=False,
                        help='whether to use advanced box point mask generation with SAM2 (default: False)')

    parser.add_argument('--only_bad_scenes', action='store_true', default=False,
                        help='whether to only process bad scenes (default: False)')
    
    parser.add_argument('--postprocess_masks', action='store_true', default=False,
                        help='whether to postprocess masks by removing background (default: False)')
    
    parser.add_argument('--display_idx', action='store_true', default=False,
                        help='whether to display index numbers on masks (default: False)')
    
    parser.add_argument('--keep_training', action='store_true', default=False,
                        help='whether to keep training from the last checkpoint (default: False)')
                        
    # Debug arguments
    parser.add_argument('--debug_scene_name', type=str, default="None",
                        help='name of the scene to debug (default: None)')
    parser.add_argument('--debug_scene_idx', type=str, default="None",
                        help='index of the scene to debug (default: None)')
                        
    parser.add_argument('--debug_plot', action='store_true', default=False,
                        help='whether to run in debug mode (default: False)') 
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='whether to print verbose information (default: False)')
    
    parser.add_argument('--check_required', action='store_true', default=False,
                        help='whether to check required files before processing (default: False)')
                        
    # Misc
    parser.add_argument('--copy_output_dir', type=str, default='None',
                        help='directory to copy output files (default: None)')

    # point cloud 
    parser.add_argument('--other_pc_dir', type=str, default='None',
                        help='directory to load other point cloud files (default: None)')

    parser.add_argument('--save_pc', action='store_true', default=False,
                        help='whether to save point cloud files (default: False)')

    parser.add_argument('--translate_method', type=str, default='None',
                        help='method for translating point cloud, can be "None" or "manual" (default: None)')
                        
    parser.add_argument('--tau', type=float, default=0.05,
                        help='threshold for F-score calculation (default: 0.05)')
    parser.add_argument('--pose_align_method', type=str, default='None',
                        help='method for pose alignment, can be "None", "umeyama", or "icp" (default: None)')
    parser.add_argument('--scale_align_method', type=str, default='None',
                        help='method for scale alignment, can be "None" or "bounding_box" (default: None)')

    parser.add_argument('--icp_downsample', type=int, default=10,
                        help='downsample factor for ICP alignment (default: 10)')     
    parser.add_argument('--downsample_factor', type=int, default=1,
                        help='Factor by which to downsample the point cloud (default: 1)')

    parser.add_argument('--f_threshold', type=float, default=0,
                        help='Threshold for F-score calculation to be considered bad (default: 0.1)')
    
    parser.add_argument('--nerf_volume_json', type=str, default='None',
                        help='path to NeRF volume JSON file (default: None)')
    
    parser.add_argument('--physx_volume_json', type=str, default='None',
                        help='path to PhysX volume JSON file (default: None)')
    
    parser.add_argument('--dim_dict', type=str, default='None',
                        help='path to dimension dictionary JSON file (default: None)')
    
    parser.add_argument('--orig_image_dir', type=str, default='None',
                        help='directory to load original images (default: None)')
                        
    # copy args 
    parser.add_argument('--copy_mode', type=str, default='image',
                        help='mode for copying files, can be "image" or "info" (default: image)')
    
    parser.add_argument('--info_copy_source', type=str, default='info',
                        help='source info file name to copy (default: info)')
    
    parser.add_argument('--info_copy_target', type=str, default='info_new',
                        help='target info file name to copy to (default: info_new)')
    
    # density evaluation
    parser.add_argument('--physx_label_mode', type=str, default='density',
                        help='mode for PhysX label, can be "density" or "material" (default: density)')

    # config files
    parser.add_argument('--general_config', type=str, default=None)
    parser.add_argument('--material_config', type=str, default=None)
    parser.add_argument('--mask_config', type=str, default=None)
    parser.add_argument('--training_config', type=str, default=None)
    parser.add_argument('--viz_config', type=str, default=None)

    # view selection
    parser.add_argument('--view_selection_algo', type=str, default='iou',
                        help='algorithm for view selection, can be "None", "heuristic", or "learned" (default: None)')
    parser.add_argument('--physx_render_dir', type=str, default='./data/PhysXNet/',
                        help='directory for PhysXNet rendered images (default: ./data/PhysXNet)')
    
    # octopi
    parser.add_argument('--octopi_dataset_dir', type=str, default='./data/octopi/',)
    parser.add_argument('--octopi_output_json_path', type=str, default='./data/octopi/octopi_output.json',)
    parser.add_argument('--octopi_part_dir', type=str, default='./data/octopi/parts/',)
    parser.add_argument('--gp_dir', type=str, default='./data/Gaussian-Property/physx100_score_custom_dirs/',)
    parser.add_argument('--concatenated_viz_dir', type=str, default='./viz/octopi/concatenated/',)
    parser.add_argument('--octopi_info_dict_path', type=str, default='None')

    parser.add_argument('--octopi_current_epoch', type=int, default=0,
                        help='current epoch for octopi training (default: 0)')
    parser.add_argument('--octopi_current_step', type=int, default=0,
                        help='current step for octopi training (default: 0)')
                        
    parser.add_argument('--coco_json_path', type=str, default='None',
                        help='path to COCO format JSON file for captions (default: None)')
    
    args = parser.parse_args()


    # if not args.use_sam:
    #     args.mask_prior_lambda = 0

    # Load and apply config files to args object
    config_categories = {
        'general_config': 'General',
        'material_config': 'Material',
        'mask_config': 'Mask',
        'training_config': 'Training',
        'viz_config': 'Visualization',
        # ...
    }
    
    # Process each config file
    for config_arg, category in config_categories.items():
        config_path = getattr(args, config_arg)  # e.g., args.general_config
        if config_path and config_path != 'None':
            # Load the JSON file
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Apply each key-value pair to args
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)  # e.g., args.feature_load_name = "2d_patch"

    return args