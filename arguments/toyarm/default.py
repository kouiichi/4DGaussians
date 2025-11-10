ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [128, 128, 128, 50] 
    },
    multires = [1, 2, 4], 
    defor_depth = 1,
    net_width = 128, 
    plane_tv_weight = 0.0005, 
    time_smoothness_weight = 0.005, 
    l1_time_planes = 0.0002, 
    no_do = True,  
    no_dshs = True,  
    no_ds = False,  
    empty_voxel = False,
    render_process = False,
    static_mlp = False,
    control_input_dim = 6,
    control_hidden_dim = 128,  
    control_use_pe = True,
    control_num_frequencies = 4,
    control_activation = 'relu'
)

OptimizationParams = dict(
    dataloader = True,
    iterations = 12000, 
    zerostamp_init = True,
    batch_size = 4, 
    coarse_iterations = 4000,
    densify_until_iter = 8000,  
    densification_interval = 100,
    opacity_reset_interval = 2000, 
    # opacity_threshold_coarse = 0.01, 
    # opacity_threshold_fine_init = 0.01,  
    # opacity_threshold_fine_after = 0.01,
    # percent_dense = 0.005,
    lambda_dssim = 0.2,  
    lambda_lpips = 0.0, 
)