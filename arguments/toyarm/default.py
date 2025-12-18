ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [128, 128, 128, 100] 
    },
    multires = [1, 2, 4], 
    defor_depth = 1,
    net_width = 128, 
    plane_tv_weight = 0.001, 
    time_smoothness_weight = 0.01, 
    l1_time_planes = 0.0001, 
    no_do = True,  
    no_dshs = True,  
    no_ds = True,  
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
    iterations = 15000, 
    zerostamp_init = False,
    batch_size = 2, 
    coarse_iterations = 4000,
    densify_until_iter = 12000,  
    densification_interval = 200,
    opacity_reset_interval = 2000, 
    lambda_dssim = 0.2,  
    lambda_lpips = 0.0, 
    use_gmflow = False,
    flow_loss_weight = 0.1
)