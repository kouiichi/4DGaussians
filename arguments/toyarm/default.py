ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 25]
    },
    multires = [1, 2, 4],
    defor_depth = 1,
    net_width = 64,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    no_do=True,
    no_dshs=True,
    no_ds=False,
    empty_voxel=False,
    render_process=False,
    static_mlp=False,
    control_input_dim = 6,
    control_hidden_dim = 64,
    control_use_pe = False,
    control_num_frequencies = 4,
    control_activation = 'relu'
)

OptimizationParams = dict(
    dataloader=True,
    iterations = 30000,
    batch_size=2,
    coarse_iterations = 3000,
    densify_until_iter = 10_000,
    # opacity_reset_interval = 60000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    # pruning_interval = 2000
)