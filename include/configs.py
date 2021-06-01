configs_initial = {
    "name": "initial",
    "down_sample_factor": 16,
    "norm_factor": 255,
    "n_clusters": 10,
    "use_spatial": False,
    "normalize": True,
    "apply_thresh": False,
    "amplify_edges": False
}
configs_advanced = {
    "name": "",
    "n_clusters": 10,
    "down_sample_factor": 20,
    "norm_factor": 255,

}

def get_configs(configs_name):
    if 'initial' in configs_name:
        configs = configs_initial
    else:
        configs = configs_advanced
    return configs