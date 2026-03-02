import wandb

#Configure hyperparameter search space
base_config = {
    'program': 'train_unified.py',
    'method': 'bayes',
    'metric':       {'name': 'test/accuracy', 'goal': 'maximize'},
    'parameters': {
        "conv1_filter": {"values": [32]},
        "conv2_filter": {"values": [16]},
        "layer_1":      {"values": [256]},
        "layer_2":      {"values": [512]},
        "layer_3":      {"values": [256]},
        "dropout":      {"values": [0.1]},
        "l1":           {"values": [0.0001]},
        "l2":           {"values": [0.0001]},
    },
}

datasets = ["ext_bary_sub", "alpha", "ext_alpha", "ext_adj"]
modes = ["dim0", "dim1", "dim0_dim1", "dim0_img", "dim1_img", "dim0_dim1_img"]

#Initialize hyperparameter sweeps for each experiment type
for ds in datasets:
    for mode in modes:
        sweep_config = base_config.copy()
        sweep_config['parameters']['dataset_class'] = {'value': ds}
        sweep_config['parameters']['experiment_mode'] = {'value': mode}

        #Initialize the sweep directly via the API
        sweep_id = wandb.sweep(sweep_config, project="Ost_new_lower")
        
        print(f"Initialized sweep for {mode} - {ds}: {sweep_id}")
        with open(f"{ds}_sweeps/sweep_id_{mode}_{ds}.txt", "w") as f:
            f.write(sweep_id)




