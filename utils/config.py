import platform

def get_params(patch_type):
    params = dict()

    params['platform_node'] = platform.node()
    params['gpu_id'] = "0"
    if patch_type == 'gray':
        params['num_channels'] = 1
        params['patch_type'] = 'gray'
    elif patch_type == 'rgb':
        params['num_channels'] = 3
        params['patch_type'] = 'rgb'
    elif patch_type == 'dep':
        params['num_channels'] = 1
        params['patch_type'] = 'dep'
    elif patch_type == 'rgbd':
        params['num_channels'] = 4
        params['patch_type'] = 'rgbd'
    else:
        print("Invaid patch_type!")

    params['patch_size'] = 32
    params['batch_size'] = 128
    params['epochs'] = 8
    params['n_triplets'] = 5000000
    params['max_walls'] = 2000

    params['lr'] = 0.1
    params['lr_m'] = 0.01
    params['lr_decay'] = 1e-6
    params['wd'] = 1e-4
    params['margin'] = 2

    return params