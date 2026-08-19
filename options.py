import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# =========================================================================
standard_3D = ['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze']
standard_5D = ['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'synllie']
CDD11_all = ['rain', 'snow', 'haze', 'low', 'haze_rain', 'haze_snow', 'low_rain', 'low_snow', 'low_haze','low_haze_rain', 'low_haze_snow']
# =========================================================================

def base_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='QuReC', help='Model name.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default='', help='Batch size per GPU.')
    parser.add_argument('--patch_size', type=int, default=128, help='Input patch size.')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs for training.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--trainset', default="standard_3D", help=["standard_3D", "standard_5D", "CDD11_all"])
    parser.add_argument('--data_file_dir', type=str, default='', help='Path to datasets.')
    parser.add_argument('--output_path', type=str, default="", help='Output save path.')
    parser.add_argument('--ckpt_dir', type=str, default="", help='Checkpoint directory.')
    parser.add_argument('--resume_from', type=str, default='', help='Resume from checkpoint.')
    parser.add_argument('--de_type', nargs='+', default='', help='Degradation types for training/testing (Auto-set by trainset).')
    parser.add_argument('--benchmarks', nargs='+', default='', help='which benchmarks to test on (Auto-set by trainset).')
    parser.add_argument('--fine_tune_from', type=str, default=None, help='Fine-tune from checkpoint.')
    parser.add_argument('--save_results', action="store_true", default=True, help="Save restored outputs.")
    parser.add_argument('--wblogger', default='QuReC', action="store_true", help='Log to Weights & Biases.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers.')
    parser.add_argument('--accum_grad', type=int, default=1, help='Gradient accumulation steps.')

    return parser

def train_options():
    parser = base_parser()
    options = parser.parse_args()

    if options.accum_grad > 1:
        options.batch_size = options.batch_size // options.accum_grad

    config_map = {"standard_3D": standard_3D, "standard_5D": standard_5D, "CDD11_all": CDD11_all}
    selected_set = options.trainset
    if selected_set in config_map:
        target_tasks = config_map[selected_set]

        print(f"\n[Auto-Config] ------------------------------------------------")
        print(f"[Auto-Config] Trainset mode detected: '{selected_set}'")
        options.de_type = target_tasks
        options.benchmarks = target_tasks
        print(f"[Auto-Config] Automatically set de_type & benchmarks to ({len(target_tasks)} tasks):")
        print(f"              {target_tasks}")
        print(f"[Auto-Config] ------------------------------------------------\n")

    else:
        print(f"[Auto-Config] Warning: '{selected_set}' not found in auto-config map.")
        print(f"[Auto-Config] Using manually provided or default de_type/benchmarks.")

    return options