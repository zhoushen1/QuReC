import os
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import clip
from net.model import Model
from options import train_options
from utils.test_utils import save_img
from data.dataset_utils import IRBenchmarks, CDD11

def build_experiment_prototypes_from_trainset(trainset_name: str):
    name = str(trainset_name).strip()

    # --------------------------------------------------
    # Setting 1: standard_3D
    # --------------------------------------------------
    if name == "standard_3D":
        degradation_vocab = ["Noise",
                             "Rain",
                             "Haze",
                             ]
        task_to_proto_idx = {"denoise_15": 0,
                             "denoise_25": 0,
                             "denoise_50": 0,
                             "derain": 1,
                             "dehaze": 2,
                             }

    # --------------------------------------------------
    # Setting 2: standard_5D
    # --------------------------------------------------
    elif name == "standard_5D":
        degradation_vocab = ["Noise",
                             "Rain",
                             "Haze",
                             "Blur",
                             "Lowlight",
                             ]

        task_to_proto_idx = {"denoise_15": 0,
                             "denoise_25": 0,
                             "denoise_50": 0,
                             "derain": 1,
                             "dehaze": 2,
                             "deblur": 3,
                             "synllie": 4,
                             }

    # --------------------------------------------------
    # Setting 3: CDD11_all
    # --------------------------------------------------
    elif name == "CDD11_all":
        degradation_vocab = ["Rain",
                             "Snow",
                             "Haze",
                             "Lowlight",
                             "Haze and Rain",
                             "Haze and Snow",
                             "Lowlight and Rain",
                             "Lowlight and Snow",
                             "Lowlight and Haze",
                             "Lowlight, Haze, and Rain",
                             "Lowlight, Haze, and Snow",
                             ]

        task_to_proto_idx = {"rain": 0,
                             "snow": 1,
                             "haze": 2,
                             "low": 3,
                             "lowlight": 3,
                             "haze_rain": 4,
                             "haze_snow": 5,
                             "low_rain": 6,
                             "low_snow": 7,
                             "low_haze": 8,
                             "low_haze_rain": 9,
                             "low_haze_snow": 10,
                             }

    else:
        raise ValueError(
            f"Unsupported opt.trainset: {name}. "
            f"Expected one of ['standard_3D', 'standard_5D', 'CDD11_all']"
        )

    return degradation_vocab, task_to_proto_idx


def calc_psnr(img1, img2, data_range=1.0):
    err = np.sum((img1 - img2) ** 2, dtype=np.float64)
    return 10 * np.log10((data_range ** 2) / (err / img1.size))

def calc_ssim(img1, img2):
    return structural_similarity(
        img1, img2,
        channel_axis=2,
        gaussian_weights=True,
        data_range=1.0,
        full=False
    )

class PLTestModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.net = Model()

        self.degradation_vocab, self.task_to_proto_idx = \
            build_experiment_prototypes_from_trainset(opt.trainset)

        print(f"Using prototype setting for test trainset: {opt.trainset}")
        print("Prototype vocabulary:")
        for idx, prompt in enumerate(self.degradation_vocab):
            print(f"[{idx:02d}] {prompt}")
        print("Loading CLIP text encoder for testing prototype bank...")
        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False

        with torch.no_grad():
            text_tokens = clip.tokenize(self.degradation_vocab)
            text_features = clip_model.encode_text(text_tokens).float()
            text_features = F.normalize(text_features, dim=-1)

        self.register_buffer("degradation_text_bank", text_features)
        del clip_model

        print(f"Testing prototype bank built. Shape: {self.degradation_text_bank.shape}")

    def forward(self, x, text_prototypes=None):
        return self.net(x, text_prototypes=text_prototypes)


def run_test(opts, net, dataset):
    testloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    if opts.save_results:
        pathlib.Path(
            os.path.join(os.getcwd(), f"{opts.output_path}/{opts.benchmarks[0]}")
        ).mkdir(parents=True, exist_ok=True)

    calc_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg",
        normalize=True,
        reduction="mean"
    ).cuda()

    psnr_list, ssim_list, lpips_list = [], [], []

    with torch.no_grad():
        for ([clean_name, de_info], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.cuda()
            clean_patch = clean_patch.cuda()

            restored = net(
                degrad_patch,
                text_prototypes=net.degradation_text_bank
            )

            if isinstance(restored, (tuple, list)):
                restored = restored[0]

            restored = torch.clamp(restored, 0, 1)

            # LPIPS
            lpips_val = calc_lpips(clean_patch, restored).detach().cpu().item()
            lpips_list.append(lpips_val)

            # PSNR / SSIM
            restored_np = restored.detach().cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            clean_np = clean_patch.detach().cpu().permute(0, 2, 3, 1).squeeze(0).numpy()

            psnr_val = peak_signal_noise_ratio(clean_np, restored_np, data_range=1.0)
            ssim_val = calc_ssim(clean_np, restored_np)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            if opts.save_results:
                save_name = os.path.splitext(os.path.split(clean_name[0])[-1])[0] + ".png"
                save_img(
                    os.path.join(
                        os.getcwd(),
                        f"{opts.output_path}/{opts.benchmarks[0]}",
                        save_name
                    ),
                    img_as_ubyte(restored_np)
                )

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_lpips = float(np.mean(lpips_list))

    print("PSNR: {:f} SSIM: {:f} LPIPS: {:f}\n".format(mean_psnr, mean_ssim, mean_lpips))
    return mean_psnr, mean_ssim, mean_lpips


def main(opt):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    start_epoch = 0
    end_epoch = 149

    pathlib.Path(opt.output_path).mkdir(parents=True, exist_ok=True)
    log_file_path = os.path.join(opt.output_path, "all_epochs_detailed_metrics.txt")

    original_benchmarks = list(opt.benchmarks)

    header = ["Epoch", "Avg_PSNR", "Avg_SSIM", "Avg_LPIPS"]
    for task_name in original_benchmarks:
        header.append(f"{task_name}_PSNR")
        header.append(f"{task_name}_SSIM")
        header.append(f"{task_name}_LPIPS")

    with open(log_file_path, "w") as f:
        f.write("\t".join(header) + "\n")

    print(f"Start testing from epoch {start_epoch} to {end_epoch}...")
    print(f"Results will be saved to: {log_file_path}")

    for epoch_idx in range(start_epoch, end_epoch + 1):
        ckpt_name = f"epoch={epoch_idx}.ckpt"
        ckpt_path = os.path.join(opt.ckpt_dir, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"\n[Warning] Checkpoint not found: {ckpt_path}. Skipping...")
            continue

        print(f"\n{'=' * 20} Testing Epoch: {epoch_idx} {'=' * 20}")
        print(f"Loading model from: {ckpt_path}")

        try:
            net = PLTestModel.load_from_checkpoint(
                ckpt_path,
                opt=opt,
                strict=False
            ).cuda()
            net.eval()
        except Exception as e:
            print(f"[Error] Failed to load checkpoint {ckpt_name}: {e}")
            continue

        total_psnr = []
        total_ssim = []
        total_lpips = []

        for i, de in enumerate(original_benchmarks):
            ind_opt = opt
            ind_opt.benchmarks = [de]

            # dataset selection
            if "CDD11" in opt.trainset:
                dataset = CDD11(opt, split="test", subset=de)
            else:
                dataset = IRBenchmarks(ind_opt)

            print(f"--> [{i + 1}/{len(original_benchmarks)}] Testing on {de}...")
            cur_psnr, cur_ssim, cur_lpips = run_test(ind_opt, net, dataset)

            total_psnr.append(cur_psnr)
            total_ssim.append(cur_ssim)
            total_lpips.append(cur_lpips)

        avg_psnr = float(np.mean(total_psnr))
        avg_ssim = float(np.mean(total_ssim))
        avg_lpips = float(np.mean(total_lpips))

        print(f"\n[Result Epoch {epoch_idx}] Avg PSNR: {avg_psnr:.4f}")

        row_data = [
            f"{epoch_idx}",
            f"{avg_psnr:.6f}",
            f"{avg_ssim:.6f}",
            f"{avg_lpips:.6f}"
        ]

        for i in range(len(original_benchmarks)):
            row_data.append(f"{total_psnr[i]:.6f}")
            row_data.append(f"{total_ssim[i]:.6f}")
            row_data.append(f"{total_lpips[i]:.6f}")

        with open(log_file_path, "a") as f:
            f.write("\t".join(row_data) + "\n")

        del net
        torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("All tests finished.")
    print(f"Detailed results saved in: {log_file_path}")
    print("=" * 50)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    test_opt = train_options()
    main(test_opt)
