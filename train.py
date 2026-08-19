import os
import pathlib
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils import AIOTrainDataset, CDD11
from utils.loss_utils import FFTLoss
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import clip
from net.model import Model
from options import train_options

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
                             "Lowlight, Haze, and Rnow",
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


class MS_SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super(MS_SSIMLoss, self).__init__()
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=data_range,
            kernel_size=5,
            betas=(0.5, 0.5, 0.5, 0.5, 0.5)
        )

    def forward(self, pred, target):
        return 1.0 - self.ms_ssim(pred, target)


class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.net = Model()
        self.rec_loss = nn.L1Loss()
        self.fft_loss = FFTLoss()
        self.lambda_balance = getattr(opt, "lambda_balance", 0.01)
        self.lambda_route = getattr(opt, "lambda_route", 0.05)
        self.alpha_soft_balance = getattr(opt, "alpha_soft_balance", 0.7)
        self.alpha_hard_balance = getattr(opt, "alpha_hard_balance", 0.3)
        self.use_st_hard_balance = getattr(opt, "use_st_hard_balance", True)
        self.route_eps = 1e-8

        self.degradation_vocab, self.task_to_proto_idx = \
            build_experiment_prototypes_from_trainset(opt.trainset)

        print(f"Using prototype setting for trainset: {opt.trainset}")
        print("Prototype vocabulary:")
        for idx, prompt in enumerate(self.degradation_vocab):
            print(f"[{idx:02d}] {prompt}")

        print("Task -> Prototype index mapping:")
        for k, v in self.task_to_proto_idx.items():
            print(f"  {k:>15s} -> {v}")
        print("Loading CLIP text encoder for degradation prototype bank...")
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

        print(f"Prototype bank built successfully. Shape: {self.degradation_text_bank.shape}")

        self.id_to_task = {}
        if hasattr(opt, "de_type") and isinstance(opt.de_type, list):
            for idx, task_name in enumerate(opt.de_type):
                self.id_to_task[idx] = str(task_name)

        print("ID -> Task mapping from opt.de_type:")
        for k, v in self.id_to_task.items():
            print(f"  {k} -> {v}")

    def forward(self, x, text_prototypes=None):
        return self.net(x, text_prototypes=text_prototypes)

    def _build_route_targets(self, de_info, device):
        targets = []

        if isinstance(de_info, torch.Tensor):
            de_ids = de_info.detach().cpu().tolist()
            for idx in de_ids:
                task_key = self.id_to_task.get(int(idx), None)
                if task_key is None:
                    targets.append(-1)
                else:
                    targets.append(self.task_to_proto_idx.get(str(task_key), -1))

        elif isinstance(de_info, (list, tuple)):
            for item in de_info:
                task_key = str(item)
                targets.append(self.task_to_proto_idx.get(task_key, -1))

        else:
            targets = [-1]

        return torch.tensor(targets, dtype=torch.long, device=device)

    def _collect_routing_distributions(self):
        routing_weights = self.net.get_routing_weights()
        if len(routing_weights) == 0:
            return None, None, None, None
        branch_means = [rw.mean(dim=1) for rw in routing_weights]
        mean_route_per_image = torch.stack(branch_means, dim=0).mean(dim=0)
        mean_route_global = mean_route_per_image.mean(dim=0)
        route_cat = torch.cat(routing_weights, dim=1)
        return mean_route_per_image, mean_route_global, route_cat, branch_means

    def balance(self, x):
        eps = 1e-10
        if x is None:
            return torch.tensor(0.0, device=self.device)
        x = x.float().reshape(-1)
        if x.numel() <= 1:
            return x.new_tensor(0.0)

        return x.var(unbiased=False) / (x.mean().pow(2) + eps)

    def _routing_supervision_loss(self, branch_means, targets):
        if branch_means is None or len(branch_means) == 0:
            return torch.tensor(0.0, device=self.device)

        valid_mask = targets >= 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        total_loss = 0.0
        num_valid_branches = 0

        for bm in branch_means:
            if bm is None:
                continue
            log_probs = torch.log(bm[valid_mask] + self.route_eps)
            total_loss = total_loss + F.nll_loss(log_probs, targets[valid_mask])
            num_valid_branches += 1

        if num_valid_branches == 0:
            return torch.tensor(0.0, device=self.device)

        return total_loss / num_valid_branches

    def _hard_routing_load(self, route_cat, straight_through=False):
        if route_cat is None:
            return None

        K = route_cat.shape[-1]
        hard_idx = route_cat.argmax(dim=-1)
        hard_onehot = F.one_hot(hard_idx, num_classes=K).float()
        hard_onehot = hard_onehot.to(route_cat.dtype)

        if straight_through:
            hard_assign = hard_onehot - route_cat.detach() + route_cat
        else:
            hard_assign = hard_onehot

        hard_load = hard_assign.mean(dim=(0, 1))
        return hard_load

    def _routing_balance_loss(self, mean_route_global, route_cat,
                              alpha_soft=0.7, alpha_hard=0.3,
                              use_st_hard=True):
        if mean_route_global is None:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero, zero, zero, zero, None

        soft_balance = self.balance(mean_route_global)

        hard_balance = torch.tensor(0.0, device=self.device)
        if route_cat is not None:
            hard_load_opt = self._hard_routing_load(route_cat, straight_through=use_st_hard)
            hard_balance = self.balance(hard_load_opt)

        balance_loss = alpha_soft * soft_balance + alpha_hard * hard_balance

        hard_load_log = self._hard_routing_load(route_cat, straight_through=False) \
            if route_cat is not None else None

        if hard_load_log is not None:
            route_global_max = hard_load_log.max()
            effective_num_prototypes = 1.0 / (torch.sum(hard_load_log.float() ** 2) + self.route_eps)
        else:
            route_global_max = mean_route_global.max()
            effective_num_prototypes = 1.0 / (torch.sum(mean_route_global.float() ** 2) + self.route_eps)

        return balance_loss, route_global_max, effective_num_prototypes, \
               soft_balance, hard_balance, hard_load_log

    def training_step(self, batch, batch_idx):
        ([clean_name, de_info], degrad_patch, clean_patch) = batch

        restored = self.net(
            degrad_patch,
            text_prototypes=self.degradation_text_bank
        )

        rec_loss = self.rec_loss(restored, clean_patch)
        fft_loss = 0.1 * self.fft_loss(restored, clean_patch)

        main_loss = rec_loss + fft_loss

        mean_route_per_image, mean_route_global, route_cat, branch_means = \
            self._collect_routing_distributions()

        route_targets = self._build_route_targets(de_info, degrad_patch.device)

        balance_loss, route_global_max, effective_num_prototypes, \
        soft_balance, hard_balance, hard_load = self._routing_balance_loss(
            mean_route_global=mean_route_global,
            route_cat=route_cat,
            alpha_soft=self.alpha_soft_balance,
            alpha_hard=self.alpha_hard_balance,
            use_st_hard=self.use_st_hard_balance,
        )

        route_loss = self._routing_supervision_loss(branch_means, route_targets)

        loss = main_loss + self.lambda_balance * balance_loss + self.lambda_route * route_loss

        # =========================
        # Logging
        # =========================
        self.log("Total_Loss", loss, sync_dist=True, prog_bar=True)
        self.log("Main_Loss", main_loss, sync_dist=True)
        self.log("Rec_Loss", rec_loss, sync_dist=True)
        self.log("FFT_Loss", fft_loss, sync_dist=True)

        self.log("Balance_Loss", balance_loss, sync_dist=True)
        self.log("Soft_Balance_Loss", soft_balance, sync_dist=True)
        self.log("Hard_Balance_Loss", hard_balance, sync_dist=True)
        self.log("Route_Loss", route_loss, sync_dist=True)

        if mean_route_global is not None:
            route_entropy = -torch.sum(
                mean_route_global * torch.log(mean_route_global + self.route_eps)
            )
            self.log("Route_Global_Entropy", route_entropy, sync_dist=True)
            self.log("Route_Global_MaxUsage", route_global_max, sync_dist=True)
            self.log("Route_Effective_NumProto", effective_num_prototypes, sync_dist=True)

        if hard_load is not None:
            self.log("Route_Hard_MaxUsage", hard_load.max(), sync_dist=True)

        if branch_means is not None:
            for i, bm in enumerate(branch_means):
                branch_global = bm.mean(dim=0)
                branch_entropy = -torch.sum(
                    branch_global * torch.log(branch_global + self.route_eps)
                )
                self.log(f"Route_Branch{i+1}_Entropy", branch_entropy, sync_dist=True)
                self.log(f"Route_Branch{i+1}_MaxUsage", branch_global.max(), sync_dist=True)

        if batch_idx == 0 and self.global_rank == 0:
            print("de_info:", de_info[:10] if isinstance(de_info, torch.Tensor) else de_info)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("LR_Schedule", lr, sync_dist=True)

        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.net.parameters(), lr=self.opt.lr)

        if self.opt.fine_tune_from:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=1,
                max_epochs=self.opt.epochs
            )
        else:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=15,
                max_epochs=150
            )

        return [optimizer], [scheduler]


def main(opt):
    print("Options")
    print(opt)

    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    log_dir = os.path.join("logs", time_stamp)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    if opt.wblogger:
        name = opt.model + "_" + time_stamp
        logger = WandbLogger(
            project="RSIR_Project",
            name=name,
            save_dir=log_dir,
            config=opt
        )
    else:
        logger = TensorBoardLogger(save_dir=log_dir)

    if opt.fine_tune_from:
        ckpt = os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt")
        print(f"Loading checkpoint from: {ckpt}")
        model = PLTrainModel.load_from_checkpoint(ckpt, opt=opt, strict=False)
    else:
        model = PLTrainModel(opt)

    checkpoint_path = opt.ckpt_dir
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="epoch={epoch}",
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_top_k=-1,
        save_last=True
    )

    if "CDD11" in opt.trainset:
        _, subset = opt.trainset.split("_")
        print(f"Initializing CDD11 Dataset with subset: {subset}")
        trainset = CDD11(opt, split="train", subset=subset)
    else:
        print("Initializing Standard AIOTrainDataset")
        trainset = AIOTrainDataset(opt)

    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers
    )

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=opt.accum_grad,
        deterministic=False
    )

    if opt.resume_from:
        resume_ckpt_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")
        print(f"Resuming from checkpoint: {resume_ckpt_path}")
    else:
        resume_ckpt_path = None

    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        ckpt_path=resume_ckpt_path
    )


if __name__ == "__main__":
    train_opt = train_options()
    main(train_opt)