from pathlib import Path
from shutil import rmtree
from datetime import timedelta

from transformer_maskgit.optimizer import get_optimizer
from eval import evaluate_internal
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_body import CTReportDataset  # 移除了未使用的 CTReportDatasetinfer
import numpy as np
import pandas as pd

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

import math
import torch.optim.lr_scheduler as lr_scheduler
from ct_clip import CTCLIP

import os


# helpers
def apply_softmax(array):
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_warmup=10000, gamma=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_warmup = T_warmup
        self.gamma = gamma
        self.T_cur = 0
        self.lr_min = 0
        self.iteration = 0

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iteration < self.T_warmup:
            lr = self.eta_max * self.iteration / self.T_warmup
        else:
            self.T_cur = self.iteration - self.T_warmup
            T_i = self.T_0
            while self.T_cur >= T_i:
                self.T_cur -= T_i
                T_i *= self.T_mult
                self.lr_min = self.eta_max * (self.gamma ** self.T_cur)
            lr = self.lr_min + 0.5 * (self.eta_max - self.lr_min) * \
                 (1 + math.cos(math.pi * self.T_cur / T_i))

        self.iteration += 1
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._update_lr()
        self._update_T()

    def _update_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.get_lr()[0]

    def _update_T(self):
        if self.T_cur == self.T_0:
            self.T_cur = 0
            self.lr_min = 0
            self.iteration = 0
            self.T_0 *= self.T_mult
            self.eta_max *= self.gamma

class CTClipTrainer(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        num_train_steps,
        batch_size,
        data_train,          # 在 run_train 中，这里传入的是 filtered_train.csv 的路径
        data_valid,          # 在 run_train 中，这里传入的是 filtered_valid.csv 的路径
        reports_file_train,  # 传入 filtered_train_reports.csv
        reports_file_valid,  # 传入 filtered_valid_reports.csv
        train_meta_file,     # 传入 filtered_train_metadata.csv
        valid_meta_file,     # 传入 filtered_valid_metadata.csv
        tokenizer = None,
        lr = 1.25e-6,
        wd = 0.,
        max_grad_norm = 0.5,
        save_results_every = 1,
        save_model_every = 1,
        results_folder = './ctclip/',
        num_workers = 8,
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        
        if tokenizer != None:
            self.tokenizer=tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("/home/huali/model/Qwen3.5-9B", trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = '<|endoftext|>'

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        all_parameters = set(CTClip.parameters())
        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)
        self.max_grad_norm = max_grad_norm
        self.lr = lr

        # ===================================================================
        # 对接重构后的全部位 data.py 数据流
        # ===================================================================
        self.ds = CTReportDataset(
            filtered_csv_path = data_train,       # 指向高纯度训练 CSV
            reports_file = reports_file_train,
            meta_file = train_meta_file,
            is_train = True                       # 标记为训练集 (触发物理截断防污染)
        )

        self.valid_ds = CTReportDataset(
            filtered_csv_path = data_valid,       # 指向高纯度验证 CSV
            reports_file = reports_file_valid,
            meta_file = valid_meta_file,
            is_train = False                      # 标记为验证集 (触发数据复制防死锁)
        )
        # ===================================================================

        self.dl = DataLoader(
            self.ds,
            num_workers=num_workers,
            batch_size=self.batch_size,
            shuffle = True,
            drop_last = True,
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            num_workers=num_workers,
            batch_size=1,
            shuffle = False,
            drop_last = True, 
        )

        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.valid_dl_iter=cycle(self.valid_dl)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)

        (
            self.dl_iter,
            self.valid_dl_iter,
            self.CTClip,
            self.optim,
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.valid_dl_iter,
            self.CTClip,
            self.optim,
        )

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.saved_checkpoints = []

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return
        pkg = dict(
            model=self.accelerator.get_state_dict(self.CTClip),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        CTClip = self.accelerator.unwrap_model(self.CTClip)
        CTClip.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def train_step(self):
        device = self.device
        steps = int(self.steps.item())
        self.CTClip.train()

        logs = {}

        # ===================================================================
        # 🚨 1. 训练数据解包：接收来自新 data.py 的 图像、文本、部位标签
        # ===================================================================
        video, text, body_labels = next(self.dl_iter)

        video = video.to(device)
        body_labels = body_labels.to(device) # 🚨 将部位标签推入显卡
        
        text = list(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

        with self.accelerator.autocast():
            # 🚨 传入 body_labels，激活内部的辅助分类头
            loss = self.CTClip(
                text_tokens, 
                video, 
                return_loss=True, 
                device=device, 
                body_labels=body_labels
            )

        self.accelerator.backward(loss)
        accum_log(logs, {'loss': loss.item()})
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.CTClip.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()
        self.print(f"{steps}: loss: {logs['loss']}")

        # ===================================================================
        # 🚨 2. 验证集解包与全量验证修复
        # ===================================================================
        if not (steps % self.save_results_every):
            if self.is_main:
                self.print(f"🔄 Running Full Validation at step {steps}...")
            
            self.CTClip.eval()
            
            with torch.no_grad():
                total_val_loss = 0.0
                # 🚨 修复：将写死的 val_steps = 2 改为全量数据集验证
                val_steps = len(self.valid_dl)
                
                for _ in range(val_steps):
                    try:
                        # 🚨 三元解包：图像、文本、验证集部位标签
                        val_video, val_text, val_body_labels = next(self.valid_dl_iter)
                    except StopIteration:
                        self.valid_dl_iter = cycle(self.valid_dl)
                        val_video, val_text, val_body_labels = next(self.valid_dl_iter)

                    val_video = val_video.to(device)
                    val_body_labels = val_body_labels.to(device) # 🚨
                    
                    val_text = list(val_text)
                    val_text_tokens = self.tokenizer(
                        val_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
                    ).to(device)

                    with self.accelerator.autocast():
                        # 🚨 传入 val_body_labels，计算验证集的联合 Loss
                        val_loss = self.CTClip(
                            val_text_tokens, 
                            val_video, 
                            return_loss=True, 
                            device=device,
                            body_labels=val_body_labels
                        )
                    
                    total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / val_steps
                
                if self.is_main:
                    self.print(f"✅ Step {steps} Full Validation Loss (InfoNCE + Auxiliary): {avg_val_loss:.4f}\n")
                
            self.CTClip.train()

        # ===================================================================
        # 绝对安全的 FSDP 切片保存逻辑 (原版保留)
        # ===================================================================
        if steps > 0 and not (steps % self.save_model_every):
            self.accelerator.wait_for_everyone()
            
            save_dir = str(self.results_folder / f'CTClip_step_{steps}')
            self.accelerator.save_state(save_dir)
            
            if self.is_main:
                self.print(f'{steps}: successfully saved sharded model to {save_dir}')
                
                if not hasattr(self, 'saved_checkpoints'):
                    self.saved_checkpoints = []
                
                self.saved_checkpoints.append(save_dir)
                
                if len(self.saved_checkpoints) > 10:
                    oldest_ckpt = self.saved_checkpoints.pop(0)
                    if os.path.exists(oldest_ckpt):
                        import shutil
                        try:
                            shutil.rmtree(oldest_ckpt)
                        except Exception as e:
                            print(f"Failed to remove {oldest_ckpt}: {e}")
                            
            self.accelerator.wait_for_everyone()

        self.steps += 1
        return logs

    def train(self, log_fn=noop):
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')

        if self.is_main:
            final_model_path = str(self.results_folder / f'CTClip_final_step_{int(self.steps.item())}.pt')
            self.accelerator.save(self.accelerator.get_state_dict(self.CTClip), final_model_path)
            self.print(f"✅ Final model saved to {final_model_path}")