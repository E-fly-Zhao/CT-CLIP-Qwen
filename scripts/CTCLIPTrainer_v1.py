from pathlib import Path
from shutil import rmtree
from datetime import timedelta

from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer

from eval import evaluate_internal
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import CTReportDataset
from data_inference_nii import CTReportDatasetinfer

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
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
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
        data_train = "train",
        data_valid = "valid",
        reports_file_train = "data_reports.xslx",
        reports_file_valid = "data_reports.xslx",
        train_meta_file = "meta_data.csv",
        valid_meta_file = "meta_data.csv",
        labels = "labels.csv",
        tokenizer = None,
        lr = 1.25e-6,
        wd = 0.,
        max_grad_norm = 0.5,
        save_results_every = 1,
        save_model_every = 1 ,
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
            # self.tokenizer=BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
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
        self.lr=lr

        # self.ds = CTReportDataset(data_folder=data_train, reports_file=reports_file_train, meta_file=train_meta_file)

        # self.valid_ds = CTReportDatasetinfer(data_folder=data_valid, reports_file=reports_file_valid, meta_file=valid_meta_file, labels = labels)
        # ✅ 新代码：训练集和验证集统一使用最干净的 CTReportDataset，且不再传 meta_file 和 labels
        #self.ds = CTReportDataset(data_folder=data_train, reports_file=reports_file_train)
        #self.valid_ds = CTReportDataset(data_folder=data_valid, reports_file=reports_file_valid)
        # 必须像这样把 meta_file 传给 Dataset：
        self.ds = CTReportDataset(
            data_folder=data_train,
            reports_file=reports_file_train,
            meta_file=train_meta_file  # <-- 把它加回来！
        )

        self.valid_ds = CTReportDataset(
            data_folder=data_valid,
            reports_file=reports_file_valid,
            meta_file=valid_meta_file  # <-- 把它加回来！
        )

        self.dl = DataLoader(
            self.ds,
            num_workers=num_workers,
            batch_size=self.batch_size,
            shuffle = True,
            drop_last = True,  # 🚨 加上这一行：强制丢弃凑不齐的尾部数据！
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            num_workers=num_workers,
            batch_size=1,
            shuffle = False,
            drop_last = True,  # 🚨 验证集也顺手加上
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

        # 在 __init__ 的最后添加：
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

        # logs
        logs = {}

        # update CTClip model
        video, text = next(self.dl_iter)

        device=self.device
        video=video.to(device)
        mask = torch.ones((video.shape[0], video.shape[2])).bool().to(device)
        #text = text.to(device)
        text = list(text)
        text_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

        #video = video
        with self.accelerator.autocast():
            loss = self.CTClip(text_tokens, video, return_loss=True, device=device)

        self.accelerator.backward(loss)
        accum_log(logs, {'loss': loss.item()})
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.CTClip.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()
        self.print(f"{steps}: loss: {logs['loss']}")

        # if self.is_main and not (steps % self.save_results_every):
        #     with torch.no_grad():

        #         models_to_evaluate = ((self.CTClip, str(steps)),)

        #         for model, filename in models_to_evaluate:
        #             model.eval()
        #             predictedall=[]
        #             realall=[]

        #             #Fast inference on 100 images
        #             for i in range(10):
        #                 print("test")
        #                 valid_data, text, onehotlabels, name_acc = next(self.valid_dl_iter)
        #                 valid_data = valid_data.to(device)

        #                 if "module" in model.__dict__:
        #                     model = model.module

        #                 pathologies = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']
        #                 plotdir = str(self.results_folder / f'CTClip_{steps}' )
        #                 plotdir = plotdir + "/"

        #                 Path(plotdir).mkdir(parents=True, exist_ok=True)

        #                 predictedlabels=[]
        #                 for pathology in pathologies:
        #                     text = [f"There is {pathology}.", f"There is no {pathology}."]
        #                     text_tokens=self.tokenizer(
        #                                     text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
        #                     output = model(text_tokens, valid_data,  device=device)


        #                     output = apply_softmax(output)

        #                     append_out=output.detach().cpu().numpy()

        #                     if output[0]>output[1]:
        #                         predictedlabels.append(append_out[0])
        #                     else:
        #                         predictedlabels.append(append_out[0])
        #                 predictedall.append(predictedlabels)
        #                 realall.append(onehotlabels.detach().cpu().numpy()[0])
        #                 # Print and save classification report
        #             realall=np.array(realall)
        #             predictedall=np.array(predictedall)

        #             dfs=evaluate_internal(predictedall,realall,pathologies, plotdir)
        #             realall = np.rint(realall).astype(int)
        #             predictedall = np.rint(predictedall).astype(int)


        #             print('Test F1 Accuracy: ', f1_score(realall, predictedall,average='micro'))
        #             print('Test Flat Accuracy: ', accuracy_score(realall.flatten(), predictedall.flatten()),'\n')

        #             writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')

        #             dfs.to_excel(writer, sheet_name='Sheet1', index=False)

        #             writer.close()
        #             del output
        # if self.is_main and not (steps % self.save_results_every):
        #     self.print(f"🔄 Running Validation at step {steps}...")
            
        #     # 切换为 eval 模式
        #     self.CTClip.eval()
            
        #     with torch.no_grad():
        #         total_val_loss = 0.0
        #         #val_steps = 10 # 从验证集中抽取 10 个 Batch 计算平均 Loss
        #         val_steps = 2
                
        #         for _ in range(val_steps):
        #             try:
        #                 val_video, val_text = next(self.valid_dl_iter)
        #             except StopIteration:
        #                 # 如果迭代器到底了，重新循环
        #                 self.valid_dl_iter = cycle(self.valid_dl)
        #                 val_video, val_text = next(self.valid_dl_iter)

        #             val_video = val_video.to(device)
        #             val_text = list(val_text)
        #             val_text_tokens = self.tokenizer(
        #                 val_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
        #             ).to(device)

        #             # 仅计算 Loss，不收集任何分类概率
        #             with self.accelerator.autocast():
        #                 val_loss = self.CTClip(val_text_tokens, val_video, return_loss=True, device=device)
                    
        #             total_val_loss += val_loss.item()

        #         avg_val_loss = total_val_loss / val_steps
        #         self.print(f"✅ Step {steps} Validation Contrastive Loss: {avg_val_loss:.4f}\n")
                
        #     # 恢复训练模式
        #     self.CTClip.train()
        # 替换：破解 NCCL 死锁的验证逻辑 
        # 绝对不能用 self.is_main 拦截！必须所有卡同时进入！
        if not (steps % self.save_results_every):
            
            # 只有主卡负责打印
            if self.is_main:
                self.print(f"🔄 Running Validation at step {steps}...")
            
            # 切换为 eval 模式
            self.CTClip.eval()
            
            with torch.no_grad():
                total_val_loss = 0.0
                val_steps = 2
                
                for _ in range(val_steps):
                    try:
                        val_video, val_text = next(self.valid_dl_iter)
                    except StopIteration:
                        # 如果迭代器到底了，重新循环
                        self.valid_dl_iter = cycle(self.valid_dl)
                        val_video, val_text = next(self.valid_dl_iter)

                    val_video = val_video.to(device)
                    val_text = list(val_text)
                    val_text_tokens = self.tokenizer(
                        val_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
                    ).to(device)

                    # 🚨 核心：所有 16 张卡都必须执行这一句 all_gather 操作！
                    with self.accelerator.autocast():
                        val_loss = self.CTClip(val_text_tokens, val_video, return_loss=True, device=device)
                    
                    total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / val_steps
                
                # 只有主卡负责打印 Loss
                if self.is_main:
                    self.print(f"✅ Step {steps} Validation Contrastive Loss: {avg_val_loss:.4f}\n")
                
            # 恢复训练模式
            self.CTClip.train()
        # 替换结束


        # save model every so often

        # if self.is_main and not (steps % self.save_model_every):
        #     model_path = str(self.results_folder / f'CTClip.{steps}.pt')
        #     state_dict=self.accelerator.get_state_dict(self.CTClip, unwrap=False)

        #     self.accelerator.save(state_dict, model_path)

        #     self.print(f'{steps}: saving model to {str(self.results_folder)}')

        # 🚨 确保到了保存步数，所有卡都进入这个代码块
        if steps > 0 and not (steps % self.save_model_every):
            
            # 强行集合，确保所有卡都跑完了第 50 步
            self.accelerator.wait_for_everyone()
            
            # 定义一个文件夹路径（注意：分片保存必须存成文件夹，不能存成单一的 .pt 文件）
            save_dir = str(self.results_folder / f'CTClip_step_{steps}')
            
            # 🚀 终极杀招：直接让 8 张卡各自向硬盘写入自己的切片！绝对不要拼装！
            self.accelerator.save_state(save_dir)
            
            # 只有主卡负责打印日志和清理旧文件
            if self.is_main:
                self.print(f'{steps}: successfully saved sharded model to {save_dir}')
                
                if not hasattr(self, 'saved_checkpoints'):
                    self.saved_checkpoints = []
                
                self.saved_checkpoints.append(save_dir)
                
                # 保留最近 10 个 Checkpoint 文件夹，防止硬盘爆满
                if len(self.saved_checkpoints) > 10:
                    oldest_ckpt = self.saved_checkpoints.pop(0)
                    if os.path.exists(oldest_ckpt):
                        import shutil
                        try:
                            # 注意：由于保存的是文件夹，必须用 shutil.rmtree 删除
                            shutil.rmtree(oldest_ckpt)
                        except Exception as e:
                            print(f"Failed to remove {oldest_ckpt}: {e}")
                            
            # 再次集合，防止主卡删文件太慢导致进度脱节
            self.accelerator.wait_for_everyone()

        self.steps += 1
        return logs


    def train(self, log_fn=noop):
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')

        # 【新增逻辑】：在训练完全结束时，强制保存最终版本
        if self.is_main:
            final_model_path = str(self.results_folder / f'CTClip_final_step_{self.steps}.pt')
            self.accelerator.save(self.accelerator.get_state_dict(self.model), final_model_path)
            self.print(f"✅ Final model saved to {final_model_path}")


