from utils.dl.common.model import get_model_device, get_model_latency, get_model_size, set_module
from utils.dl.common.model import set_module, get_module, get_super_module
from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer,MBart50TokenizerFast,MBartForConditionalGeneration
from utils.dl.common.lr_scheduler import get_linear_schedule_with_warmup
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
from data.dataset import build_trainloader
from utils.common.log import logger
from utils.dl.common.model import get_module
import numpy as np
import torch
from utils.dl.common.env import create_tbwriter
import os
import tqdm

res_save_dir = 'G:/NMT/result'
train_batch_size = 4
val_batch_size = 4
ab_r = 8
samples_size = (1,113)
num_workers = 16
optimizer_name ='Adam'
optimizer_args ={'lr': 5e-4, 'betas': [0.9, 0.999]}
scheduler_name ='LambdaLR'
scheduler_args ={'lr_lambda': get_linear_schedule_with_warmup(10000, 70000)}
num_iters =3000
val_freq =1000

device = 'cuda'
source = "生活就像一盒巧克力。"
# tokenizer = M2M100Tokenizer.from_pretrained('G:/Transformer_model/mtom')
tokenizer = MBart50TokenizerFast.from_pretrained('G:/Transformer_model/MBART')
tokenizer.src_lang = "zh_CN"
sample = tokenizer(source,return_tensors="pt")
for k,v in sample.items():
    sample[k] = v.to(device)
from nmt.m2m import NMTmodel
fm_models_dict_path = save_models_dict_for_init({
        'main': MBartForConditionalGeneration.from_pretrained('G:/Transformer_model/MBART')
    },__file__,'MBART_pretrrained')
model = NMTmodel(name='fm',models_dict_path=fm_models_dict_path,device=device)


#1. add LoRA
lora_util = model.get_lora_util()
lora_util.add_lora_ab_to_fm(model.models_dict['main'],ab_r,sample)

#2.train



lora_params = lora_util.train_only_lora(model.models_dict['main'])
head_params = model.get_task_head_params()

num_lora_params = sum([np.prod(p.size()) for p in lora_params])
total_params = sum([np.prod(p.size()) for p in model.models_dict['main'].parameters()])
logger.info(f'num lora params: {num_lora_params}, total params: {total_params}, ratio: {num_lora_params / total_params}')

optimizer = torch.optim.__dict__[optimizer_name](lora_params+head_params,**optimizer_args)
scheduler = torch.optim.lr_scheduler.__dict__[scheduler_name](optimizer,**scheduler_args)

fbs_tb_writer = create_tbwriter(os.path.join(res_save_dir,'tb_log'), launch_tbboard=False)


best_val_acc = 0
val_acc = 0

test_path = ['dev.zh2en.txt','dev.zh2es.txt','dev.zh2fr.txt','dev.zh2ru.txt']
train_path = ['train.zh2en.txt','train.zh2es.txt','train.zh2fr.txt','train.zh2ru.txt']

pbar = tqdm.tqdm(range(num_iters), dynamic_ncols=True)
test_loader = build_trainloader(val_batch_size, test_path[3], 0)
train_loader = iter(build_trainloader(train_batch_size, train_path[3], 0))

for iter_index in pbar:
    if iter_index == num_iters:
        break
    model.to_train_mode()
    x = next(train_loader)

    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(device)
    task_loss = model.infer(x).loss
    optimizer.zero_grad()
    task_loss.backward()
    optimizer.step()
    scheduler.step()

    if (iter_index + 1) % val_freq == 0:
        model.to_eval_mode()
        val_acc = model.get_accuracy(test_loader)
        model.save_model(os.path.join(res_save_dir, 'models/fm_last3.pt'))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_model(os.path.join(res_save_dir, 'models/fm_best3.pt'))

    fbs_tb_writer.add_scalar(f'losses/task_loss', task_loss, iter_index)
    fbs_tb_writer.add_scalar(f'accs/val_acc', val_acc, iter_index)
    fbs_tb_writer.add_scalar(f'lr', optimizer.param_groups[0]['lr'], iter_index)
    pbar.set_description(f'loss: {task_loss:.6f}, val_acc: {val_acc:.4f}')

