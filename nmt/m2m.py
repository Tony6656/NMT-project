import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import List
from copy import deepcopy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import tqdm
import torch.nn.functional as F
from utils.dl.common.model import get_model_device, get_model_latency, get_model_size, set_module
from utils.dl.common.model import set_module, get_module, get_super_module
from utils.common.log import logger
import math
from base.model import BaseModel
from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer,MBart50TokenizerFast,MBartForConditionalGeneration
from nmt.bleu import BLEU
from nltk.translate.bleu_score import sentence_bleu



class LoRA(nn.Linear):
    pass


class FMLoRA_Util(ABC):
    @abstractmethod
    def add_lora_ab_to_fm(self, fm: nn.Module, ab_r: int, samples: torch.Tensor):
        """
        only applying LoRA to attention weights.
        """
        pass

    def train_only_lora(self, fm: nn.Module):
        res = []
        for n, m in fm.named_modules():
            if isinstance(m, LoRA):
                for p in m.parameters():
                    p.requires_grad = True
                    res += [p]
            else:
                for p in m.parameters():
                    p.requires_grad = False
        return res

    @abstractmethod
    def absorb_lora_and_recover_net_structure(self, fm: nn.Module):
        pass



class ToQKV_WrappedWithLoRA(nn.Module):
    def __init__(self, fc: nn.Linear, ab_r: int):
        super(ToQKV_WrappedWithLoRA, self).__init__()

        self.fc = fc
        self.ab = self.create_ab_as_linear(fc.weight.data, ab_r)

    def create_ab_as_linear(self, fc_weight: torch.Tensor, ab_r: int):
        res = nn.Sequential(
            LoRA(fc_weight.size(1), fc_weight.size(0) // ab_r, bias=False),
            LoRA(fc_weight.size(0) // ab_r, fc_weight.size(0), bias=False)
        ).to(fc_weight.device)
        nn.init.kaiming_uniform_(res[0].weight, a=5 ** 0.5)
        nn.init.zeros_(res[1].weight)
        return res

    def forward(self, x):
        x1 = self.fc(x)
        x2 = self.ab(x)
        return x1 + x2


class FMLoRA_m2m_Util(FMLoRA_Util):

    @torch.no_grad()
    def add_lora_ab_to_fm(self, fm: nn.Module, ab_r: int, samples: torch.Tensor):
        fm.eval()

        # print(samples)
        # for k, v in samples.items():
        #     if isinstance(v, torch.Tensor):
        #         samples[k] = v.to(get_model_device(fm))

        o1 = fm.generate(**samples)
        #o1 = fm(samples)
        for name, module in fm.named_modules():
            if name.endswith(('query', 'key', 'value')):
                set_module(fm, name, ToQKV_WrappedWithLoRA(module, ab_r))
            elif name.endswith('.qkv'):
                set_module(fm, name, ToQKV_WrappedWithLoRA(module, ab_r))
            elif name.endswith(('k_proj', 'q_proj', 'v_proj')):
                set_module(fm, name, ToQKV_WrappedWithLoRA(module, ab_r))

        o2 = fm.generate(**samples)
        #o2 = fm(samples)
        if isinstance(o1, tuple):
            o1 = o1[-1]
            o2 = o2[-1]
        output_diff = ((o1 - o2) ** 2).sum()
        assert output_diff < 1e-5
        return fm

    @torch.no_grad()
    def absorb_lora_and_recover_net_structure(self, fm: nn.Module, samples: torch.Tensor):
        fm.eval()
        # print('absorb lora before')

        # for k, v in samples.items():
        #     if isinstance(v, torch.Tensor):
        #         samples[k] = v.to(get_model_device(fm))

        o1 = fm(samples)

        for name, module in fm.named_modules():
            if not isinstance(module, ToQKV_WrappedWithLoRA):
                continue

            fc = module.fc
            ab = module.ab

            fc.weight.add_(ab[1].weight @ ab[0].weight)

            set_module(fm, name, fc)

        # print('absorb lora after')
        o2 = fm(samples)

        if isinstance(o1, tuple):
            o1 = o1[-1]
            o2 = o2[-1]
        output_diff = ((o1 - o2) ** 2).sum()
        assert output_diff < 1e-6, output_diff

        return fm

class NMTmodel(BaseModel):
    def __init__(self,name:str,models_dict_path: str,device:str):
        super().__init__(name,models_dict_path,device)
        self.tokenizer = M2M100Tokenizer.from_pretrained('G:/Transformer_model/mtom')
        #self.tokenizer = MBart50TokenizerFast.from_pretrained('G:/Transformer_model/MBART')

    def get_required_model_components(self) -> List[str]:
        return ['main']
    def get_accuracy(self,test_loader,*args,**kwargs):
        acc = 0
        sample_num = 0
        device = self.device
        self.to_eval_mode()
        import tqdm
        phar = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=False,dynamic_ncols=True)
        with torch.no_grad():
            for batch_index, (x,y,z) in phar:
                if batch_index==125:
                    break
                for k, v in x.items():
                    x[k] = v.to(device)
                output = self.infer(x,z)
                pred = self.tokenizer.batch_decode(output,skip_special_tokens=True)
                #correct = torch.eq(pred,y).sum().item()
                bleu = BLEU(pred,y)
                # bleu = 0
                # for i in range(len(pred)):
                #     candidate = pred[i].strip().split()
                #     print(candidate)
                #     reference = y[i].strip().split()
                #     print(reference)
                #     bleu += sentence_bleu(reference, candidate,weights=(0.25, 0.25, 0.25, 0.25))
                #acc += correct
                # bleu /= len(pred)
                acc+=bleu
                sample_num += 1

                phar.set_description(f'cur_batch_total: {len(y)},cur_batch_acc: {acc/sample_num:.4f}')
        #acc /= sample_num
        # bleu /= sample_num
        acc /= sample_num
        return acc

    def infer(self, x, z=None,*args, **kwargs):
        if z==None:
            return self.models_dict['main'](**x)
        else:
            return self.models_dict['main'].generate(**x,forced_bos_token_id=self.tokenizer.get_lang_id(z[0]))
            #return self.models_dict['main'].generate(**x, forced_bos_token_id=self.tokenizer.lang_code_to_id[z[0]])
    def get_lora_util(self) -> FMLoRA_Util:
        return FMLoRA_m2m_Util()

    def get_task_head_params(self):
        head = get_module(self.models_dict['main'],'lm_head')
        return list(head.parameters())
