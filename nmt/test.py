from nmt.m2m import NMTmodel
from utils.common.exp import save_models_dict_for_init, get_res_save_dir
import torch
import os
from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer,MBart50TokenizerFast,MBartForConditionalGeneration
from data.dataset import build_trainloader
import string
device = 'cuda'
fm_models_dict_path = 'G:/NMT/fm_best.pt'
fm_models = torch.load(fm_models_dict_path)
fm_models_dict_path = save_models_dict_for_init(fm_models,__file__,'fm_nmt')
# fm_models_dict_path = save_models_dict_for_init({
#         'main': MBartForConditionalGeneration.from_pretrained('G:/Transformer_model/MBART')
#     },__file__,'mbart_pretrrained')
# model = NMTmodel(name='fm',models_dict_path=fm_models_dict_path,device=device)

model = NMTmodel('fm',fm_models_dict_path,device)

def read_data_nmt(data_name):
    text = []
    labels = []
    data_dir = 'F:/data/test/test/'
    with open(os.path.join(data_dir,data_name),'r',encoding='utf-8') as f:
        read_txt = f.read()
        lines = read_txt.split('\n')
        for line in lines:
            x = line.split('\t')
            if len(x)<2 and len(x)!=0:
                text.append(x[0])
            else:
                text.append(x[0])
                labels.append(x[1])
        return text,labels

data_names = ['test.en.zh.txt','test.es.zh.txt','test.fr.zh.txt','test.ru.zh.txt']

model.to_eval_mode()
tokenizer = M2M100Tokenizer.from_pretrained('G:/Transformer_model/mtom')
#tokenizer = MBart50TokenizerFast.from_pretrained('G:/Transformer_model/MBART')
tokenizer.src_lang = "zh"
id = 1500
# for data_name in data_names:
    # if data_name != data_names[3]:
    #     continue
# z = data_name.split('.')[1]
test_loader = iter(build_trainloader(1,'test.ru.zh.txt', 0))
index = 500
for batch, (x, z) in enumerate(test_loader):
    if batch == 500:
        break
    for k, v in x.items():
        x[k] = v.to(device)
    res = list(z)

    if res[0] == 'en':
        res[0] = "en_XX"
    elif res[0] == 'fr':
        res[0] = "fr_XX"
    elif res[0] == 'es':
        res[0] = "es_XX"
    elif res[0] == 'ru':
        res[0] = "ru_RU"

    output = model.infer(x,z)
    pred = tokenizer.batch_decode(output, skip_special_tokens=True)
    for i in range(len(pred)):
        id += 1
        str1 = str(id) + '\t' + pred[i] + '\t' + z[0]
        # str1 = pred[i]
        with open('text4.txt', 'a', encoding='utf-8') as f:
            f.writelines(str1)
            f.writelines('\n')



