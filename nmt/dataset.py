#处理数据集，困难
import os
from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer,MBart50TokenizerFast,MBartForConditionalGeneration
import torch
from torch.utils.data import Dataset
data_name = []#依次读入数据
class NMTvaldata(Dataset):
    def __init__(self,data_name):
        self.text,self.label = self.read_data_nmt(data_name)
        self.name = data_name.split('.')[1]
        self.src_lang = self.name.split('2')[0]
        self.tgt_lang = self.name.split('2')[1]
        if self.tgt_lang == 'en':
            self.tgt_lang = "en_XX"
        elif self.tgt_lang == 'fr':
            self.tgt_lang = "fr_XX"
        elif self.tgt_lang == 'es':
            self.tgt_lang = "es_XX"
        elif self.tgt_lang == 'ru':
            self.tgt_lang = "ru_RU"
        #self.tokenizer = M2M100Tokenizer.from_pretrained('G:/Transformer_model/mtom', src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        self.tokenizer = MBart50TokenizerFast.from_pretrained('G:/Transformer_model/MBART', src_lang="zh_CN",
                                                              tgt_lang=self.tgt_lang)
        self.dataset = self.tokenizer(self.text, text_target=self.label,return_tensors="pt", padding=True, truncation=True,max_length=512)
        self.length = len(self.text)
    def __getitem__(self,item):
        return {
            'input_ids':self.dataset['input_ids'][item],
            'attention_mask':self.dataset['attention_mask'][item]
        },self.label[item],self.tgt_lang

    def __len__(self):
        return self.length
    def read_data_nmt(self,data_name):
        text = []
        labels = []
        data_dir = 'F:/data/dev/dev'
        with open(os.path.join(data_dir, data_name), 'r', encoding='utf-8') as f:
            read_txt = f.read()
            lines = read_txt.split('\n')
            for line in lines:
                x = line.split('\t')
                if len(x) < 2:
                    continue
                else:
                    text.append(x[0])
                    labels.append(x[1])
            return text, labels

class NMTdata(Dataset):
    def __init__(self,data_name):
        self.text,self.label = self.read_data_nmt(data_name)
        self.name = data_name.split('.')[1]
        self.src_lang = self.name.split('2')[0]
        self.tgt_lang = self.name.split('2')[1]
        if self.tgt_lang == 'en':
            self.tgt_lang = "en_XX"
        elif self.tgt_lang == 'fr':
            self.tgt_lang = "fr_XX"
        elif self.tgt_lang == 'es':
            self.tgt_lang = "es_XX"
        elif self.tgt_lang == 'ru':
            self.tgt_lang = "ru_RU"
        #self.tokenizer = M2M100Tokenizer.from_pretrained('G:/Transformer_model/mtom', src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        self.tokenizer = MBart50TokenizerFast.from_pretrained('G:/Transformer_model/MBART', src_lang="zh_CN",
                                                         tgt_lang=self.tgt_lang)
        self.dataset = self.tokenizer(self.text, text_target=self.label, return_tensors="pt", padding=True, truncation=True,
                                 max_length=512)
        self.length = len(self.text)
    def __getitem__(self,item):
        return {
            'input_ids':self.dataset['input_ids'][item],
            'attention_mask':self.dataset['attention_mask'][item],
            'labels':self.dataset['labels'][item]
        }


    def __len__(self):
        return self.length
    def read_data_nmt(self,data_name):
        text = []
        labels = []
        data_dir = 'F:/data/train/train'
        with open(os.path.join(data_dir, data_name), 'r', encoding='utf-8') as f:
            read_txt = f.read()
            lines = read_txt.split('\n')
            for line in lines:
                x = line.split('\t')
                if len(x) < 2:
                    continue
                else:
                    text.append(x[0])
                    labels.append(x[1])
            return text, labels
# class NMTdata(Dataset):
#     def __init__(self,data_names):
#         self.dataset = {"input_ids":[],
#                         "attention_mask":[],
#                         "labels":[]}
#         for data_name in data_names:
#             self.text,self.label = self.read_data_nmt(data_name)
#             self.name = data_name.split('.')[1]
#             self.src_lang = self.name.split('2')[0]
#             self.tgt_lang = self.name.split('2')[1]
#             self.tokenizer = M2M100Tokenizer.from_pretrained('G:/Transformer_model/mtom', src_lang=self.src_lang, tgt_lang=self.tgt_lang)
#             self.dataset1 = self.tokenizer(self.text, text_target=self.label, return_tensors="pt", padding=True, truncation=True,
#                                      max_length=512)
#             self.dataset.update(self.dataset1)
#         self.length = len(self.text)
#     def __getitem__(self,item):
#         return {
#             'input_ids':self.dataset['input_ids'][item],
#             'attention_mask':self.dataset['attention_mask'][item],
#             'labels':self.dataset['labels'][item]
#         }
#
#
#     def __len__(self):
#         return self.length
#     def read_data_nmt(self,data_name):
#         text = []
#         labels = []
#         data_dir = 'F:/data/train/train'
#         with open(os.path.join(data_dir, data_name), 'r', encoding='utf-8') as f:
#             read_txt = f.read()
#             lines = read_txt.split('\n')
#             for line in lines:
#                 x = line.split('\t')
#                 if len(x) < 2:
#                     continue
#                 else:
#                     text.append(x[0])
#                     labels.append(x[1])
#             return text, labels
    # def read_data_nmt(self,data_names):
    #     text = []
    #     labels = []
    #     data_dir = 'F:/data/train/train'
    #     for data_name in data_names:
    #         with open(os.path.join(data_dir, data_name), 'r', encoding='utf-8') as f:
    #             read_txt = f.read()
    #             lines = read_txt.split('\n')
    #             for line in lines:
    #                 x = line.split('\t')
    #                 if len(x) < 2:
    #                     continue
    #                 else:
    #                     text.append(x[0])
    #                     labels.append(x[1])
    #     return text, labels

class NMTtestdata(Dataset):
    def __init__(self,data_name):
        self.text,self.label = self.read_data_nmt(data_name)
        self.name = data_name.split('.')[1]
        #self.src_lang = self.name.split('2')[0]
        self.tgt_lang = data_name.split('.')[1]
        self.tokenizer = M2M100Tokenizer.from_pretrained('G:/Transformer_model/mtom')
        #self.tokenizer = MBart50TokenizerFast.from_pretrained('G:/Transformer_model/MBART')
        self.tokenizer.src_lang = "zh"
        self.dataset = self.tokenizer(self.text,return_tensors="pt", padding=True, truncation=True)
        self.length = len(self.text)

    def __getitem__(self,item):
        return {
            'input_ids':self.dataset['input_ids'][item],
            'attention_mask':self.dataset['attention_mask'][item]
        },self.tgt_lang


    def __len__(self):
        return self.length
    def read_data_nmt(self,data_name):
        text = []
        labels = []
        data_dir = 'F:/data/test/test'
        with open(os.path.join(data_dir, data_name), 'r', encoding='utf-8') as f:
            read_txt = f.read()
            lines = read_txt.split('\n')
            for line in lines:
                if len(line)==0:
                    continue
                text.append(line)
            return text, labels


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    # def __init__(self,batch_size,num_workers,dataset):
    #     super().__init__()
    #     dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size)
    #     self._infinite_iterator = iter(dataloader)
    def __init__(self, batch_size, num_workers, dataset,shuffle=False, collate_fn=None):
        super().__init__()

        self.num_workers = num_workers

        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=False,
        )
        if collate_fn is not None:
            self._infinite_iterator = iter(
                torch.utils.data.DataLoader(
                    dataset,
                    num_workers=num_workers,
                    batch_sampler=_InfiniteSampler(batch_sampler),
                    pin_memory=False,
                    collate_fn=collate_fn
                )
            )
        else:
            self._infinite_iterator = iter(
                torch.utils.data.DataLoader(
                    dataset,
                    num_workers=num_workers,
                    batch_sampler=_InfiniteSampler(batch_sampler),
                    pin_memory=False,
                )
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self._length = len(batch_sampler)

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length


# def read_data_nmt(data_name):
#     text = []
#     labels = []
#     data_dir = 'F:/data/train/train'
#     with open(os.path.join(data_dir,data_name),'r',encoding='utf-8') as f:
#         read_txt = f.read()
#         lines = read_txt.split('\n')
#         for line in lines:
#             x = line.split('\t')
#             if len(x)<2:
#                 continue
#             else:
#                 text.append(x[0])
#                 labels.append(x[1])
#         return text,labels


def build_trainloader(batch_size,data_name,num_workers):#这一步再加tokenizer
    # name = data_name.split('.')[1]
    # src_lang = name.split('2')[0]
    # tgt_lang = name.split('2')[1]
    #tokenizer = M2M100Tokenizer.from_pretrained('G:/Transformer_model/mtom',src_lang=src_lang, tgt_lang=tgt_lang)
    #dataset = tokenizer(text,text_target=label,return_tensors="pt",padding=True,truncation=True,max_length=512)
    if data_name.split('.')[0] == 'train':
        dataset = NMTdata(data_name)
    elif data_name.split('.')[0] == 'test':
        dataset = NMTtestdata(data_name)
    else:
        dataset = NMTvaldata(data_name)
    dataloader = InfiniteDataLoader(batch_size,num_workers,dataset)
    return dataloader

# test = iter(build_trainloader(256,'dev.zh2en.txt',0))
# print(next(test))
