#from transformers import BartTokenizer, BartForConditionalGeneration
# from modeling_bart import BartForConditionalGeneration
from transformers import BertTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling

import os
import pdb
import random
import torch 
from torch import nn
import torchvision.models as models
import argparse
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from datasets import load_dataset

def list_to_str(list):
    item_all=""
    for item in list:
        item = item+" "
        item_all=item_all+item
    return item_all[:-1]

#mask所有有标签的词
def ner_mask(text,label):
    for i in range (0,len(label)):
        if label[i]!="O":
            text[i]="[MASK]"
    return text, label

#返回mask原文, mask所有有标签的词
def ner_preprocess(file_path: str,file_path2: str): 
    token_file_path = file_path
    label_file_path = file_path2
    with open(token_file_path, "r", encoding="utf-8") as f_token:
        token_lines = f_token.readlines()
    with open(label_file_path, "r", encoding="utf-8") as f_label:
        label_lines = f_label.readlines()
    # assert len(token_lines) == len(label_lines)
    token_all=[]
    # import pdb
    # pdb.set_trace()
    for token_line, label_line in zip(token_lines, label_lines):
        if not token_line.strip() or not label_line.strip():
            continue
        tokens = token_line.strip().split(" ")
        labels = label_line.strip().split(" ")
       
        # import pdb
        # pdb.set_trace()
        assert len(tokens)==len(labels),"输入与标签存在不相等"
        
        tokens,labels=ner_mask(tokens,labels)
        
        if len(tokens) == 0 or len(labels) == 0:
            continue
        tokens = [token.strip() for token in tokens if token.strip()]
        
        
        tokens=list_to_str(tokens)  #转化成字符串进行append
        
        # import pdb
        # pdb.set_trace()
        
        token_all.append(tokens)
        
    return token_all 

#返回未处理的原文
def ner_origin_text(file_path: str,file_path2: str):    ##读取原文
    '''返回原文'''
    token_file_path = file_path
    label_file_path = file_path2
    with open(token_file_path, "r", encoding="utf-8") as f_token:
        token_lines = f_token.readlines()
    with open(label_file_path, "r", encoding="utf-8") as f_label:
        label_lines = f_label.readlines()
    # assert len(token_lines) == len(label_lines)
    token_all=[]
    label_all=[]
    origin_all=[]
    # import pdb
    # pdb.set_trace()
    for token_line, label_line in zip(token_lines, label_lines):
        if not token_line.strip() or not label_line.strip():
            continue
        tokens = token_line.strip().split(" ")
        labels = label_line.strip().split(" ")
        origin_tokens=tokens   #加载了原文本
        origin_all.append(origin_tokens)
        
        
        if len(tokens) == 0 or len(labels) == 0:
            continue
        tokens = [token.strip() for token in tokens if token.strip()]
        labels = [label.strip() for label in labels if label.strip()]
        
        tokens=list_to_str(tokens)  #转化成字符串进行append
        labels=list_to_str(labels)
        # import pdb
        # pdb.set_trace()
        
        token_all.append(tokens)
        label_all.append(labels)
    
    return token_all

def KBQA_origin_text(file_path: str):    ##读取原文
    '''返回原文'''
    token_file_path = file_path


    
    with open(token_file_path, 'r') as ouf:
        reader = csv.reader(ouf, delimiter='\t')
        token_all=[]
        for row in tqdm(reader):
            id,text,title = row
            # import pdb
            # pdb.set_trace()
            text = text.strip().split(" ")
            tokens=list_to_str(text)
            token_all.append(tokens)
            # import pdb
            # pdb.set_trace()
    return token_all

def KBQA_mask_text(file_path: str):    ##读取原文
    '''返回原文'''
    token_file_path = file_path


    
    with open(token_file_path, 'r') as ouf:
        reader = csv.reader(ouf, delimiter='\t')
        token_all=[]
        for row in tqdm(reader):
            id,text,title = row
            # import pdb
            # pdb.set_trace()
            # [MASK]
            text = text.strip().split(" ")
            text=random_mask(text)
            tokens=list_to_str(text)
            token_all.append(tokens)
            # import pdb
            # pdb.set_trace()
    return token_all


def random_mask(text):
    mask_id=random.randint(0,len(text)-1)
    text[mask_id]="[MASK]"
    return text




def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', help="train mode")
    parser.add_argument('--epoch', type=int, default=5, help='epoch num')
    parser.add_argument('--train_data', default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR_Input_DATA/dpr_inputs_debug/individual_relations/all_relations_complex.tsv", help='train_data dir')
    parser.add_argument('--train_data2', default='/data1/lxf2020/bart-maskfilling/data_pretrain/inter_train_10_5_seq_out.txt', help='train_data dir')
    parser.add_argument('--model_save_dir', default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/bert_pretraining_checkpoints', help='train_data dir')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='crop size of image')
    parser.add_argument('--batch_size', type=int, default=4, help='epoch num')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--mlm_probability', type=int, default=0.15, help='random seed')
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    

    args = parser.parse_args()
    random.seed(args.seed)
    
    if args.do_train:
        print("训练模式")
    #if args.test_only:
    #    print("测试模式")
    
    
    print("loading train & test data")
    
    train_tokens = KBQA_origin_text(args.train_data)

    print("finish loading")

    print("tokenizer loading")
    tokenizer = BertTokenizer.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/bert-base-uncased")
    print("finish loading")
    
    
    train_origin_ids = tokenizer(train_tokens, return_tensors="pt",padding="max_length",max_length=200,truncation=True)["input_ids"].to(device) 
    print("train_origin_ids finish loading")


    
    #在这里计算mask的数据
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
    # import pdb
    # pdb.set_trace()
    # train_ids = TensorDataset(train_origin_ids)
    # import pdb
    # pdb.set_trace()
    # train_ids = train_origin_ids   
    train_loader = DataLoader(dataset=train_origin_ids.to(device) , collate_fn=data_collator,batch_size=args.batch_size, shuffle=False)
    # import pdb
    # pdb.set_trace()
    
    print("data loader process success")
    print(f"batch_size:{args.batch_size}")

    
    model = BertForMaskedLM.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/bert-base-uncased")
    model.to(device)
    count=0
    
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = args.learning_rate
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    running_loss = 0.0


    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir, exist_ok=True)
        print("创建保存路径")


    if args.do_train:
        print("------------------------------")
        print("training begining!")
        print("------------------------------")
        model.train()
        for epoch in range(args.epoch):
            tq = tqdm(enumerate(train_loader)) 
            for step, batch in enumerate(train_loader,0): #这里的data就是train_loader的每一个元组（mask的输入embedding，原文输入embedding）
                
                # import pdb
                # pdb.set_trace()

                
                #logits = model(input_ids).logits.to(device)   
                logits = model(input_ids).logits.to(device)
                
                loss = model(**batch).loss
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                count+=1
                
                running_loss += loss.item()
                
                if step % 20 == 19:   #打印每2000步mini batch
                    tq.set_description('[epoch: %d, step: %5d] loss: %.3f' % (epoch + 1, count+1, running_loss / 20))
                    running_loss = 0.0
            count = 0
            # import pdb
            # pdb.set_trace()
            
            torch.save(model.state_dict(),os.path.join(args.model_save_dir,f'bert_MLM_epoch{epoch}.bin'))
        print("finish training")
           
if __name__ == "__main__":
    main()
    

           
    
    
    
    
    
    
    
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    # # DataLoaders creation:
    # train_dataloader = DataLoader(
    #     train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    # )
    # eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # # Optimizer
    # # Split weights in two groups, one with weight decay and the other not.
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)