#from transformers import BartTokenizer, BartForConditionalGeneration
# from modeling_bart import BartForConditionalGeneration
from transformers import BertTokenizer, BertForMaskedLM

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

def cos_similarity(mat1, mat2, temperature):
    norm1 = torch.norm(mat1, p=2, dim=1).view(-1, 1)
    norm2 = torch.norm(mat2, p=2, dim=1).view(1, -1)
    # import pdb
    # pdb.set_trace()
    cos_sim = torch.matmul(mat1, mat2.t()) / torch.matmul(norm1, norm2)
    norm_sim = torch.exp(cos_sim / temperature)
    return norm_sim

def info_NCE(sim_pos, sim_total):
    deno = torch.sum(sim_total) - sim_total[0][0] + sim_pos[0][0]
    loss = -torch.log(sim_pos[0][0] / deno)
    return loss

def instance_CL_Loss(ori_hidden, aug_hidden, type="origin", temp=0.5):
    inputs_mean = ori_hidden
    positive_examples_mean = aug_hidden
    batch_size = inputs_mean.size()[0]
    cons_loss = 0
    count = 0
    for ori_input, pos_exp in zip(inputs_mean, positive_examples_mean):
        ori_input = torch.reshape(ori_input, (1, -1))
        pos_exp = torch.reshape(pos_exp, (1, -1))
        # import pdb
        # pdb.set_trace()
        sim_pos = cos_similarity(ori_input, pos_exp, temp)
        '''魔改对比学习，让他去和其余的负样本对比'''
        # negative=torch.Tensor()
        if type == "origin":
            negative = positive_examples_mean[torch.arange(positive_examples_mean.size(0)) != count]  # batch中其他样本作为negative
        elif type == "aug":
            negative = positive_examples_mean[torch.arange(positive_examples_mean.size(0)) != count]  # 和其他batch样本的keyword 拉大
        count += 1
        # import pdb
        # pdb.set_trace()
        negative = torch.reshape(negative, (batch_size-1, -1))
        sim_total = cos_similarity(ori_input, negative, temp)  ##修改为了正样例的平均
        cur_loss = info_NCE(sim_pos, sim_total)
        cons_loss += cur_loss

    return cons_loss / batch_size



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
    

    args = parser.parse_args()
    random.seed(args.seed)
    
    if args.do_train:
        print("训练模式")
    #if args.test_only:
    #    print("测试模式")
    
    
    print("loading train & test data")
    
    train_tokens = KBQA_origin_text(args.train_data)
    # import pdb
    # pdb.set_trace()
    train_mask_tokens = KBQA_mask_text(args.train_data)
    
    print("finish loading")

    print("tokenizer loading")
    tokenizer = BertTokenizer.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/bert-base-uncased")
    print("finish loading")
    
    
    train_origin_ids = tokenizer(train_tokens, return_tensors="pt",padding="max_length",max_length=200,truncation=True)["input_ids"].to(device) 
    print("train_origin_ids finish loading")

    # import pdb
    # pdb.set_trace()

    train_mask_ids = tokenizer(train_mask_tokens, return_tensors="pt",padding="max_length",max_length=200,truncation=True)["input_ids"].to(device)
    print("train_augment_ids finish loading")  
    
    train_ids = TensorDataset(train_mask_ids,train_origin_ids)   
    train_loader = DataLoader(dataset=train_ids, batch_size=args.batch_size, shuffle=False)
    
    
    print("data loader process success")
    print(f"batch_size:{args.batch_size}")

    
    model = BertForMaskedLM.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/bert-base-uncased")
    model.to(device)
    count=0
    
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = args.learning_rate
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
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
            tq = tqdm(enumerate(train_loader, 0)) 
            for step, data in enumerate(train_loader, 0): #这里的data就是train_loader的每一个元组（mask的输入embedding，原文输入embedding）
                
                input_ids, origin_ids = data
                # import pdb
                # pdb.set_trace()
                #logits = model(input_ids).logits.to(device)   
                logits = model(input_ids).logits.to(device)
                
                mlm_loss = model(input_ids,labels=origin_ids).loss
                
                
                #正样本
                ori_hidden = model(origin_ids,output_hidden_states=True).hidden_states[-1]
                dropout = nn.Dropout(0.1)
                aug_hidden = dropout(ori_hidden)

                cl_loss = instance_CL_Loss(ori_hidden, aug_hidden, 'origin', 0.5)

                loss = mlm_loss+ 0.2 * cl_loss

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