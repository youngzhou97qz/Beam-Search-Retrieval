import math
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_weights = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)      # tokenizer.vocab_size = 21128
token_id = tokenizer.convert_tokens_to_ids
mask_model = BertForMaskedLM.from_pretrained('bert-base-chinese').to(device)
mask_model.eval()
ser = 'dango'

# data part
# reading data
questions, answers, answer_ids = [], [], []
f = open('/home/'+ser+'/STC3/data/questions.txt','r',encoding='gbk')
lines = f.readlines()
for line in lines:
    line = line.strip()
    questions.append(line)
f.close()
f = open('/home/'+ser+'/STC3/data/answers.txt','r',encoding='gbk')
lines = f.readlines()
for line in lines:
    line = line.strip()
    answers.append(line)
f.close()
f = open('/home/'+ser+'/STC3/data/answers_id.txt','r',encoding='gbk')
lines = f.readlines()
for line in lines:
    line = line.strip()
    answer_ids.append(int(line))
f.close()

# judging chinese
def check_contain_chinese(check_str):
    length = len(check_str)
    count = 0
    for ch in check_str:
        if '\u4e00' <= ch <= '\u9fff':
            count += 1
    if count >= length // 3:  # [泪]
        return True
    else:
        return False

# delete sentences
i = len(questions)-1
while i >= 0:
    if answer_ids[i] != 4:
        questions.pop(i)
        answers.pop(i)
    elif check_contain_chinese(questions[i])==False or check_contain_chinese(answers[i])==False or len(questions[i])==0 or len(answers[i])==0:
        questions.pop(i)
        answers.pop(i)
    i -= 1
# print('问答对：', len(questions))    #   1630292  anger: 184590

# standardization
import string
punc = string.punctuation + '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'

def del_punc_dup(text,slide_len,punc):  # // 2
    for j in range(slide_len, len(text)-slide_len+1):
        if text[j:j+slide_len] == text[j-slide_len:j]:
            for char in text[j:j+slide_len]:
                if char in punc or (len(char)>2 and char[:2]=='##'):
                    return text[:j] + text[j+slide_len:]
                    break
    return text

def del_char_dup(text,slide_len):  # // 4
    for j in range(3*slide_len, len(text)-slide_len+1):
        if text[j:j+slide_len] == text[j-slide_len:j] and text[j-2*slide_len:j-slide_len] == text[j-slide_len:j] and text[j-2*slide_len:j-slide_len] == text[j-3*slide_len:j-2*slide_len]:
            return text[:j] + text[j+slide_len:]
            break
    return text

def pre_process(text, punc):  # 去除多余空格和''，保留一定数量的重复元素
    for i in tqdm(range(len(text))):
        text[i] = tokenizer.tokenize(text[i])
        slide_len = len(text[i]) // 2
        while slide_len >= 1:
            origin_text = ''
            while text[i] != origin_text:
                origin_text = text[i]
                text[i] = del_punc_dup(text[i],slide_len,punc)
            slide_len -= 1
        slide_len = len(text[i]) // 4
        while slide_len >= 1:
            origin_text = ''
            while text[i] != origin_text:
                origin_text = text[i]
                text[i] = del_char_dup(text[i],slide_len)
            slide_len -= 1
        new = text[i][0]
        for j in range(1,len(text[i])):
            if len(text[i][j]) > 2 and text[i][j][:2] == '##':
                new = new + text[i][j][2:]
            else:
                new = new + ' ' + text[i][j]
        text[i] = new
    return text

questions = pre_process(questions, punc)
answers = pre_process(answers, punc)

# answer vocabulary
import collections
def get_dict(answers):
    char_answ = []
    for i in range(len(answers)):
        answers[i] = tokenizer.tokenize(answers[i])
        for j in range(len(answers[i])):
            char_answ.append(answers[i][j])
    answ_dict = collections.Counter(char_answ)
#     rest_answ = dict(filter(lambda x: (x[1] > 250 and (x[0] >= '\u4e00' and x[0] <= '\u9fff')) or (x[1] > 500 and (x[0] < '\u4e00' or x[0] > '\u9fff')), answ_dict.items()))
    rest_answ = dict(filter(lambda x: (x[1] > 50 and (x[0] >= '\u4e00' and x[0] <= '\u9fff')) or (x[1] > 100 and (x[0] < '\u4e00' or x[0] > '\u9fff')), answ_dict.items()))
    count = 2
    for key in rest_answ.keys():
        rest_answ[key] = count
        count += 1
    rest_answ['[SEP]'], rest_answ['[OOV]'] = 0, 1
    return rest_answ
char2id = get_dict(answers)
id2char = {value:key for key, value in char2id.items()}
# print('词表数：', len(char2id))    #   2495  anger: 1918

# ids conversion 
def id2id(ids, mode='bert2answ'):
    if mode == 'bert2answ':
        text = tokenizer.convert_ids_to_tokens([ids])[0]
        if text in char2id.keys():
            ids = char2id[text]
        else:
            ids = 1
    elif mode == 'answ2bert':
        text = id2char[ids]
        ids = tokenizer.convert_tokens_to_ids(text)
    return ids

# train & valid data
temp = [(ques, answ) for ques, answ in zip(questions, answers)]
temp.sort(key = lambda i: len(i[1]), reverse=True)
questions = [ques for ques, answ in temp]
answers = [answ for ques, answ in temp]

def data_loader(ques, answ, batch_size, max_len, model):
    count = 0
    while count < len(ques):
        batch = []
        size = min(batch_size, len(ques) - count)
        for _ in range(size):
            part1 = tokenizer.encode(prediction_replace(ques[count], max_len, model))
            part2 = tokenizer.encode(answ[count])
            truncate_tokens(part1, part2, max_len-2)
            tokens = part1 + token_id(['[SEP]']) + part2 + token_id(['[SEP]'])
            temp_tokens = part1 + token_id(['[SEP]'])
            num = len(part1)+1
            segment_ids = [0]*(num) + [1]
            input_mask = [1]*(num+1)
            masked_tokens, masked_pos = [], []
            masked_tokens.append(id2id(tokens[num], mode='bert2answ'))
            masked_pos.append(num)
            n_pad = max_len - num - 1
            tokens.extend([0]*(max_len - len(tokens)))
            temp_tokens.extend([0]*(max_len - len(temp_tokens)))
            segment_ids.extend([0]*n_pad)
            input_mask.extend([0]*n_pad) 
            batch.append((tokens, temp_tokens, segment_ids, input_mask, masked_pos, masked_tokens, len(part2)+1))
            count += 1
        yield batch

# using BERT to replace characters
def prediction_replace(sentence, max_len, model, rate=0.1):
    output_text = tokenizer.tokenize(sentence)
    num = int(len(output_text)//((max_len//2-1)/((max_len//2-1)*rate+1)))
    if num > 0:
        random_sequence = list(range(len(output_text)))
        random.shuffle(random_sequence)
        count = 0
        for index in random_sequence:
            tokenized_text = tokenizer.tokenize(sentence)
            reference_text = tokenized_text[index]
            tokenized_text[index] = '[MASK]'
            tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)]).to(device)
            segments_ids = torch.tensor([[0] * len(tokenized_text)]).to(device)
            with torch.no_grad():
                outputs = model(tokens_tensor, token_type_ids=segments_ids)
            predicted_index = torch.argmax(outputs[0][0, index]).item()
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
            if predicted_token != reference_text:
                count += 1
            output_text[index] = predicted_token
            if count >= num:
                break
    return ''.join(output_text)

# keeping max_len
def truncate_tokens(tokens_a, tokens_b, max_len):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

# 模型: bert 预训练 + transformer + generative
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Pre_trained(nn.Module):
    def __init__(self, model=BertModel.from_pretrained(pretrained_weights)):
        super().__init__()
        self.model = model
        for p in self.parameters():
            p.requires_grad=False
    def forward(self, input_ids, segment_ids):
        input_ids = torch.tensor(input_ids).to(device)
        segment_ids = torch.tensor(segment_ids).to(device)
        self.model.eval()
        with torch.no_grad():
            hidden_states, _ = self.model(input_ids, token_type_ids=segment_ids)
        return hidden_states

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, dim=768, drop=0.1, heads=12):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
        self.scores = None
        self.n_heads = heads
    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = self.merge_last(h, 2)
        self.scores = scores
        return h
    def split_last(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)
    def merge_last(self, x, n_dims):
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim=768, ffn=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*ffn)
        self.fc2 = nn.Linear(dim*ffn, dim)
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))

class BertLayer(nn.Module):
    def __init__(self, share='none', norm='pre', dim=768, eps=1e-12, drop=0.1, n_layers=4):
        super(BertLayer, self).__init__()
        self.share = share
        self.norm_pos = norm
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        if self.share == 'ffn':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention() for _ in range(n_layers)])
            self.proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
            self.feedforward = PositionWiseFeedForward()
        elif self.share == 'att':
            self.attention = MultiHeadedSelfAttention()
            self.proj = nn.Linear(dim, dim)
            self.feedforward = nn.ModuleList([PositionWiseFeedForward() for _ in range(n_layers)])
        elif self.share == 'all':
            self.attention = MultiHeadedSelfAttention()
            self.proj = nn.Linear(dim, dim)
            self.feedforward = PositionWiseFeedForward()
        elif self.share == 'none':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention() for _ in range(n_layers)])
            self.proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
            self.feedforward = nn.ModuleList([PositionWiseFeedForward() for _ in range(n_layers)])
    def forward(self, hidden_states, attention_mask, layer_num):
        attention_mask = torch.tensor(attention_mask).to(device)
        if self.norm_pos == 'pre':
            if isinstance(self.attention, nn.ModuleList):
                h = self.proj[layer_num](self.attention[layer_num](self.norm1(hidden_states), attention_mask))
            else:
                h = self.proj(self.attention(self.norm1(hidden_states), attention_mask))
            out = hidden_states + self.drop1(h)
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](self.norm1(out))
            else:
                h = self.feedforward(self.norm1(out))
            out = out + self.drop2(h)
        if self.norm_pos == 'post':
            if isinstance(self.attention, nn.ModuleList):
                h = self.proj[layer_num](self.attention[layer_num](hidden_states, attention_mask))
            else:
                h = self.proj(self.attention(hidden_states, attention_mask))
            out = self.norm1(hidden_states + self.drop1(h))
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](out)
            else:
                h = self.feedforward(out)
            out = self.norm2(out + self.drop2(h))
        return out

class Final_model(nn.Module):
    def __init__(self, n_layers=4, dim=768, eps=1e-12, n_vocab=len(char2id)):
        super().__init__()
        self.pre_trained = Pre_trained()
        self.n_layers = n_layers
        self.blocks = BertLayer()
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.decoder = nn.Linear(dim, n_vocab)
    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.pre_trained(input_ids, segment_ids)
        for i in range(self.n_layers):
            h = self.blocks(h, input_mask, i)
        masked_pos = torch.tensor(masked_pos)[:, :, None].expand(-1, -1, h.size(-1)).to(device)
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.decoder(self.norm(gelu(self.fc2(h_masked))))
        return h_masked

# 训练
def epoch_train(model, iterator, optimizer, epoch, max_len, miu=4, clip=True): #词汇级
    samp = miu/(miu-1+math.exp(epoch/miu))
    print('teacher force rate: %3.3f'%samp)
    model.train()
    epoch_loss, count = 0, 0
    iter_bar = tqdm(iterator, desc='Training')
    for _, batch in enumerate(iter_bar):         # in a batch
        tokens, temp_tokens, segment_ids, input_mask, masked_pos, masked_tokens, resp_len = zip(*batch)
        tokens, masked_tokens = torch.tensor(tokens).to(device), torch.tensor(masked_tokens).to(device)
        for _ in range(min(max(resp_len), max_len//2-1)):         # in a sequence
            optimizer.zero_grad()
            output = model(temp_tokens, segment_ids, input_mask, masked_pos)
            loss = nn.CrossEntropyLoss(reduction='none')(output.transpose(1, 2), masked_tokens)
            loss = loss.mean()
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count += 1
            epoch_loss += loss.item()
            iter_bar.set_description('loss=%3.3f'%loss.item())
            temp_tokens, segment_ids, input_mask, masked_pos = list(temp_tokens), list(segment_ids), list(input_mask), list(masked_pos)
            if max(masked_pos)[0] == max_len - 1:
                break
            if random.random() < samp:
                for i in range(len(resp_len)):
                    temp_tokens[i][masked_pos[i][0]] = int(tokens[i][masked_pos[i][0]])
                    segment_ids[i][masked_pos[i][0]+1] = 1
                    input_mask[i][masked_pos[i][0]+1] = 1
                    masked_pos[i][0] += 1
                    masked_tokens[i][0] = id2id(int(tokens[i][masked_pos[i][0]]), mode='bert2answ')
            else:
                model.eval()
                with torch.no_grad():
                    pred = model(temp_tokens, segment_ids, input_mask, masked_pos)
                model.train()
                out = np.argsort(pred.cpu().detach().numpy())
                out_list = []
                for i in range(len(out)):
                    out_list.append(id2id(int(out[i][0][-1]), mode='answ2bert'))
                for i in range(len(resp_len)):
                    temp_tokens[i][masked_pos[i][0]] = out_list[i]
                    segment_ids[i][masked_pos[i][0]+1] = 1
                    input_mask[i][masked_pos[i][0]+1] = 1
                    masked_pos[i][0] += 1
                    masked_tokens[i][0] = id2id(int(tokens[i][masked_pos[i][0]]), mode='bert2answ')
            temp_tokens, segment_ids, input_mask, masked_pos = tuple(temp_tokens), tuple(segment_ids), tuple(input_mask), tuple(masked_pos)
    return epoch_loss / count

def epoch_valid(model, iterator, max_len):
    model.eval()
    epoch_loss, count = 0, 0
    with torch.no_grad():
        iter_bar = tqdm(iterator, desc='Validation')
        for _, batch in enumerate(iter_bar):
            tokens, temp_tokens, segment_ids, input_mask, masked_pos, masked_tokens, resp_len = zip(*batch)
            tokens, masked_tokens = torch.tensor(tokens).to(device), torch.tensor(masked_tokens).to(device)
            for _ in range(min(max(resp_len), max_len//2-1)):
                output = model(temp_tokens, segment_ids, input_mask, masked_pos)
                loss = nn.CrossEntropyLoss(reduction='none')(output.transpose(1, 2), masked_tokens)
                loss = loss.mean()
                count += 1
                epoch_loss += loss.item()
                iter_bar.set_description('loss=%3.3f'%loss.item())
                temp_tokens, segment_ids, input_mask, masked_pos = list(temp_tokens), list(segment_ids), list(input_mask), list(masked_pos)
                if max(masked_pos)[0] == max_len - 1:
                    break
                out = np.argsort(output.cpu().detach().numpy())
                out_list = []
                for i in range(len(out)):
                    out_list.append(id2id(int(out[i][0][-1]), mode='answ2bert'))
                for i in range(len(resp_len)):
                    temp_tokens[i][masked_pos[i][0]] = out_list[i]
                    segment_ids[i][masked_pos[i][0]+1] = 1
                    input_mask[i][masked_pos[i][0]+1] = 1
                    masked_pos[i][0] += 1
                    masked_tokens[i][0] = id2id(int(tokens[i][masked_pos[i][0]]), mode='bert2answ')
                temp_tokens, segment_ids, input_mask, masked_pos = tuple(temp_tokens), tuple(segment_ids), tuple(input_mask), tuple(masked_pos)
    return epoch_loss / count

# BEAM Search
import copy
import heapq
from gensim.summarization import bm25
bm25_ques = bm25.BM25(questions)
bm25_answ = bm25.BM25(answers)

def epoch_test(ques, model_1, max_len, beam=3):  # str list
    ques = pre_process(ques, punc)
    temp_tokens, segment_ids, input_mask, masked_pos, answers = [], [], [], [], []
    for i in range(len(ques)):
        token = tokenizer.encode(ques[i])[:max_len//2-1] + token_id(['[SEP]'])
        num = len(token)
        ids = [0]*(num) + [1]
        mask = [1]*(num+1)
        n_pad = max_len - num - 1
        token.extend([0]*(max_len - len(token)))
        ids.extend([0]*n_pad)
        mask.extend([0]*n_pad)
        for _ in range(beam):
            temp_tokens.append(copy.deepcopy(token))
            segment_ids.append(ids)
            input_mask.append(mask)
            masked_pos.append([num])
    model_1.eval()
    with torch.no_grad():
        for _ in range(max_len//2-1):
            if max(masked_pos)[0] == max_len - 1 or min(lists.count(token_id(['[SEP]'])[0]) for lists in temp_tokens) >= 2:
                    break
            temp_tokens, segment_ids, input_mask, masked_pos = tuple(temp_tokens), tuple(segment_ids), tuple(input_mask), tuple(masked_pos)
            output = model_1(temp_tokens, segment_ids, input_mask, masked_pos)
            temp_tokens, segment_ids, input_mask, masked_pos = list(temp_tokens), list(segment_ids), list(input_mask), list(masked_pos)
            out = np.argsort(output.cpu().detach().numpy())
            scores = [0]*len(temp_tokens)
            k_tokens, k_scores = [], []
            for i in range(len(temp_tokens)):
                for j in range(beam):
                    k_tokens.append(id2id(int(out[i][0][-1-j]), mode='answ2bert'))
                    k_scores.append(F.softmax(output.cpu().detach(), dim=-1).numpy()[i][0][int(out[i][0][-1-j])])
            for i in range(0,len(k_tokens),beam*beam):
                temp_list = [(score, token) for score, token in zip(k_scores[i:i+beam*beam], k_tokens[i:i+beam*beam])]
                temp_list.sort(key = lambda i: i[0], reverse=True)
                k_scores[i:i+beam*beam] = [score for score, token in temp_list]
                k_tokens[i:i+beam*beam] = [token for score, token in temp_list]
            for i in range(len(scores)):
                count = 0
                if i % beam != 0:
                    if scores[i] + k_scores[i//beam*beam*beam+i%beam+count] == scores[i-1]:
                        count += 1
                        scores[i] += k_scores[i//beam*beam*beam+i%beam+count]
                        temp_tokens[i][masked_pos[i][0]] = k_tokens[i//beam*beam*beam+i%beam+count]
                    else:
                        scores[i] += k_scores[i//beam*beam*beam+i%beam+count]
                        temp_tokens[i][masked_pos[i][0]] = k_tokens[i//beam*beam*beam+i%beam+count]
                else:
                    scores[i] += k_scores[i*beam]
                    temp_tokens[i][masked_pos[i][0]] = k_tokens[i*beam]
                segment_ids[i][masked_pos[i][0]+1] = 1
                input_mask[i][masked_pos[i][0]+1] = 1
                masked_pos[i][0] += 1
    for i in range(len(ques)):
        for j in range(beam):
            temp_tokens[i*beam+j] = tokenizer.convert_ids_to_tokens(temp_tokens[i*beam+j])
            start = temp_tokens[i*beam+j].index('[SEP]')
            temp_tokens[i*beam+j] = temp_tokens[i*beam+j][start+1:]
            if '[SEP]' in temp_tokens[i*beam+j]:
                end = temp_tokens[i*beam+j].index('[SEP]')
                temp_tokens[i*beam+j] = temp_tokens[i*beam+j][:end]
            while '[PAD]' in temp_tokens[i*beam+j]:
                temp_tokens[i*beam+j].remove('[PAD]')
            while '[UNK]' in temp_tokens[i*beam+j]:
                temp_tokens[i*beam+j].remove('[UNK]')
            temp_tokens[i*beam+j] = ''.join(temp_tokens[i*beam+j])
        answers.append(bm25(bm25_ques, bm25_answ, ques[i], temp_tokens[i*beam:(i+1)*beam]))
    return answers

def bm25(bm25_ques, bm25_answ, question, answers, k=4):
    ques_scores = bm25_ques.get_scores(question)
    ques_max_k = heapq.nlargest(k, ques_scores)
    scores, indexes = [], []
    for i in range(len(ques_scores)):
        if ques_scores[i] in ques_max_k:
            indexes.append(i)
    for i in range(len(answers)):
        temp_score = 0
        answ_scores = bm25_answ.get_scores(answers[i])
        for index in indexes:
            temp_score += ques_scores[index] * answ_scores[index]
        scores.append(temp_score)
    return answers[scores.index(max(scores))]

import os
def model_train(model, mask_model, ques_t, answ_t, test_ques, batch_size, max_len, learning_rate, epochs, load=False):
    log_file = '/home/'+ser+'/STC3/result/log_anger.txt'
    out_file = '/home/'+ser+'/STC3/result/out_anger.txt'
    if load == True:
        load_model(model, '/home/'+ser+'/STC3/result/7.844.pt')
        start = 5
    else:
        with open(log_file, 'w') as log_f:
            log_f.write('epoch, train_loss, valid_loss\n')
        with open(out_file, 'w') as out_f:
            out_f.write(str(test_ques) + '\n')
        start = 0
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)
    stop = 0
    loss_list = []
    for epoch in range(start, epochs):
        r = random.randint(0,len(ques_t)-VALID)
        train_iterator = data_loader(ques_t,answ_t, batch_size, max_len, mask_model)
        valid_iterator = data_loader(ques_t[r:r+VALID],answ_t[r:r+VALID], batch_size, max_len, mask_model)
        print('Epoch: ' + str(epoch+1))
        train_loss = epoch_train(model, train_iterator, optimizer, epoch, max_len)
        valid_loss = epoch_valid(model, valid_iterator, max_len)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss)
        with open(log_file, 'a') as log_f:
            log_f.write('{epoch},{train_loss: 3.3f},{valid_loss: 3.3f}\n'.format(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss))
        if valid_loss == min(loss_list):
            stop = 0
            with open(out_file, 'a') as out_f:
                out_f.write(str(valid_loss)[:5] + '\n')
                out_f.write(str(epoch_test(test_ques, model, 64)) + '\n')
            torch.save(model.state_dict(), os.path.join('/home/'+ser+'/STC3/result/', str(valid_loss)[:5]+'.pt'))
        else:
            stop += 1
            if stop > 5:       # patience**2+1
                break

def load_model(model, model_file):
    _model = model
    state_dict = torch.load(model_file)
    _model.load_state_dict(state_dict)
    return _model

# 导出test 问句
import json
test_ques = []
with open('/home/'+ser+'/STC3/result/TUA1_1_TokushimaUniversity_base.json', 'r') as f:
    for line in f:
        a = json.loads(line)
for i in range(40):
    test_ques.append(a[i][0][0])

VALID = 16384
model_train(Final_model().to(device), mask_model, questions, answers, test_ques, 256, 64, 0.0001, 999, load=True)
