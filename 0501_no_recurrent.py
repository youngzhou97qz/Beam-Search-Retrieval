# import
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
from apex import amp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_weights = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)      # tokenizer.vocab_size = 21128
token_id = tokenizer.convert_tokens_to_ids
token_to = tokenizer.convert_ids_to_tokens
# mask_model = BertForMaskedLM.from_pretrained('bert-base-chinese').to(device)
# mask_model.eval()
ser = 'yzhou'
MAX_LEN = 66

# data
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
    if check_contain_chinese(questions[i])==False or check_contain_chinese(answers[i])==False or len(questions[i])==0 or len(answers[i])==0:
        questions.pop(i)
        answers.pop(i)
        answer_ids.pop(i)
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
# print(questions[:5])
# print(answers[:5])

# QUES_RATE = 0.1
# ANSW_RATE = 0.4

# using BERT to replace characters
def prediction_replace(sentence, model, rate, max_len):
    output_text = tokenizer.tokenize(sentence)
    num = int(len(output_text)//((max_len-1)/((max_len-1)*rate+1)))
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
            predicted_token = token_to([predicted_index])[0]
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

def data_loader(ques, answ, ids, batch_size, max_len, model=None):
    state = np.random.get_state()
    np.random.shuffle(ques)
    np.random.set_state(state)
    np.random.shuffle(answ)
    np.random.set_state(state)
    np.random.shuffle(ids)
    batch = []
    for i in range(len(questions)):
        part1 = tokenizer.encode(questions[i])
        part2 = tokenizer.encode(answers[i])  # 先不改！！！！！！
        truncate_tokens(part1, part2, max_len-4)
        tokens = [answer_ids[i]] + token_id(['[SEP]']) + part1 + token_id(['[SEP]']) + part2 + token_id(['[SEP]'])
        for j in range(len(part2)+1):
            num = len(part1)+3+j
            temp_tokens = tokens[:num]
            segment_ids = [0]*2 + [1]*(len(part1)+1) + [2]*(1+j)
            input_mask = [1]*(num+1)
            masked_tokens = [tokens[num]]
            masked_pos = [num]
            n_pad = max_len - num - 1
            temp_tokens.extend([0]*(n_pad+1))
            segment_ids.extend([0]*n_pad)
            input_mask.extend([0]*n_pad)
            batch.append((temp_tokens, segment_ids, input_mask, masked_pos, masked_tokens))
            if len(batch) == batch_size:
                yield batch
                batch = []
                
# load = data_loader(questions[:2], answers[:2], answer_ids[:2], batch_size=2, max_len=MAX_LEN)
# aa = next(load)
# print(aa)

# train
VOCAB = 21128
EMB = 128
DIM = 312
FFN = 4
N_LAY = 4
HEAD = 12
DROP = 0.0
EPS = 1e-12

class Embeddings(nn.Module):
    def __init__(self, vocab_size=VOCAB, emb_size=EMB, dim=DIM, max_len=MAX_LEN, eps=EPS, drop=DROP):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.word_embeddings_2 = nn.Linear(emb_size, dim, bias=False)
        self.position_embeddings = nn.Embedding(max_len, dim)
        self.type_embeddings = nn.Embedding(3, dim)
        self.LayerNorm = nn.LayerNorm(dim, eps=eps)
        self.dropout = nn.Dropout(drop)
        self.len = max_len
    def forward(self, input_ids, segment_ids, position_ids=None):
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = torch.arange(self.len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = self.word_embeddings_2(words_embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.type_embeddings(segment_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, dim=DIM, drop=DROP, heads=HEAD):
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

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim=DIM, ffn=FFN):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*ffn)
        self.fc2 = nn.Linear(dim*ffn, dim)
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))

class BertLayer(nn.Module):
    def __init__(self, share='none', norm='pre', dim=DIM, n_layers=N_LAY, eps=EPS, drop=DROP):
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
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
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

class Transformer(nn.Module):
    def __init__(self, n_layers=N_LAY):
        super().__init__()
        self.embed = Embeddings()
        self.blocks = BertLayer()
        self.n_layers = n_layers
    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for i in range(self.n_layers):
            h = self.blocks(h, mask, i)
        return h

class Final_model(nn.Module):
    def __init__(self, dim=DIM, eps=EPS):
        super().__init__()
        self.transformer = Transformer()
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, eps=eps)
        embed_weight = self.transformer.embed.word_embeddings.weight
        n_vocab, embed_dim = embed_weight.size()
        self.decoder = nn.Linear(dim, embed_dim, bias=False)
        self.decoder_2 = nn.Linear(embed_dim, n_vocab, bias=False)
        self.decoder_2.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)
        masked_pos = torch.tensor(masked_pos, dtype=torch.long, device=device)[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(gelu(self.fc2(h_masked)))
        logits_lm = self.decoder_2(self.decoder(h_masked)) + self.decoder_bias
        return logits_lm

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
print('Embeddings: ',get_parameter_number(Embeddings()))
print('MultiHeadedSelfAttention: ',get_parameter_number(MultiHeadedSelfAttention()))
print('PositionWiseFeedForward: ',get_parameter_number(PositionWiseFeedForward()))
print('Final_model: ',get_parameter_number(Final_model()))

# 训练
def epoch_train(model, iterator, optimizer, epoch, max_len, clip=True): #词汇级
    model.train()
    epoch_loss, count = 0, 0
    iter_bar = tqdm(iterator, desc='Training')
    for _, batch in enumerate(iter_bar):         # in a batch
        temp_tokens, segment_ids, input_mask, masked_pos, masked_tokens = zip(*batch)
        masked_tokens = torch.tensor(masked_tokens, dtype=torch.long, device=device)
        optimizer.zero_grad()
        output = model(temp_tokens, segment_ids, input_mask, masked_pos)
        loss = nn.CrossEntropyLoss(reduction='none')(output.transpose(1, 2), masked_tokens)
        loss = loss.mean()
#         loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count += 1
        epoch_loss += loss.item()
        iter_bar.set_description('loss=%3.3f'%loss.item())
    return epoch_loss / count

import os
def model_train(model, ques_t, answ_t, ids_t, test_ques, test_ids, batch_size, max_len, learning_rate, epochs, mask_model, load=False):
    log_file = '/home/'+ser+'/STC3/result/log_0501.txt'
    out_file = '/home/'+ser+'/STC3/result/out_0501.txt'
    if load == True:
        load_model(model, '/home/'+ser+'/STC3/result/4.699.pt')
        start = 22
    else:
        with open(log_file, 'w') as log_f:
            log_f.write('epoch, train_loss\n')
#         with open(out_file, 'w') as out_f:
#             out_f.write(str(test_ques) + '\n')
#             out_f.write(str(answ_t) + '\n')
        start = 0
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)
    stop = 0
    loss_list = []
    for epoch in range(start, epochs):
        train_iterator = data_loader(ques_t, answ_t, ids_t, batch_size, max_len, mask_model)
        print('Epoch: ' + str(epoch+1))
        train_loss = epoch_train(model, train_iterator, optimizer, epoch, max_len)
        scheduler.step(train_loss)
        loss_list.append(train_loss)
        with open(log_file, 'a') as log_f:
            log_f.write('{epoch},{train_loss: 3.3f}\n'.format(epoch=epoch+1, train_loss=train_loss))
        if train_loss == min(loss_list):
            stop = 0
            with open(out_file, 'a') as out_f:
#                 out_f.write(str(train_loss)[:5] + '\n')
#                 out_f.write(str(epoch_test(test_ques, test_ids, model, 66)) + '\n')
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
# import json
# test_ques = []
# with open('/home/'+ser+'/STC3/result/TUA1_1_TokushimaUniversity_base.json', 'r') as f:
#     for line in f:
#         a = json.loads(line)
# for i in range(40):
#     test_ques.append(a[i][0][0])

model_train(Final_model().to(device), questions, answers, answer_ids, questions[:1], answer_ids[:1], 512, 66, 0.0001, 999, mask_model=None)
