#  import
import os
import json
import random
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.layers import Input
from keras.models import Model
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import *
pretrained_weights = 'bert-base-chinese'
pre_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)      # tokenizer.vocab_size = 21128

#  parameters
max_len = 64
batch_size = 64
epochs = 99999
ques_rate = 0.1
answ_rate = 0.1
ser = 'dango'

#  setting vocabulary
config_path = '/home/'+ser+'/STC3/code/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/'+ser+'/STC3/code/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/'+ser+'/STC3/code/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

token_dict, keep_tokens = load_vocab(dict_path=dict_path, simplified=True,
    startswith=['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[PAD]', '[UNK]', '[CLS]', '[SEP]'],)
#  [unused1]: Other, [unused2]: Like, [unused3]: Sadness, [unused4]: Disgust, [unused5]: Anger, [unused6]: Happiness
tokenizer = Tokenizer(token_dict, do_lower_case=True)

#  reading data
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

#  judging chinese
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

#  delete sentences
i = len(questions)-1
while i >= 0:
    if check_contain_chinese(questions[i])==False or check_contain_chinese(answers[i])==False or len(questions[i])==0 or len(answers[i])==0:
        questions.pop(i)
        answers.pop(i)
        answer_ids.pop(i)
    i -= 1
print('问答对：', len(questions))    #   1630292  anger: 184590

#  standardization
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
    for i in tqdm(range(len(text)), desc='Data processing'):
        text[i] = pre_tokenizer.tokenize(text[i])
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

#  data preparing
train_data = []
for i in tqdm(range(len(questions)), desc='Data preparing'):
    train_data.append((questions[i], answers[i], answer_ids[i]))
del questions, answers, answer_ids

class data_generator(DataGenerator):
    def __iter__(self, random=True):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_o_token_ids = [], [], []
        for i in idxs:
            ques, answ, ids = self.data[i]
            token_ids, segment_ids = tokenizer.encode(ques, answ, max_length=max_len)
            o_token_ids = token_ids
            token_ids[1:-1] = [t if (s == 0 and np.random.random() > ques_rate) or (s == 1 and np.random.random() > answ_rate)\
                         else np.random.choice(token_ids) for t, s in zip(token_ids[1:-1], segment_ids[1:-1])]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_o_token_ids.append(o_token_ids)
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_o_token_ids = sequence_padding(batch_o_token_ids)
                yield [batch_token_ids, batch_segment_ids, batch_o_token_ids], None
                batch_token_ids, batch_segment_ids, batch_o_token_ids = [], [], []

train_generator = data_generator(train_data, batch_size)

#  问句替换比例，答句替换比例调参
model = build_transformer_model(config_path, checkpoint_path, application='unilm', keep_tokens=keep_tokens)
o_in = Input(shape=(None, ))
train_model = Model(model.inputs + [o_in], model.outputs + [o_in])

# 交叉熵作为loss，并mask掉输入部分的预测
y_true = train_model.input[2][:, 1:]  # 目标tokens
y_mask = train_model.input[1][:, 1:]
y_pred = train_model.output[0][:, :-1]  # 预测tokens，预测与目标错开一位
cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

train_model.add_loss(cross_entropy)
train_model.compile(optimizer=Adam(1e-5))

class AutoTitle(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]
    def generate(self, text, emotion, topk=2):
        max_c_len = max_len - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, max_length=max_c_len)
        token_ids[0] = emotion
        output_ids = self.beam_search([token_ids, segment_ids], topk)  # beam search
        return tokenizer.decode(output_ids)

autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)

test_ques = []
with open('/home/'+ser+'/STC3/result/TUA1_1_TokushimaUniversity_base.json', 'r') as f:
    for line in f:
        a = json.loads(line)
for i in range(200):
    test_ques.append(a[i][0][0])

def just_show(lowest):
    print(u'希望九哥日日开心同我打羽毛波: ', autotitle.generate(u'希望九哥日日开心同我打羽毛波', 0))
    out_file = os.path.join('/home/'+ser+'/STC3/result/', str(lowest)[:5]+'.txt')
    with open(out_file, 'w') as out_f:
        for i in tqdm(range(len(test_ques)), desc='Response generating'):
            out_f.write(str(test_ques[i]) + '\n')
            out_f.write(str(autotitle.generate(str(test_ques[i]), 1)) + '\n')
            out_f.write(str(autotitle.generate(str(test_ques[i]), 2)) + '\n')
            out_f.write(str(autotitle.generate(str(test_ques[i]), 3)) + '\n')
            out_f.write(str(autotitle.generate(str(test_ques[i]), 4)) + '\n')
            out_f.write(str(autotitle.generate(str(test_ques[i]), 5)) + '\n')
            out_f.write('\n')

class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 3.5
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.
    def on_epoch_end(self, epoch, logs=None):
        rad = random.randint(0,len(train_data)-1000)
        metrics = self.evaluate(train_data[rad:rad+1000])  # 评测模型
        if metrics['bleu'] > self.best_bleu and logs['loss'] <= self.lowest:
            self.best_bleu = metrics['bleu']
            self.lowest = logs['loss']
            model.save_weights(os.path.join('/home/'+ser+'/STC3/result/', str(self.lowest)[:5]+'.weights'))  # 保存最优
            metrics['best_bleu'] = self.best_bleu
            print('valid_data:', metrics)
            just_show(self.lowest)  # 演示效果
    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for ques, answ, ids in tqdm(data):
            total += 1
            answ = ''.join(answ.split(' '))
            pred_answ = ''.join(autotitle.generate(ques, ids))
            scores = self.rouge.get_scores(hyps=pred_answ, refs=answ)
            rouge_1 += scores[0]['rouge-1']['f']
            rouge_2 += scores[0]['rouge-2']['f']
            rouge_l += scores[0]['rouge-l']['f']
            bleu += sentence_bleu(references=[list(answ)], hypothesis=list(pred_answ), smoothing_function=self.smooth)
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l': rouge_l, 'bleu': bleu,}

evaluator = Evaluate()
train_generator = data_generator(train_data, batch_size)

train_model.fit_generator(train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=epochs, callbacks=[evaluator])
