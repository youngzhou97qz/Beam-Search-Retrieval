import heapq
from gensim.summarization import bm25
# q1 = '你是谁'
# a1 = '我是学生'
# a2 = '你是谁'
q1 = '帮我把剪刀拿来'
a1 = '吃饭的时候小心天气'
a2 = '好哒'
# a2 = '帮我把剪刀拿来'
k = 2
example = []
bm25_ques = bm25.BM25(ques)
bm25_answ = bm25.BM25(answ)

# example = []
ques_scores = bm25_ques.get_scores(q1)
ques_maxk = heapq.nlargest(k, ques_scores)
for i in range(len(ques_scores)):
    if ques_scores[i] in ques_maxk:
        example.append(i)

a1_score = 0
answ_scores = bm25_answ.get_scores(a1)
for index in example:
    a1_score += ques_scores[index] * answ_scores[index]

a2_score = 0
answ_scores = bm25_answ.get_scores(a2)
for index in example:
    a2_score += ques_scores[index] * answ_scores[index]

# print(example)
print(a1_score)
print(a2_score)