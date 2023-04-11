import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from konlpy.tag import Komoran
import matplotlib.pyplot as plt

#데이터 불러오기
purpose = pd.read_csv("Pretreatment/training_data/custom/purpose_talk_data.csv")
topic = pd.read_csv("Pretreatment/training_data/daily_talk_data.csv")
common_sense = pd.read_csv("Pretreatment/training_data/common_sense_data.csv")
movie = pd.read_csv("Pretreatment/training_data/custom/m_rating_data.csv")
add = pd.read_csv("Pretreatment/training_data/add.csv")

purpose.dropna(inplace=True)
topic.dropna(inplace=True)
movie.dropna(inplace=True)
common_sense.dropna(inplace=True)
add.dropna(inplace=True)

all_data =  list(purpose['text']) + list(topic['text']) + list(common_sense['query']) + list(movie['document']) \
+ list(common_sense['answer']) + list(add['query'])

# 통합본 생성하고 저장하기
total = pd.DataFrame({'text': all_data})
total.to_csv("Pretreatment/training_data/custom/total_data.csv", index=False)

#의도 분류 데이터 생성
number = []
place = []
time = []
etc = []

number_label = []
for _ in range(len(number)):
    number_label.append(0)

place_label = []
for _ in range(len(place)):
    place_label.append(1)

time_label = []
for _ in range(len(time)):
    time_label.append(2)

train_df = pd.DataFrame({'text':number+place+time,
                         'label':number_label+place_label+time_label})

train_df.reset_index(drop=True, inplace=True)
train_df.to_csv("training_data.csv", index=False)

# 토크나이저 - 올바른 패딩

data = pd.read_csv('training_data.csv')
tokenizer = Komoran()

ata_tokenized = [[token+"/"+POS for token, POS in tokenizer.pos(text_)] for text_ in data['text']]

exclusion_tags = [
    'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
            'JX', 'JC',
            'SF', 'SP', 'SS', 'SE', 'SO',
            'EP', 'EF', 'EC', 'ETN', 'ETM',
            'XSN', 'XSV', 'XSA'
]

f = lambda x: x in exclusion_tags

data_list = []
for i in range(len(data_tokenized)):
        temp = []
        for j in range(len(data_tokenized[i])):
            if f(data_tokenized[i][j].split('/')[1]) is False:
                temp.append(data_tokenized[i][j].split('/')[0])
        data_list.append(temp)

num_tokens = [len(tokens) for tokens in data_list]
num_tokens = np.array(num_tokens)

# 평균값, 최댓값, 표준편차
print(f"토큰 길이 평균: {np.mean(num_tokens)}")
print(f"토큰 길이 최대: {np.max(num_tokens)}")
print(f"토큰 길이 표준편차: {np.std(num_tokens)}")

plt.title('all text length')
plt.hist(num_tokens, bins=100)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

select_length = 25

def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1
        
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))))

