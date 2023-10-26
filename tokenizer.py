
# text='我是一个句子'
# text2='I am a sentence......;;;'


# unicodetext1=text.encode('unicode_escape').decode()
# unicodetext2=text2.encode('unicode_escape').decode()
# print(unicodetext1)
# print(unicodetext2)


'''
参考https://github.com/soaxelbrooke/python-bpe/blob/master/bpe/encoder.py的代码进行修改

'''

from collections import Counter
from collections import defaultdict
from tqdm import tqdm
import toolz
import re
import matplotlib.pyplot as plt
import json
import os
# a='yuasdbiasjfi,ojaoijqpoqkfnurnwofqejojd           qowpdqwdqw妮妮你你你'
# c=Counter(tqdm(a))
# c.most_common()
# print(c)




# def is_chinese(word):
#     ch_p=re.compile(r'[\u4e00-\u9fa5]')
#     return bool(ch_p.search(text))


def word_tokenizer(text):
    '''分词函数，提取文本里面的单词，标点，和中文字，空格舍弃'''
    # text = "Hello, this is a example. 接下来是另一段句子"
    words=[]
    temp=''
    for i in range(len(text)):
        if 'a'<=text[i]<='z' or 'A'<=text[i]<='Z':
            temp=temp+text[i]
        else:
            if temp != '':
                words.append(temp)
            if text[i] !=' ' :
                words.append(text[i])
            temp=''
        if i==len(text)-1:
            if temp:
                words.append(temp)
    # print(words)
    return words





DEFAULT_EOW = '__eow'
DEFAULT_SOW = '__sow'
DEFAULT_UNK = '__unk'
DEFAULT_PAD = '__pad'


class BPE():
    def __init__(self,vocab_size=8192,silent=True,required_tokens=None,
                 EOW=DEFAULT_EOW, SOW=DEFAULT_SOW, UNK=DEFAULT_UNK, PAD=DEFAULT_PAD):
        if vocab_size <1: 
            raise ValueError('vocab size must be greater than 0')
        self.EOW = EOW
        self.SOW = SOW
        self.UNK = UNK
        self.PAD = PAD
        self._progress_bar=iter if silent else tqdm#iter是python的内置函数，用来创建可迭代对象
        self.word_tokenizer=word_tokenizer
        self.alphabet_table={}#词表，这里会看到我们已经合并了的token
        self.token_table=defaultdict(int)#token表，英文来说就是字母，中文来说就是单个字符
        self.required_tokens = list(set(required_tokens or []).union({self.UNK, self.PAD}))
    def mute(self):
        """ Turn on silent mode """
        self._progress_bar = iter

    def unmute(self):
        """ Turn off silent mode """
        self._progress_bar = tqdm

    def count_token(self,words):#数总的单词数(token)
        #看源码的时候，没有看到对文本进行</w>的处理，而且Counter返回单个字符，不能直接把这个函数的返回结果拿去bpe函数处理
        #我们想要的应该是数每个单词的出现次数
        # token_counts=Counter(self._progress_bar(words))
        words=word_tokenizer(words)

        vocab=defaultdict(int)#借助collection模块创建一个字典

        for word in words :
            vocab[' '.join(list(word))+' </w>']+=1#每个单词或者中文字添加结束符，计算出现次数
            for token in word:
                self.token_table[token]+=1# 如果键存在，不改变值；如果不存在，初始化为 0
            self.token_table['</w>']+=1
        return vocab


    def byte_pair_counts(self,vocab):#数字符对
        """ Counts space separated token character pairs:
            [('T h i s </w>', 4}] -> {'Th': 4, 'hi': 4, 'is': 4}
        """
        pairs=defaultdict(int)
        for word, freq in vocab.items():
            symbols=word.split()#这里word是类似w o r d </w>,中间已经加入了空格
            for i in range(len(symbols)-1):#数字符对i和i+1，i的范围要比长度少1
                #我们针对每个单词，开始数里面连续出现的词对的次数，中文字符的话只有单个字符，所以执行一次就退出这个循环
                pairs[symbols[i],symbols[i+1]]+=freq#创建字符对，计算字符对出现次数
        return pairs

    def merge_vocab(self,words,merge_num):#开始合并出现次数最多的字符对，这个函数执行一次合并一个最频繁字符对
        self.alphabet_table=self.count_token(words)#数单词
        x_for_plot=[]
        y_for_plot=[]
        for i in tqdm(range(merge_num)):
            self.pair=self.byte_pair_counts(self.alphabet_table)#数字符对
            if not self.pair:#如果pair是空的
                print('合并次数已到最大')
                break
            v_out={}
            pair_max=max(self.pair,key=self.pair.get)#key参数传进一个方法，指定比较的内容，在这里比较键值，返回的是键
            bigram_to_sub=re.escape(' '.join(pair_max))
            '''
            下面这一行运用了正则表达式的代码让没怎么用过正则表达式的我非常困惑
            (?<!)表示负向先行断言，\S表示的是任何非空白字符，表示我们要匹配的字符对，前面不能是除了空白以外的字符
            同理(?!)是负向后行断言，表示后面不能是除了空白以外的字符。
            这么写的目的是我们多次进行bpe操作的时候，可能会出现前面的字符对结尾字母，和后面的字符对开头字母构成了和我们查找中的相同字符对。
            比如：
            gg=['a p b e d',
                'a c dbe d ',
                'a c be ',
                'a b obeo ',
                'ab ecd']
            bigram=re.escape(' '.join(['b','e']))
            p = re.compile(r'(?<!\S)' + bigram +r'(?!\S)')
            for word in gg: 
            print(p.findall(word))
            '''
            p = re.compile(r'(?<!\S)' + bigram_to_sub + r'(?!\S)')
            for word ,fre in self.alphabet_table.items():
                w_out=p.sub(''.join(pair_max),word)#用参数1替换参数2里面匹配到的字符对,返回替换之后的字符串
                v_out[w_out]=self.alphabet_table[word]#把单词出现次数复制过来,v_out就是下一次的self.alphabet_table
                # self.token_table.setdefault(''.join(pair_max),0)
                # self.token_table[''.join(pair_max)]+=len(p.findall(word))*fre
                self.token_table[''.join(pair_max)]+=len(p.findall(word))*fre
                try:
                    if self.token_table[pair_max[0]]>len(p.findall(word))*fre:
                        # print(pair_max[0])
                        # print(f'匹配到次数:{len(p.findall(word))*fre}')
                        # print(f'词表内次数：{self.token_table[pair_max[0]]}')
                        self.token_table[pair_max[0]]-=len(p.findall(word))*fre
                    else:
                        del self.token_table[pair_max[0]]#进入删除就表示剩下的word里面不会包含pair_max元组了
                except:
                    pass
                try:
                    if self.token_table[pair_max[1]]>len(p.findall(word))*fre:
                        self.token_table[pair_max[1]]-=len(p.findall(word))*fre
                    else:
                        del self.token_table[pair_max[1]]
                except:
                    pass
            y_for_plot.append(len(self.token_table))
            x_for_plot.append(i)
            self.alphabet_table=v_out

        # temp=sorted(self.alphabet_table.items(),key=lambda item:-item[1])
        # print(temp)
        plt.figure()
        plt.title('Num of token')
        plt.grid(True)
        plt.plot(x_for_plot,y_for_plot)
        plt.show()
        #画图，观察token数量的变量，找到最合适的
        return v_out#返回更新之后的词表



class tokenizer_ch():
    def __init__(self):
        self.encoder={}
        self.decoder={}

    def creat_enco_deco(self,text):
        # with open(path,'r') as f:
        #     text=f.read()
        chars=['<PAD>']+['<SOW>']+['<EOW>']+['<UNK>']+list(sorted(set(text)))
        # chars=list(sorted(set(text)))

        self.encoder={ch:i for i,ch in enumerate(chars)}
        self.decoder={i:ch for i,ch in enumerate(chars)}
        print(len(self.encoder))

    def encode(self,text):#对每一段文字预训练语录进行编码，在段落结束要添加结束标记符，用list方式一段段进行编码
        # return [self.encoder[c] for c in text]+[self.encoder['<EOW>']]
        return [self.encoder[c] for c in text]

    def decode(self,label):
        return ''.join([self.decoder[i] for i in label])
    
    def __len__(self):
        return len(self.decoder)

    def save_tokenizer(self,filename='TestTokenizer.json',path=None):
        current_dir=os.path.dirname(os.path.abspath(__file__))
        filepath=os.path.join(current_dir,filename)
        if os.path.exists(filepath):
            print("覆盖原有的Tokenizer！！！")
        with open(filepath,'w') as f:
            json.dump({'encoder':self.encoder,'decoder':self.decoder},f,default=str)









if __name__ =='__main__':
    # print(os.path.dirname(os.path.abspath(__file__)))
    bpe=BPE()
    ch_tokenizer=tokenizer_ch()
    with open(r'/home/yuzhaohao/LanguageModel/yzh_model/wikipedia-cn-20230720-filtered.json','r',encoding='utf-8') as f:
    #     text=f.read().split()
    #     # print(text)
    #     print(len(text))
    #     print(len(set(text)))
    #     print(len(''.join(text)))
        data=json.load(f)
        print(data[:2])
    print(len(data))
    data=[c['completion']  for c in data]
    print(max([len(s) for s in data]))
    print(min([len(s) for s in data]))





    # print(data[:2])
    print(type(data))
    print(len(data))
    data=''.join(data)
    print(data[:500])
    print(type(data))
    ch_tokenizer.creat_enco_deco(data)
    ch_tokenizer.save_tokenizer()
    print(ch_tokenizer.decode(ch_tokenizer.encode('你好')))
    # bpe.merge_vocab(text,1000)