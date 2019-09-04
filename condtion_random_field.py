# -*- coding: utf-8 -*-

import math
import datetime

class sentence:
    def __init__(self):
        self.word = []
        self.tag = []
        self.wordchars = []

class dataset:
    def __init__(self):
        self.sentences = []
        self.name = ""
        self.total_word_count = 0
    
    def open_file(self, inputfile):
        self.inputfile = open(inputfile, mode = 'r', encoding='utf-8')
        self.name = inputfile.split('.')[0]

    def close_file(self):
        self.inputfile.close()

    def read_data(self, sentenceLen):
        wordCount = 0
        sentenceCount = 0
        sen = sentence()
        for s in self.inputfile:
            if(s == '\r\n' or s == '\n'):
                sentenceCount += 1
                self.sentences.append(sen)
                sen = sentence()
                if(sentenceLen !=-1 and sentenceCount >= sentenceLen):
                    break
                continue
            list_s = s.split('\t')
            str_word = list_s[1]
            str_tag = list_s[3]
            list_wordchars = list(str_word)
            sen.word.append(str_word)
            sen.tag.append(str_tag)
            sen.wordchars.append(list_wordchars)
            wordCount += 1
        self.total_word_count = wordCount
        print(self.name + ".conll contains " + str(len(self.sentences)) + " sentences")
        print(self.name + ".conll contains " + str(self.total_word_count) + " words")
        
class conditional_random_field:
    def __init__(self):
        self.feature_dict = {}
        self.feature_keys = []
        self.feature_values = []
        self.feature_length = 0
        self.tag_dict = {}
        self.tag_dict_stop = {} # contain "STOP"
        self.tag_length = 0
        self.bigram_feature_id = {} # bigram_feature:id contain "START"
        
        self.g = []
        self.w = []
        self.train = dataset()
        self.dev = dataset()

        self.train.open_file("train.conll")
        self.train.read_data(100)
        self.train.close_file()

        self.dev.open_file("dev.conll")
        self.dev.read_data(100)
        self.dev.close_file()
        
    def create_bigram_feature(self, left_tag):
        f = []
        f.append("01:" + left_tag)
        return f
    
    def create_unigram_feature(self, sen, pos):
        f = []
            
        wi = sen.word[pos]
        f.append("02:" + wi)
        
        if(pos == 0):
            wim1 = "$$"
        else:
            wim1 = sen.word[pos - 1]
        f.append("03:" + wim1)
        
        len_sen = len(sen.word)
        if(pos == len_sen - 1):
            wip1 = "##"
        else:
            wip1 = sen.word[pos + 1]
        f.append("04:" + wip1)
        
        cim1m1 = wim1[-1]
        f.append("05:" + wi + cim1m1)
        
        cip10 = wip1[0]
        f.append("06:" + wi + cip10)
        
        ci0 = wi[0]
        f.append("07:" + ci0)
        
        cim1 = wi[-1]
        f.append("08:" + cim1)
        
        len_str = len(wi)
        for k in range(1, len_str - 1):
            cik = wi[k]
            f.append("09:" + cik)
            f.append("10:" + ci0 + cik)
            f.append("11:" + cim1 + cik)
            
        if(len_str == 1):
            f.append("12:" + wi + cim1m1 + cip10)
            
        for k in range(len_str - 1):
            cik = wi[k]
            cikp1 = wi[k + 1]
            if(cik == cikp1):
                f.append("13:" + cik + "consecutive")
        
        for k in range(1, len_str):
            if k > 4:
                break
            f.append("14:" + wi[:k])
            f.append("15:" + wi[-k:])
        #print(pos, f)
        return f
    
    def create_feature(self, sen, pos, left_tag):
        f = []
        f.extend(self.create_bigram_feature(left_tag))
        f.extend(self.create_unigram_feature(sen, pos))
        return f
            
    def create_feature_space(self):
        for sen in self.train.sentences:
            for pos in range(len(sen.word)):
                if(pos == 0):
                    left_tag = "START"
                else:
                    left_tag = sen.tag[pos]
                f = self.create_feature(sen, pos, left_tag)
                for feature in f:
                    if feature not in self.feature_dict:
                        self.feature_dict[feature] = len(self.feature_dict)
                
                tag = sen.tag[pos]
                if tag not in self.tag_dict:
                    self.tag_dict[tag] = len(self.tag_dict)
                    self.tag_dict_stop[tag] = len(self.tag_dict_stop)
        
        self.tag_dict_stop["STOP"] = len(self.tag_dict_stop)
                    
        self.feature_length = len(self.feature_dict)
        self.tag_length = len(self.tag_dict_stop)
        self.feature_keys = list(self.feature_dict.keys())
        self.feature_values = list(self.feature_dict.values())
        
        self.w = [0]*(self.feature_length * self.tag_length)
        self.g = [0]*(self.feature_length * self.tag_length)
        
        """for tag in self.tag_dict:
            bigram_feature = self.create_bigram_feature(tag)
            bigram_feature_id = self.get_feature_id(bigram_feature)
            self.bigram_feature_id[tag] = bigram_feature_id[0]"""   
        
        print("the total number of features is " + str(self.feature_length))
        print("the total number of tags is " + str(self.tag_length - 1))
        
        #for i in range(100):
        #    print(self.feature_keys[i], self.feature_values[i])
        
    def get_feature_id(self, fv):
        fv_id = []
        for f in fv:
            if f in self.feature_dict:
                fv_id.append(self.feature_dict[f])
            #else:
                #print("not in")
        return fv_id
        
        
    def dot(self, fv_id, offset):
        score = 0
        for f_id in fv_id:
            score += self.w[f_id + offset]
        return score
                
    def log_exp_sum(self, a, b):
        if a > b:
            return a + math.log(1 + (math.exp (b - a)))
        else:
            return b + math.log(1 + (math.exp (a - b)))
        #return math.log(math.exp(a) + math.exp(b)) range question
    
    def forward(self, sen):
        forward_score = [] # list of dict 记录句子中每个位置的forward score
        current_score = {} # 当前pos不同tag的forward score
        last_score = {} # 前一个pos同tag的forward score
        score = 0.0
        # 初始化
        for tag in self.tag_dict:
            offset = self.tag_dict[tag]*self.feature_length
            
            unigram_feature = self.create_unigram_feature(sen, 0)
            unigram_feature_id = self.get_feature_id(unigram_feature)
            unigram_score = self.dot(unigram_feature_id, offset)
            
            bigram_feature = self.create_bigram_feature("START")
            bigram_feature_id = self.get_feature_id(bigram_feature)
            bigram_score = self.dot(bigram_feature_id, offset)
            
            score = unigram_score + bigram_score
            current_score[tag] = score
        forward_score.append(current_score) # alpha(0, t)
        
        # 后面的词
        len_sen = len(sen.word)
        for pos in range(1, len_sen):
            last_score = current_score # 保存上个位置的score
            current_score = {} # 重置当前位置的score
            for tag in self.tag_dict: # 当前pos的tag
                score = 0.0
                offset = self.tag_dict[tag]*self.feature_length
                
                unigram_feature = self.create_unigram_feature(sen, pos)
                unigram_feature_id = self.get_feature_id(unigram_feature)
                unigram_score = self.dot(unigram_feature_id, offset)
                
                for left_tag in self.tag_dict: # 前一个位置的tag
                    bigram_feature = self.create_bigram_feature(left_tag)
                    bigram_feature_id = self.get_feature_id(bigram_feature)
                    bigram_score = self.dot(bigram_feature_id, offset)
                    
                    temp_score = bigram_score + unigram_score + last_score[left_tag]
                    if self.tag_dict[left_tag] == 0: # 第一次计算
                        score = temp_score
                    else:
                        score = self.log_exp_sum(score, temp_score)
                        
                current_score[tag] = score
            forward_score.append(current_score) # alpha(pos, t)
            
        # STOP位置
        offset = self.feature_length*self.tag_dict_stop["STOP"]
        for left_tag in self.tag_dict:
            bigram_feature = self.create_bigram_feature(left_tag)
            bigram_feature_id = self.get_feature_id(bigram_feature)
            bigram_score = self.dot(bigram_feature_id, offset)
            
            temp_score = bigram_score + current_score[left_tag]
            if self.tag_dict[left_tag] == 0: # 第一次计算
                score = temp_score
            else:
                score = self.log_exp_sum(score, temp_score)
        forward_score.append(score)
            
        return forward_score
    
    def backward(self, sen):
        backward_score = [] # list of dict 记录句子中每个位置的forward score
        current_score = {} # 当前pos不同tag的forward score
        last_score = {} # 前一个pos同tag的forward score
        score = 0.0
        len_sen = len(sen.word)
        
        # 初始化，最后一个词
        for tag in self.tag_dict:
            offset = self.tag_dict_stop["STOP"]*self.feature_length
            
            bigram_feature = self.create_bigram_feature(tag)
            bigram_feature_id = self.get_feature_id(bigram_feature)
            bigram_score = self.dot(bigram_feature_id, offset)
            
            score = bigram_score
            current_score[tag] = score
        backward_score.append(current_score)
        
         # 前面的词
        range_list = list(reversed(range(0, len_sen - 1)))
        for pos in range_list:
            last_score = current_score # 保存上个位置的score
            current_score = {} # 重置当前位置的score
            for tag in self.tag_dict: # 左边的tag
                score = 0.0
                unigram_feature = self.create_unigram_feature(sen, pos + 1)
                unigram_feature_id = self.get_feature_id(unigram_feature)
                
                bigram_feature = self.create_bigram_feature(tag)
                bigram_feature_id = self.get_feature_id(bigram_feature)
                for right_tag in self.tag_dict:
                    offset = self.tag_dict[right_tag]*self.feature_length 
                    bigram_score = self.dot(bigram_feature_id, offset)
                    unigram_score = self.dot(unigram_feature_id, offset)
                    # 从右边开始计算
                    temp_score = bigram_score + unigram_score + last_score[right_tag]
                    if self.tag_dict[right_tag] == 0: # 第一次计算
                        score = temp_score
                    else:
                        score = self.log_exp_sum(score, temp_score)
                        
                current_score[tag] = score
            backward_score.insert(0, current_score) # alpha(pos, t)
            
        # "START" 位置
        """tag = "START"
        for right_tag in self.tag_dict:
            offset = self.tag_dict[right_tag]*self.feature_length
            
            unigram_feature = self.create_unigram_feature(sen, 0)
            unigram_feature_id = self.get_feature_id(unigram_feature)
            unigram_score = self.dot(unigram_feature_id, offset)
            
            bigram_feature = self.create_bigram_feature("START")
            bigram_feature_id = self.get_feature_id(bigram_feature)
            bigram_score = self.dot(bigram_feature_id, offset)
            
            temp_score = bigram_score + unigram_score + last_score[right_tag]
            
            if self.tag_dict[right_tag] == 0: # 第一次计算
                score = temp_score
            else:
                score = self.log_exp_sum(score, temp_score)
        backward_score.insert(0, score)"""
            
        return backward_score
                    
        
    def max_tag(self, sen, pos):
        max_score = float("-Inf")
        max_tag = ""
        if pos == 0:
            left_tag = "START"
        else:
            left_tag = sen.tag[pos - 1]
        fv = self.create_feature(sen, pos, left_tag)
        fv_id = self.get_feature_id(fv)
        for t in self.tag_dict:
            offset = self.tag_dict[t]*self.feature_length
            score = self.dot(fv_id, offset)
            if(score > max_score):
                max_score = score
                max_tag = t
        #print(max_tag)
        return max_tag
    
    def max_tag_sequence(self, sen):
        tag_sequence = [] # 最终的词性序列
        forward_score = self.forward(sen)
        backward_score = self.backward(sen)
        sen_len = len(sen.word)
        
        for pos in range(1, sen_len):
            max_tag = ""
            max_left_tag = ""
            max_score = 0
            for tag in self.tag_dict: # 当前词性
                current_backward_score = backward_score[pos][tag]
                offset = self.tag_dict[tag]*self.feature_length
                for left_tag in self.tag_dict: # 前一个词性
                    current_forward_score = forward_score[pos - 1][left_tag]
                    fv = self.create_feature(sen, pos, left_tag)
                    fv_id = self.get_feature_id(fv)
                    score = self.dot(fv_id, offset)
                    temp_score = current_backward_score + score + current_forward_score
                    if max_score < temp_score:
                        max_score = temp_score
                        max_tag = tag
                        max_left_tag = left_tag
            tag_sequence.append(max_left_tag)
        tag_sequence.append(max_tag)
        return tag_sequence
    
    def update_g(self, sen):
        forward_score = self.forward(sen)
        backward_score = self.backward(sen)
        len_sen = len(sen.word)
        for pos in range(len_sen):
            correct_tag = sen.tag[pos]
            offset = self.tag_dict[correct_tag]*self.feature_length
            if pos == 0:
                left_tag = "START"
            else:
                left_tag = sen.tag[pos - 1]
            fv = self.create_feature(sen, pos, left_tag)
            fv_id = self.get_feature_id(fv)
            for f_id in fv_id:
                self.g[offset + f_id] += 1
                
        dinominator = forward_score[-1]
        
        # "START" 位置
        for tag in self.tag_dict:
            current_backward_score = backward_score[0][tag]
            left_tag = "START"
            fv = self.create_feature(sen, 0, left_tag)
            fv_id = self.get_feature_id(fv)
            score = self.dot(fv_id, offset)
            numerator = current_backward_score + score
            p = math.exp(numerator - dinominator) # log形式
            for f_id in fv_id:
                self.g[offset + f_id] -= p 
            
        for pos in range(1, len_sen):
            for tag in self.tag_dict:
                current_backward_score = backward_score[pos][tag]
                offset = self.tag_dict[tag]*self.feature_length
                for left_tag in self.tag_dict:
                    current_forward_score = forward_score[pos - 1][left_tag]
                    fv = self.create_feature(sen, pos, left_tag)
                    fv_id = self.get_feature_id(fv)
                    score = self.dot(fv_id, offset)
                    numerator = current_backward_score + score + current_forward_score
                    p = math.exp(numerator - dinominator) # log形式
                    for f_id in fv_id:
                        self.g[offset + f_id] -= p   

    
    def update_w(self):
        for i in range(self.feature_length*self.tag_length):
            self.w[i] += self.g[i]

            
    def evaluate(self, dataset):
        count = 0
        total_count = 0
        for sen in dataset.sentences:
            max_tag_sequence = self.max_tag_sequence(sen)
            for pos in range(len(sen.word)):
                max_tag = max_tag_sequence[pos]
                #max_tag = self.max_tag(sen, pos)
                correct_tag = sen.tag[pos]
                if(max_tag == correct_tag):
                    count += 1
                total_count += 1
                #print("total_count:", total_count)
        print(dataset.name +".conll precision:" + str(count / total_count))
        return count, total_count, count / total_count
        
    def online_training(self, max_epochs):
        max_train_precision = 0.0
        max_dev_precision = 0.0
        max_iterator = 0
        b = 0
        batch = 1
        print("*******start iteration************")
        for epoch in range(max_epochs):
            print("epoch:" + str(epoch))
            for sen in self.train.sentences:
                self.update_g(sen)
                b += 1
                if(b % batch == 0):
                    self.update_w()
                    self.g = [0]*(self.feature_length*self.tag_length)
                    
            if(b % batch != 0):
                self.update_w()
                self.g = [0]*(self.feature_length*self.tag_length)
                    
            
            count_train, total_count_train, pre_train = self.evaluate(self.train)
            count_dev, total_count_dev, pre_dev = self.evaluate(self.dev)
            #print("w: " ,self.w[:10])
            
            if(pre_train > max_train_precision):
                max_train_precision = pre_train
            if(pre_dev > max_dev_precision):
                max_dev_precision = pre_dev
                max_iterator = epoch
            
        print("**********stop iteration**************")
        print("train.conll max precision:" + str(max_train_precision))
        print("dev.conll max precision:" + str(max_dev_precision) + " in epoch " + str(max_iterator))
            
if __name__ == "__main__":
    starttime = datetime.datetime.now()
    crf = conditional_random_field()
    crf.create_feature_space()
    max_epochs = 15
    crf.online_training(max_epochs)
    endtime = datetime.datetime.now()
    print("executing time is "+str((endtime-starttime).seconds)+" s")