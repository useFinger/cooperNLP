#coding=utf-8
import numpy as np
import time
import tensorflow as tf
# import codecs


with open('data/letters_source.txt', 'r', encoding='utf-8') as f:
	source_data = f.read()

with open('data/letters_target.txt', 'r', encoding='utf-8') as f:
	target_data = f.read()

# f=codecs.open('data/letters_source.txt','rb','utf-8')
# source_data=f.read()

# f=codecs.open('data/letters_target.txt','rb','utf-8')
# target_data=f.read()




# 数据预览
# source_data.split('\n')[:10]

def extract_character_vocab(data):
	'''
	构造映射表
	'''
	special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']

	set_words = list(set([character for line in data.split('\n') for character in line]))
	# 这里要把四个特殊字符添加进词典
	int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
	vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

	return int_to_vocab, vocab_to_int

# 构造映射表
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)

# 对字母进行转换
source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>']) 
			   for letter in line] for line in source_data.split('\n')]
target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>']) 
			   for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data.split('\n')] 

# 查看一下转换结果
# source_int[:10]
print (source_int[:10])

# target_int[:10]
print (target_int[:10])

