#coding=utf-8
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
import time
import dataload as da
import seq2seq_model as sq

# 超参数
# Number of Epochs
epochs = 20
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001


def get_inputs():
    '''
    模型输入tensor
    '''
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    
    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


# 构造graph
train_graph = tf.Graph()

with train_graph.as_default():
	
	# 获得模型输入	
	input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()
	
	training_decoder_output, predicting_decoder_output = sq.seq2seq_model(input_data, 
																	  targets, 
																	  lr, 
																	  target_sequence_length, 
																	  max_target_sequence_length, 
																	  source_sequence_length,
																	  len(da.source_letter_to_int),
																	  len(da.target_letter_to_int),
																	  encoding_embedding_size, 
																	  decoding_embedding_size, 
																	  rnn_size, 
																	  num_layers,
																	  batch_size)	
	
	training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
	predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
	
	masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

	with tf.name_scope("optimization"):
		
		# Loss function
		cost = tf.contrib.seq2seq.sequence_loss(
			training_logits,
			targets,
			masks)

		# Optimizer
		optimizer = tf.train.AdamOptimizer(lr)

		# Gradient Clipping
		gradients = optimizer.compute_gradients(cost)
		capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
		train_op = optimizer.apply_gradients(capped_gradients)



def pad_sentence_batch(sentence_batch, pad_int):
	'''
	对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
	
	参数：
	- sentence batch
	- pad_int: <PAD>对应索引号
	'''
	max_sentence = max([len(sentence) for sentence in sentence_batch])
	return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
	'''
	定义生成器，用来获取batch
	'''
	for batch_i in range(0, len(sources)//batch_size):
		start_i = batch_i * batch_size
		sources_batch = sources[start_i:start_i + batch_size]
		targets_batch = targets[start_i:start_i + batch_size]
		# 补全序列
		pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
		pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
		
		# 记录每条记录的长度
		targets_lengths = []
		for target in targets_batch:
			targets_lengths.append(len(target))
		
		source_lengths = []
		for source in sources_batch:
			source_lengths.append(len(source))
		
		yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths


# 将数据集分割为train和validation
train_source = da.source_int[batch_size:]
train_target = da.target_int[batch_size:]
# 留出一个batch进行验证
valid_source = da.source_int[:batch_size]
valid_target = da.target_int[:batch_size]

(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(get_batches(valid_target, valid_source, batch_size,
						   da.source_letter_to_int['<PAD>'],
						   da.target_letter_to_int['<PAD>']))

display_step = 50 # 每隔50轮输出loss

checkpoint = "model/trained_model.ckpt" 
with tf.Session(graph=train_graph) as sess:
	sess.run(tf.global_variables_initializer())
		
	for epoch_i in range(1, epochs+1):
		for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
				get_batches(train_target, train_source, batch_size,
						   da.source_letter_to_int['<PAD>'],
						   da.target_letter_to_int['<PAD>'])):
			
			_, loss = sess.run(
				[train_op, cost],
				{input_data: sources_batch,
				 targets: targets_batch,
				 lr: learning_rate,
				 target_sequence_length: targets_lengths,
				 source_sequence_length: sources_lengths})

			if batch_i % display_step == 0:
				
				# 计算validation loss
				validation_loss = sess.run(
				[cost],
				{input_data: valid_sources_batch,
				 targets: valid_targets_batch,
				 lr: learning_rate,
				 target_sequence_length: valid_targets_lengths,
				 source_sequence_length: valid_sources_lengths})
				
				print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
					  .format(epoch_i,
							  epochs, 
							  batch_i, 
							  len(train_source) // batch_size, 
							  loss, 
							  validation_loss[0]))

	
	
	# 保存模型
	saver = tf.train.Saver()
	saver.save(sess, checkpoint)
	print('Model Trained and Saved')



def source_to_seq(text):
	'''
	对源数据进行转换
	'''
	sequence_length = 7
	return [da.source_letter_to_int.get(word, da.source_letter_to_int['<UNK>']) for word in text] + [da.source_letter_to_int['<PAD>']]*(sequence_length-len(text))



# 输入一个单词
input_word = 'common'
text = source_to_seq(input_word)

checkpoint = "./model/trained_model.ckpt"

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
	# 加载模型
	loader = tf.train.import_meta_graph(checkpoint + '.meta')
	loader.restore(sess, checkpoint)

	input_data = loaded_graph.get_tensor_by_name('inputs:0')
	logits = loaded_graph.get_tensor_by_name('predictions:0')
	source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
	target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
	
	answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
									  target_sequence_length: [len(input_word)]*batch_size, 
									  source_sequence_length: [len(input_word)]*batch_size})[0] 


pad = da.source_letter_to_int["<PAD>"] 

print('原始输入:', input_word)

print('\nSource')
print('  Word 编号:	{}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([da.source_int_to_letter[i] for i in text])))

print('\nTarget')
print('  Word 编号:	   {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([da.target_int_to_letter[i] for i in answer_logits if i != pad])))