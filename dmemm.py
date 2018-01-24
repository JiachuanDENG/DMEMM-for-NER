import subprocess
import argparse

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import sys
import gzip
import cPickle
import timeit
torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"

class DMEMM(nn.Module):
	def __init__(self,vocab_size, tag_to_ix, embedding_dim, hidden_dim):
		super(DMEMM, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.tag_to_ix = tag_to_ix
		self.tag_size = len(tag_to_ix)
		self.we = nn.Embedding(vocab_size, embedding_dim)
		self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
		self.hidden_layer=nn.Linear(self.hidden_dim,self.tag_size)

		self.tag_trans=nn.Parameter(torch.randn([self.tag_size,self.tag_size]))

		self.tag_trans.data[self.tag_to_ix[START_TAG],:]=-1000.
		self.tag_trans.data[:,self.tag_to_ix[STOP_TAG]]=-1000.

	def lstm_feature(self,sentence):
		hidden_cell=(autograd.Variable(torch.randn(2,1,self.hidden_dim//2)),
				autograd.Variable(torch.randn(2,1,self.hidden_dim//2)))
		we=self.we(sentence).view(len(sentence),1,-1)

		lstm_features,hidden_cell=self.lstm_layer(we,hidden_cell)

		lstm_features=lstm_features.view(len(sentence),self.hidden_dim)

		lstm_features=self.hidden_layer(lstm_features)
		return lstm_features
	

	def MEMM_partition(self,features,tags):
		score_sum=autograd.Variable(torch.Tensor([0]))
		tags=torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]),tags])
		i=0
		tag_prev=tags[0]
		for feature in features:
			score=autograd.Variable(torch.Tensor([0]))
			for next_tag in range(self.tag_size):
				score+=torch.exp(feature[next_tag]+self.tag_trans[next_tag,tag_prev])
			i+=1
			tag_prev=tags[i]
			score_sum+=torch.log(score)
		score=autograd.Variable(torch.Tensor([0]))
		for next_tag in range(self.tag_size):
			score+=torch.exp(self.tag_trans[next_tag,tag_prev])
		score_sum+=torch.log(score)
		return score_sum

	def sentenceProb(self,features,tags):
		prob=autograd.Variable(torch.Tensor([0]))
		tags=torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]),tags])

		for i,feature in enumerate(features):
			prob+=self.tag_trans[tags[i+1],tags[i]]+feature[tags[i+1]]
		prob+=self.tag_trans[self.tag_to_ix[STOP_TAG],tags[-1]]
		return prob

	def neg_logLikelihood(self,sentence,tags):
		features=self.lstm_feature(sentence)
		prob=self.sentenceProb(features,tags)
		partition=self.MEMM_partition(features,tags)
		# print 'prob',prob
		# print 'partition',partition
		return partition-prob
	

	def viterbi(self,features):
		def argmax(vec):
			_,idx=torch.max(vec,1)
			return idx.view(-1).data.tolist()[0]
		backtrace=[]

		init=torch.Tensor(1,self.tag_size).fill_(-1000.)
		init[0][self.tag_to_ix[START_TAG]]=0.
		pi=autograd.Variable(init)

		for feature in features:
			backtrace_temp=[]
			viterbi_t=[]
			for tag in range(self.tag_size):
				pi_next=pi+self.tag_trans[tag]
				best_tag_id=argmax(pi_next)
				backtrace_temp.append(best_tag_id)
				viterbi_t.append(pi_next[0][best_tag_id])
			pi=(torch.cat(viterbi_t)+feature).view(1,-1)
			backtrace.append(backtrace_temp)

		#STOP TAG
		pi+=self.tag_trans[self.tag_to_ix[STOP_TAG]]
		best_tag_id=argmax(pi)
		score=pi[0][best_tag_id]

		path=[best_tag_id]
		for backtrace_temp in reversed(backtrace):
			best_tag_id=backtrace_temp[best_tag_id]
			path.append(best_tag_id)
		path.pop()
		path.reverse()
		return path

	def predict(self,sentence):
		lstm_features=self.lstm_feature(sentence)

		path=self.viterbi(lstm_features)
		return path

def conlleval(p, g, w, filename='tempfile.txt'):
	out = ''
	for sl, sp, sw in zip(g, p, w):
		out += 'BOS O O\n'
		for wl, wp, ww in zip(sl, sp, sw):
			out += ww + ' ' + wl + ' ' + wp + '\n'
		out += 'EOS O O\n\n'
	f = open(filename, 'w')
	f.writelines(out)
	f.close()

	return get_perf(filename)


def get_perf(filename):
	_conlleval = 'conlleval.pl'
	proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
	stdout, _ = proc.communicate(open(filename).read())
	for line in stdout.split('\n'):
		if 'accuracy' in line:
			out = line.split()
			break
	precision = float(out[6][:-2])
	recall    = float(out[8][:-2])
	f1score   = float(out[10])

	return (precision, recall, f1score)
def main():
    

	start = timeit.default_timer()

	argparser = argparse.ArgumentParser()
	argparser.add_argument("--data", type=str, default="atis.small.pkl.gz", help="The zipped dataset")

	parsed_args = argparser.parse_args(sys.argv[1:])

	filename = parsed_args.data
	f = gzip.open(filename,'rb')
	train_set, valid_set, test_set, dicts = cPickle.load(f)
	train_lex, _, train_y = train_set

	valid_lex, _, valid_y = valid_set

	test_lex,  _,  test_y  = test_set


	idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())

	idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())


	'''

	To have a look what the original data look like, commnet them before your submission

	'''

	# print test_lex[0], map(lambda t: idx2word[t], test_lex[0])

	# print test_y[0], map(lambda t: idx2label[t], test_y[0])


	'''

	implement you training loop here

	'''
    

	EMBEDDING_DIM = 15

	HIDDEN_DIM = 10

	MODEL_TRAIN=False


	idx2label[len(idx2label)]=START_TAG

	idx2label[len(idx2label)]=STOP_TAG


	tag_to_ix={}

	for k in idx2label:
		tag_to_ix[idx2label[k]]=k
        


	model_load = DMEMM(len(idx2word), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

	model_load.load_state_dict(torch.load('parametersTrain.pkl'))
	optimizer=optim.Adam(model_load.parameters()) 

	if (MODEL_TRAIN):
		for epoch in range(8):  
			print '***************************new Epoch ',epoch,"**********************"
			for i in range(len(train_lex)):
				sentence_in=autograd.Variable(torch.from_numpy(train_lex[i]).type(torch.LongTensor))
				targets=torch.from_numpy(train_y[i]).type(torch.LongTensor)
				model_load.zero_grad()

				neg_log_likelihood=model_load.neg_logLikelihood(sentence_in,targets)
				if i%100==0:
					print 'neg_log_likelihood',neg_log_likelihood
                
				if i%1000==0:
					torch.save(model_load.state_dict(),'parametersTrain.pkl') 
				neg_log_likelihood.backward()
				optimizer.step()

	'''
	how to get f1 score using my functions, you can use it in the validation and training as well
	'''
	# print model_load.predict(autograd.Variable(torch.from_numpy(train_lex[12]).type(torch.LongTensor)))
	# print train_y[12].tolist()
	predictions_test = [map(lambda t: idx2label[t], model_load.predict(autograd.Variable(torch.from_numpy(x).type(torch.LongTensor))) )for x in test_lex[:] ]
	groundtruth_test = [ map(lambda t: idx2label[t], y) for y in test_y[:] ]
	words_test = [ map(lambda t: idx2word[t], w) for w in test_lex[:] ]
	test_precision, test_recall, test_f1score = conlleval(predictions_test, groundtruth_test, words_test)

	print test_precision, test_recall, test_f1score
	stop = timeit.default_timer()

	# print 'time use:',stop - start 



    


if __name__ == '__main__':
	main()












