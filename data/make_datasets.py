import numpy
import codecs
import h5py
import yaml
from fuel.datasets import H5PYDataset
from config import config

def sentenceSplit(data):
    return data.split(u'\n')[:-1];
def WordAndIndex(data):
    if type(data) is not list:
        data = [data];
    word_idx = {None:0,'unk':1,'EOF':2};
    wordlist = [];
    maxLen = 0;
    for sentence in data:
        temp = list();
        sentence = sentence.split(u' ')[:-1]
        sentence.append('EOF')
        for word in sentence:
            if word not in word_idx:
                word_idx[word] = len(word_idx);
            temp.append(word_idx[word]);
        wordlist.append(temp);
        maxLen = len(temp) if maxLen < len(temp) else maxLen
    matrix = numpy.zeros((len(wordlist),maxLen),dtype=int);
    idx_word = dict()
    for x in word_idx.keys():
        idx_word[word_idx[x]] = x;
    for i in xrange(len(wordlist)):
        for j in xrange(len(wordlist[i])):
            matrix[i,j] = int( wordlist[i][j]);
    return word_idx,idx_word,matrix

# load config parameters
locals().update(config);
numpy.random.seed(0)

with codecs.open(source_file,'r','utf-8') as f:
    sourceData = f.read()
    source_sen = sentenceSplit(sourceData);
    source_widx,source_idxw,source_matrix = WordAndIndex(source_sen);

with codecs.open(target_file,'r','utf-8') as f:
    targetData = f.read()
    target_sen = sentenceSplit(targetData);
    target_widx,target_idxw,target_matrix = WordAndIndex(target_sen);

assert len(source_matrix) == len(target_matrix)
nsamples = len(source_matrix);
f = h5py.File(hdf5_file,mode='w')
features = f.create_dataset('features',source_matrix.shape,dtype='int')
targets  = f.create_dataset('targets',target_matrix.shape,dtype='int')

features.attrs['word2index'] = yaml.dump(source_widx);
features.attrs['index2word'] = yaml.dump(source_idxw);
features[...] = source_matrix;

targets.attrs['word2index'] = yaml.dump(target_widx);
targets.attrs['index2word'] = yaml.dump(target_idxw);
targets[...] = target_matrix;

nsamples_train = int(nsamples * train_size);
split_dict = {
        'train' : {'features':(0,nsamples_train),'targets':(0,nsamples_train)},
        'dev'   : {'features':(nsamples_train,nsamples),'targets':(nsamples_train,nsamples)}}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()






