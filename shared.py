import numpy as np
import lasagne
import theano
import time
import io
import sys
import argparse

from datetime import timedelta
from sklearn.metrics import jaccard_similarity_score, precision_score, recall_score
from six.moves import cPickle as pkl
from collections import Counter, OrderedDict
from itertools import chain

CHARS_DICT_FILE = 'chars_dict.pkl'
LABELS_DICT_FILE = 'labels_dict.pkl'
BEST_MODEL_FILE = 'best_model.npz'
TRAIN_RESULT_FILE = 'train_result.npz'
TEST_RESULT_FILE = 'test_result.npz'
ENCODE_RESULT_FILE = 'encode_result.npz'

# dimensions of network
INPUT_DIM = 145 # max length of input text
CHAR_DIM = 250 # embeddings for character
GRU_DIM = 500 # GRU layer
EMBED_DIM = 500 # embeddings for document

# other model hyper parameters
SCALE = 0.1  # sclae of initial weight values
REGULARIZATION = 0.000001
GRAD_CLIP = 0

# training settings
MAX_EPOCHS = 10
BATCH_SIZE = 64
TRAIN_STOP_RATE = 0.0001
TRAIN_STOP_RANGE = 5

# Frequencies
DISPLAY_FREQUENCY = 5

def model_path(model_name, file_name):
    return 'models/' + model_name + '/' + file_name

def load_docs(path):
    """
    Load documents and labels from txt file.
    
    Args:
        path: File path of txt file.
    
    Returns:
        numpy.ndarray of labels and documents
        example:
        [['foo', 'I am Mr. Foo'],
         ['bar', 'I am Ms. Bar'],
         ['baz,foo'], 'We are Baz & Foo']
    """
    labels = []
    docs = []
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            label, doc = line.rstrip('\n').split('\t', 1)
            labels.append(label.split(','))
            docs.append(doc)
    
    return labels, docs

def build_chars_dict(docs):
    """
    Build character dictionary
    
    Args:
        docs: list of documents
    
    Returns:
        chars_dict: A lookup dict of characters and indices ordered by frequency.
        Index 0 is reserved for unseen characters.
        example:
        { ' ': 1, 'a': 2, 'b': 3 }
    """
    
    char_docs = [list(doc) for doc in docs]
    char_count = Counter(list(chain(*char_docs)))
    chars_dict = OrderedDict([(c[0], i + 1) for i, c in enumerate(char_count.most_common())])
    return chars_dict

def build_labels_dict(labels):
    """
    Build label dictionary
    
    Args:
        labels: nested list of labels
    
    Returns:
        labels_dict: A lookup dict of labels and indices ordered by frequency.
        Index 0 is reserved for unseen labels.
        example:
        { 'Android': 1, 'ShellScript': 2, 'SSH': 3 }
    """
    label_count = Counter(list(chain(*labels)))
    labels_dict = OrderedDict([(l[0], i + 1) for i, l in enumerate(label_count.most_common())])
    return labels_dict

def shuffle_docs(x_chars, x_masks, y):
    p = np.random.permutation(len(x_chars))
    return x_chars[p], x_masks[p], y[p]

def batches(x_chars, x_masks, y, size=BATCH_SIZE):
    return [(x_chars[p:p + size], x_masks[p:p + size], y[p:p + size]) for p in range(0, len(y), size)]

def vectorize_x(docs, chars_dict):
    '''
    Convert input letters from string to integer,
    and make corresponding mask vector.
    
    Args:
        docs: list of docs
        chars_dict: char => integer lookup dictionary
    
    Returns:
        x_docs:
        x_masks: 
    '''
    
    n_samples = len(docs)
    x_docs  = np.zeros((n_samples, INPUT_DIM)).astype('int32')
    x_masks = np.zeros((n_samples, INPUT_DIM)).astype('float32')
    
    for i, doc in enumerate(docs):
        x = [chars_dict[c] if c in chars_dict else 0 for c in list(doc[:INPUT_DIM])]
        len_x = len(x)
        x_docs[i, :len_x] = x
        x_masks[i, :len_x] = 1.
    
    return x_docs, x_masks

def vectorize_y(labels_list, labels_dict):
    y = np.zeros((len(labels_list), len(labels_dict) + 1)).astype('int32')
    for i, labels in enumerate(labels_list):
        for label in labels:
            y[i, labels_dict[label]] = 1 if label in labels_dict else 0
    return y

class Seyade():

    def __init__(self, model_name):
        self.model_name = model_name
    
    def build(self, n_chars, n_classes):
        self.n_chars = n_chars
        self.n_classes = n_classes
        self.params = self.init_params()
        self.compile()
        return self

    def load(self):
        self.params = self.load_params()
        self.n_chars = self.params['W_char_embed'].shape[0]
        self.n_classes =  self.params['W_classify'].shape[1]
        self.compile()
        return self

    def compile(self):
        """
        Prepare model variables and functions
        """
        self.tensor_docs = theano.tensor.imatrix() # TensorType(int32, matrix)
        self.tensor_masks = theano.tensor.fmatrix() # TensorType(float32, matrix)
        self.tensor_labels = theano.tensor.imatrix() # TensorType(int32, vector)
        self.init_network(self.params, self.tensor_docs, self.tensor_masks)
        
        loss = theano.tensor.mean(lasagne.objectives.binary_crossentropy(self.output_classify, self.tensor_labels))
        penalty = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2) * REGULARIZATION
        self.cost = loss + penalty
        
        self.network_params = lasagne.layers.get_all_params(self.network)
        updates = lasagne.updates.adam(self.cost, self.network_params)
        
        print("Compiling Training function...")
        self.fn_train = theano.function([self.tensor_docs, self.tensor_masks, self.tensor_labels], self.cost, updates=updates)
        
        print("Compiling Prediction function...")
        self.fn_predict = theano.function([self.tensor_docs, self.tensor_masks], self.output_classify)
        
        print("Compiling Encoding function...")
        self.fn_embed = theano.function([self.tensor_docs, self.tensor_masks], self.output_embed)
        
        print("Compiling Regularization function...")
        self.fn_penalty = theano.function([], penalty)
    
    def init_params(self):
        """
        Initializes all params
        
        Returns:
            A dict contains initial weights and biases of the model.
        """
        
        np.random.seed(0)
        params = OrderedDict()
        
        # character embedding layer
        params['W_char_embed'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(self.n_chars, CHAR_DIM)).astype('float32'))
        
        # f-GRU layer
        params['W_f_reset']  = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM, GRU_DIM)).astype('float32'))
        params['W_f_update'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM, GRU_DIM)).astype('float32'))
        params['W_f_hidden'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM, GRU_DIM)).astype('float32'))
        params['b_f_reset']  = theano.shared(np.zeros((GRU_DIM)).astype('float32'))
        params['b_f_update'] = theano.shared(np.zeros((GRU_DIM)).astype('float32'))
        params['b_f_hidden'] = theano.shared(np.zeros((GRU_DIM)).astype('float32'))
        params['U_f_reset']  = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(GRU_DIM, GRU_DIM)).astype('float32'))
        params['U_f_update'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(GRU_DIM, GRU_DIM)).astype('float32'))
        params['U_f_hidden'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(GRU_DIM, GRU_DIM)).astype('float32'))
        params['hit_init_f'] = theano.shared(np.zeros((1, GRU_DIM)).astype('float32'))
        
        # b-GRU layer
        params['W_b_reset']  = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM, GRU_DIM)).astype('float32'))
        params['W_b_update'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM, GRU_DIM)).astype('float32'))
        params['W_b_hidden'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM, GRU_DIM)).astype('float32'))
        params['b_b_reset']  = theano.shared(np.zeros((GRU_DIM)).astype('float32'))
        params['b_b_update'] = theano.shared(np.zeros((GRU_DIM)).astype('float32'))
        params['b_b_hidden'] = theano.shared(np.zeros((GRU_DIM)).astype('float32'))
        params['U_b_reset']  = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(GRU_DIM, GRU_DIM)).astype('float32'))
        params['U_b_update'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(GRU_DIM, GRU_DIM)).astype('float32'))
        params['U_b_hidden'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(GRU_DIM, GRU_DIM)).astype('float32'))
        params['hid_init_b'] = theano.shared(np.zeros((1, GRU_DIM)).astype('float32'))
        
        # dense layers
        params['W_f_dense'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(GRU_DIM, EMBED_DIM)).astype('float32'))
        params['W_b_dense'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(GRU_DIM, EMBED_DIM)).astype('float32'))
        params['b_f_dense'] = theano.shared(np.zeros((EMBED_DIM)).astype('float32'))
        params['b_b_dense'] = theano.shared(np.zeros((EMBED_DIM)).astype('float32'))
        
        # softmax layer
        params['W_classify'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(EMBED_DIM, self.n_classes)).astype('float32'))
        params['b_classify'] = theano.shared(np.zeros((self.n_classes)).astype('float32'))
        
        return params
    
    def save_params(self):
        """
        Save model parameters
        """
        save_params = OrderedDict()
        for k, v in self.params.items():
            if type(v) is theano.tensor.sharedvar.TensorSharedVariable:
                save_params[k] = v.get_value()
            else:
                save_params[k] = v
        
        np.savez(self.file_path(BEST_MODEL_FILE), **save_params)
    
    def load_params(self):
        """
        Load previously saved model
        """
        params = OrderedDict()
        with io.open(self.file_path(BEST_MODEL_FILE), 'rb') as f:
            npzfile = np.load(f)
            for k, v in npzfile.items():
                params[k] = v
        
        return params
    
    def init_network(self, params, tensor_docs, tensor_masks):
        
        # Input layer over characters
        l_input = lasagne.layers.InputLayer(shape=(BATCH_SIZE, INPUT_DIM), input_var=tensor_docs)
        
        # Character embedding layer
        l_char_embed = lasagne.layers.EmbeddingLayer(l_input, input_size=self.n_chars, output_size=CHAR_DIM, W=params['W_char_embed'])
        
        # Mask layer for variable length sequences
        l_mask = lasagne.layers.InputLayer(shape=(BATCH_SIZE, INPUT_DIM), input_var=tensor_masks)
        
        # forward-GRU cells and layer
        gate_f_reset  = lasagne.layers.Gate(W_in=params['W_f_reset'],  W_hid=params['U_f_reset'],  W_cell=None, b=params['b_f_reset'],  nonlinearity=lasagne.nonlinearities.sigmoid)
        gate_f_update = lasagne.layers.Gate(W_in=params['W_f_update'], W_hid=params['U_f_update'], W_cell=None, b=params['b_f_update'], nonlinearity=lasagne.nonlinearities.sigmoid)
        gate_f_hidden = lasagne.layers.Gate(W_in=params['W_f_hidden'], W_hid=params['U_f_hidden'], W_cell=None, b=params['b_f_hidden'], nonlinearity=lasagne.nonlinearities.elu)
        l_f_gru = lasagne.layers.GRULayer(l_char_embed, GRU_DIM, resetgate=gate_f_reset, updategate=gate_f_update, hidden_update=gate_f_hidden, hid_init=params['hit_init_f'], backwards=False, learn_init=True, gradient_steps=-1, grad_clipping=GRAD_CLIP, unroll_scan=False, precompute_input=True, mask_input=l_mask)
        
        # backward-GRU cells and layer
        gate_b_reset  = lasagne.layers.Gate(W_in=params['W_b_reset'],  W_hid=params['U_b_reset'],  W_cell=None, b=params['b_b_reset'],  nonlinearity=lasagne.nonlinearities.sigmoid)
        gate_b_update = lasagne.layers.Gate(W_in=params['W_b_update'], W_hid=params['U_b_update'], W_cell=None, b=params['b_b_update'], nonlinearity=lasagne.nonlinearities.sigmoid)
        gate_b_hidden = lasagne.layers.Gate(W_in=params['W_b_hidden'], W_hid=params['U_b_hidden'], W_cell=None, b=params['b_b_hidden'], nonlinearity=lasagne.nonlinearities.elu)
        l_b_gru = lasagne.layers.GRULayer(l_char_embed, GRU_DIM, resetgate=gate_b_reset, updategate=gate_b_update, hidden_update=gate_b_hidden, hid_init=params['hid_init_b'], backwards=True, learn_init=True, gradient_steps=-1, grad_clipping=GRAD_CLIP, unroll_scan=False, precompute_input=True, mask_input=l_mask)
        
        # Slice final states
        l_f_sliced = lasagne.layers.SliceLayer(l_f_gru, -1, 1)
        l_b_sliced = lasagne.layers.SliceLayer(l_b_gru,  0, 1)
        
        
        # Dense layers
        l_f_dense = lasagne.layers.DenseLayer(l_f_sliced, EMBED_DIM, W=params['W_f_dense'], b=params['b_f_dense'], nonlinearity=lasagne.nonlinearities.elu)
        l_b_dense = lasagne.layers.DenseLayer(l_b_sliced, EMBED_DIM, W=params['W_b_dense'], b=params['b_b_dense'], nonlinearity=lasagne.nonlinearities.elu)
        
        # Embed layer by elementwise sum
        l_doc_embed = lasagne.layers.ElemwiseSumLayer([l_f_dense, l_b_dense], coeffs=1)
        
        # Dense layer for classes
        l_classify = lasagne.layers.DenseLayer(l_doc_embed, self.n_classes, W=params['W_classify'], b=params['b_classify'], nonlinearity=lasagne.nonlinearities.sigmoid)
        
        self.network = l_classify
        self.output_embed = lasagne.layers.get_output(l_doc_embed)
        self.output_classify = lasagne.layers.get_output(l_classify)
    
    def train(self, X, X_mask, y):
        return float(self.fn_train(X, X_mask, y))
    
    def predict(self, X, X_mask):
        return self.fn_predict(X, X_mask)
    
    def embed(self, X, X_mask):
        return self.fn_embed(X, X_mask)
    
    def penalty(self):
        return float(self.fn_penalty())
    
    def update_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        updates = lasagne.updates.nesterov_momentum(self.cost, self.network_params, self.learning_rate)
        self.fn_train = theano.function([self.tensor_docs, self.tensor_masks, self.tensor_labels], self.cost, updates=updates)
    
    def save_result(self, result, file_name='result.npz'):
        np.savez(self.file_path(file_name), **result)
    
    def load_result(self, threshold = 0.5):
        
        train_result = {}
        with io.open(self.file_path(TRAIN_RESULT_FILE), 'rb') as f:
            npzfile = np.load(f)
            for k, v in npzfile.items():
                train_result[k] = v
        
        test_result = {}
        with io.open(self.file_path(TEST_RESULT_FILE), 'rb') as f:
            npzfile = np.load(f)
            for k, v in npzfile.items():
                test_result[k] = v
        
        encode_result = {}
        with io.open(self.file_path(ENCODE_RESULT_FILE), 'rb') as f:
            npzfile = np.load(f)
            for k, v in npzfile.items():
                encode_result[k] = v
        
        with io.open(self.file_path(LABELS_DICT_FILE), 'rb') as f:
            self.labels_dict = pkl.load(f)
        
        self.labels_dict_invert = {v: k for k, v in self.labels_dict.items()}
        self.labels_dict_invert[0] = '<unk>'
        
        self.train_docs = self.build_docs(train_result, threshold)
        self.test_docs = self.build_docs(test_result, threshold)
        self.encode_docs = self.build_docs(encode_result, threshold)
    
    def build_docs(self, result, threshold = 0.5):
        docs = []
        for i in range(len(result['docs'])):
            doc = {}
            
            doc['text'] = result['docs'][i]
            
            if i < len(result['embeddings']):
                doc['embedding'] = result['embeddings'][i]
            
            targs = []
            for idx, score in enumerate(result['targets'][i]):
                if score == 1:
                    targs.append(self.labels_dict_invert[idx])
            doc['targets'] = targs
            
            preds = []
            for idx, score in enumerate(result['prediction_scores'][i]):
                if score > threshold:
                    preds.append((self.labels_dict_invert[idx], score))
            doc['scored_predictions'] = sorted(preds, key=lambda l: l[1], reverse=True)
            
            doc['predictions'] = [pred[0] for pred in doc['scored_predictions']]
            
            docs.append(doc)
        
        return docs
    
    def file_path(self, file_name):
        return 'models/' + self.model_name + '/' + file_name
