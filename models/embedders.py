from transformers import *

from utils.pixelcnnpp_utils import *
import pdb
from torch.nn.utils import weight_norm as wn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

import numpy as np


def bert_encoder():
    return BERTEncoder()


def class_embedding(n_classes, embedding_dim):
    return nn.Embedding(n_classes, embedding_dim)


def unconditional(n_classes, embedding_dim):
    return nn.Embedding(n_classes, embedding_dim)


class Embedder(nn.Module):
    def __init__(self, embed_size):
        super(Embedder, self).__init__()
        self.embed_size = embed_size

    def forward(self, class_labels, captions):
        raise NotImplementedError


class BERTEncoder(Embedder):
    '''
    pretrained model used to embed text to a 768 dimensional vector
    '''

    def __init__(self):
        super(BERTEncoder, self).__init__(embed_size=768)
        self.pretrained_weights = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_weights)
        self.model = BertModel.from_pretrained(self.pretrained_weights)
        self.max_len = 50

    def tokenize(self, text_batch):
        text_token_ids = [
            torch.tensor(self.tokenizer.encode(string_, add_special_tokens=False, max_length=self.max_len)) for
            string_ in text_batch]
        padded_input = pad_sequence(text_token_ids, batch_first=True, padding_value=0)
        return padded_input

    def forward(self, class_labels, captions):
        '''
        :param class_labels : torch.LongTensor, class ids
        :param list captions: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size=768)
        '''

        padded_input = self.tokenize(captions)
        device = list(self.parameters())[0].device
        padded_input = padded_input.to(device)
        # takes the mean of the last hidden states computed by the pre-trained BERT encoder and return it
        return self.model(padded_input)[0].mean(dim=1)

class GloveEncoder(Embedder):
    '''
    pretrained model used to embed text to a dimensional vector
    '''

    def __init__(self):
        super(GloveEncoder, self).__init__(embed_size=300)
        self.glovepath = 'word_vectors/glove.6B.300d.txt'

        self.embeddings_dict = {}
        with open(self.glovepath, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector

    def forward(self, class_labels, captions):
        '''
        :param class_labels : torch.LongTensor, class ids
        :param list captions: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size=300)
        '''
        embeds = np.stack([self.embeddings_dict[cap] for cap in captions])
        return torch.Tensor(embeds).to(class_labels.device)

class OneHotClassEmbedding(Embedder):

    def __init__(self, num_classes):
        super(OneHotClassEmbedding, self).__init__(embed_size=num_classes)
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.eye(self.num_classes))

    def forward(self, class_labels, captions):
        '''
        :param class_ids : torch.LongTensor, class ids
        :param list text_batch: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size=768)
        '''
        return self.weights[class_labels]


class UnconditionalClassEmbedding(Embedder):
    def __init__(self):
        super(UnconditionalClassEmbedding, self).__init__(embed_size=1)

    def forward(self, class_labels, captions):
        '''
        :param class_ids : torch.LongTensor, class ids
        :param list text_batch: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size=768)
        '''
        zero = torch.zeros(class_labels.size(0), 1).to(class_labels.device)
        return zero
