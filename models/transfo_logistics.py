import pdb
from torch.nn.utils import weight_norm as wn
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

if __name__ == "__main__":
    from interface import ConditionedGenerativeModel
    import sys, os
    sys.path.insert(1, os.path.join(sys.path[0], '../utils')) 
    from pixelcnnpp_utils import *
else:
    from models.interface import ConditionedGenerativeModel
    from utils.pixelcnnpp_utils import *


class GenerativeTransformer(ConditionedGenerativeModel):
    ''' Code partly taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html '''

    def __init__(self, embd_size, d_model, nhead, nlayers, dropout=0.2):
        '''
        :param embd_size: int, dimension of the conditional embedding
        '''

        super(GenerativeTransformer, self).__init__(embd_size)
        self.model_type = 'Transformer'

        # Model:
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.color_to_embedding = nn.Linear(3, d_model)

        self.loss = discretized_mix_logistic_loss

        self.condition_embed_layer = nn.Linear(embd_size, d_model)

        self.embedding_to_logistic_distrib = nn.Sequential(
            nn.Linear(self.d_model, 100),
            nn.ELU(),
            nn.Linear(100, 100)
        )

        # Next attributes will be initialized at the first iteration
        self.first_token_embedding = None
        self.mask = None
        self.h, self.w = None, None

    def _generate_square_subsequent_mask(self, sz):
        """ Returns tensor (sz * sz) strictly upper triangular of -infty (rest is 0) """
        mask = (torch.triu(torch.ones(sz, sz), diagonal=0) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, imgs, condition_embd):
        '''
        :param imgs: torch.FloatTensor bsize * c * h * w
        :param condition_embd: torch.FloatTensor bsize * embd_size
        :return: outputs : dict of ouputs, this can be {"d_loss" : d_loss, "g_loss" : g_loss"} for a gan
        '''

        device = imgs.device

        bsize, c, h, w = imgs.shape # c is 3
        self.h = h
        self.w = w
        assert c == 3

        initial_imgs = imgs

        imgs = imgs.reshape(bsize, c, -1) # bsize, 3, h*w
        imgs = imgs.permute(2, 0, 1) # h*w, bsize, 3
        imgs = self.color_to_embedding(imgs) # h*w, bsize, d_model

        if self.mask is None:
            self.mask = self._generate_square_subsequent_mask(imgs.size(0)).to(device) #h, w

        if self.first_token_embedding is None:
            self.first_token_embedding = (torch.randn(
                (1, 1, self.d_model), dtype=imgs.dtype
            ) / math.sqrt(self.d_model)).to(device) #1, 1, d_model

        condition_embd_model = self.condition_embed_layer(condition_embd).squeeze(0)

        # shift everything by 1 to the right, such that embedding h,w will predict pixel h,w and not h,w+1
        imgs_for_prediction = torch.cat(
            [self.first_token_embedding.expand(1, bsize, self.d_model) + condition_embd_model, 
            imgs[:-1,:,:]], 0
        )
        imgs_for_prediction = self.pos_encoder(imgs_for_prediction)
        output_transformer = self.transformer_encoder(imgs_for_prediction, self.mask) # h*w, bsize, d_model

        output_transformer = output_transformer.reshape(h, w, bsize, self.d_model)

        model_output = self.embedding_to_logistic_distrib(output_transformer) # h, w, bsize, 100
        model_output = model_output.permute(2,3,0,1) #bsize, 100, h, w
        
        loss = self.loss(initial_imgs, model_output, nmix=10) / (
                    h * w * bsize * c)

        outputs = {"loss": loss, "log_likelihood": None,
                   "log_probs": model_output}  

        return outputs


    def likelihood(self, imgs, condition_embd):
        '''
        :param imgs: torch.FloatTensor bsize * c * h * w
        :param condition_embd: torch.FloatTensor bsize * embd_size
        :return: likelihoods : torch.FloatTensor of size bSize, likelihoods of the images conditioned on the captions
        '''
        return None # As in pixelcnnpp

    def sample(self, captions_embd):
        '''
        :param captions_embd: torch.FloatTensor bsize * embd_size
        :return: imgs : torch.FloatTensor of size n_imgs * c * h * w
        '''

        # self.eval()
        # bsize, channels, h, w = [captions_embd.size(0)] + [3, self.h, self.w]

        # data = torch.zeros((bsize, channels, h, w), dtype=torch.float32, device=captions_embd.device,
        #                    requires_grad=False)

        # with torch.no_grad():
        #     for i in tqdm(range(h)):
        #         for j in range(w):
        #             out = self.forward(data, captions_embd)
        #             # print(out)
        #             out_sample = sample_from_discretized_mix_logistic(out, 10)
        #             data[:, :, i, j] = out_sample[:, :, i, j]
        # return data


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



if __name__ == "__main__":

    transformer_model = GenerativeTransformer(16, 4, 2, 1) #embedzise, d_model, nhead, nlayers
    transformer_model.eval()
    example_img = torch.rand((32, 3, 28, 28)) * 2 -1 # b, c, h, w
    conditionning = torch.randn((32, 16)) # b, embedsize
    result = transformer_model(example_img, conditionning)
    print(result['loss'])
    print(result['log_probs'].shape) # bsize, 100, h, w

    # Test: the algorithm should always return the same result with eval mode
    result2 = transformer_model(example_img, conditionning)
    print(torch.max(result2['log_probs'] - result['log_probs']), 'should be 0')
  
    # Test: batch elements should not influence themselves together
    example_img3 = example_img
    example_img3[2:,:,:,:] = torch.rand((30, 3, 28, 28))
    result3 = transformer_model(example_img3, conditionning)
    print(torch.max(result3['log_probs'][:2] - result['log_probs'][:2]), 'should be 0')

    # Test: the output of a cell should not be influenced by the not visible cells
    # Let's try with cell 12,13
    example_img4 = example_img
    example_img4[0, :, 13:, :] = torch.rand(example_img4[0, :, 13:, :].shape)
    example_img4[0, :, 12, 13:] = torch.rand(example_img4[0, :, 12, 13:].shape)
    result4 = transformer_model(example_img4, conditionning)

    print("\nNot visible cells section:")
    print(torch.max(result4['log_probs'][0,:,:12,:] - result['log_probs'][0,:,:12,:]), 'should be 0')
    print(torch.max(result4['log_probs'][0,:,12,:14] - result['log_probs'][0,:,12,:14]), 'should be 0')
    # Cell 12,13 is supposed to be determined only with the cells before, i.e.  up to 12,12 so it shouldn't change
    # between the two examples
    print(torch.max(result4['log_probs'][0,:,12, 14] - result['log_probs'][0,:,12, 14]), 'should not be 0')
    # As cell 12,13 is not the same in the 2 examples, cell 12,14 of the output should change
    print(torch.max(result4['log_probs'][0,:,13,:] - result['log_probs'][0,:,13,:]), 'should not be 0')


    # Sampling
    transformer_model.sample(torch.randn((6, 16)))


