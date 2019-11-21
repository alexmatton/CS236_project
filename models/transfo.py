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
else:
    from models.interface import ConditionedGenerativeModel



class TransfoSoftmax(ConditionedGenerativeModel):
    ''' Code partly taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html '''

    def __init__(self, embd_size, d_model, nhead, nlayers, dropout=0.5):
        '''
        :param embd_size: int, dimension of the conditional embedding
        '''

        super(TransfoSoftmax, self).__init__(embd_size)
        self.model_type = 'Transformer'

        # Model:
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.color_to_embedding = nn.Sequential(
            nn.Linear(3, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.lin_r = nn.Linear(d_model, 256)
        self.lin_g = nn.Linear(d_model+256, 256)
        self.lin_b = nn.Linear(d_model+2*256, 256)
        self.init_weights()

        self.loss = nn.CrossEntropyLoss()

        self.condition_embed_layer = nn.Linear(embd_size, d_model)

        self.h, self.w = None, None

        # Next attributes will be initialized at the first iteration
        self.first_token_embedding = None
        self.mask = None

    def _generate_square_subsequent_mask(self, sz):
        """ Returns tensor (sz * sz) strictly upper triangular of -infty (rest is 0) """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.color_to_embedding.weight.data.uniform_(-initrange, initrange)
        self.lin_r.weight.data.uniform_(-initrange, initrange)        
        self.lin_g.weight.data.uniform_(-initrange, initrange)
        self.lin_b.weight.data.uniform_(-initrange, initrange)

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

        imgs = imgs.reshape(bsize, c, -1) # bsize, 3, h*w
        initial_imgs_256 = ((imgs+1)/2*255).round().long() # bsize, 3, h*w
        imgs = imgs.permute(2, 0, 1) # h*w, bsize, 3
        imgs = self.color_to_embedding(imgs) # h*w, bsize, d_model

        if self.mask is None:
            self.mask = self._generate_square_subsequent_mask(imgs.size(0)).to(device) #h, w

        if self.first_token_embedding is None:
            self.first_token_embedding = torch.randn(
                (1, 1, self.d_model), dtype=imgs.dtype
            ).to(device) #1, 1, d_model

        condition_embd_model = self.condition_embed_layer(condition_embd).squeeze(0)

        # shift everything by 1 to the right, such that embedding h,w will predict pixel h,w and not h,w+1
        imgs_for_prediction = torch.cat(
            [self.first_token_embedding.expand(1, bsize, self.d_model) + condition_embd_model, 
            imgs[:-1,:,:]], 0
        )
        imgs_for_prediction = self.pos_encoder(imgs_for_prediction)
        output_transformer = self.transformer_encoder(imgs_for_prediction, self.mask) # h*w, bsize, d_model

        # Below: h*w, bsize, 256
        red_distrib = self.lin_r(output_transformer)
        green_distrib = self.lin_g(torch.cat([output_transformer, red_distrib], -1))
        blue_distrib = self.lin_b(
            torch.cat([output_transformer, red_distrib, green_distrib], -1)
        )

        red_distrib = red_distrib.permute(1, 2, 0) # bsize, 256, h*w
        green_distrib = green_distrib.permute(1, 2, 0)
        blue_distrib = blue_distrib.permute(1, 2, 0)

        model_output = torch.stack([red_distrib, green_distrib, blue_distrib], 2) #bsize, 256, 3, h*w
        loss = self.loss(model_output, initial_imgs_256)

        output = {"loss": loss, "log_likelihood": None,
                   "model_output": model_output.reshape(bsize, 256, 3, h, w)} #bsize, 256, 3, h, w

        return output


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

        self.eval()
        bsize, channels, h, w = [captions_embd.size(0)] + [3, self.h, self.w]

        data = torch.zeros((bsize, channels, h, w), dtype=torch.float32, device=captions_embd.device,
                           requires_grad=False)

        with torch.no_grad():
            for i in tqdm(range(h)):
                for j in range(w):
                    out = self.forward(data, captions_embd) # bsize, 256, 3, h, w
                    prob_dist = torch.distributions.Categorical(logits=out['model_output'].permute(0,2,3,4,1))
                    sample_result = prob_dist.sample() # bsize, 3, h, w
                    data[:, :, i, j] = sample_result[:, :, i, j].float() / 255 * 2 - 1
        return data


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

    transformer_model = TransfoSoftmax(16, 12, 2, 3) #embedzise, d_model, nhead, nlayers
    transformer_model.eval()
    example_img = torch.rand((32, 3, 28, 28)) * 2 -1 # b, c, h, w
    conditionning = torch.randn((32, 16)) # b, embedsize
    result = transformer_model(example_img, conditionning)
    print(result['loss'])
    print(result['model_output'].shape) # bsize, 256, 3, h, w

    # Test: the algorithm should always return the same result with eval mode
    result2 = transformer_model(example_img, conditionning)
    print(torch.max(result2['model_output'] - result['model_output']), 'should be 0')
  
    # Test: batch elements should not influence themselves together
    example_img3 = example_img
    example_img3[2:,:,:,:] = torch.rand((30, 3, 28, 28))
    result3 = transformer_model(example_img3, conditionning)
    print(torch.max(result3['model_output'][:2] - result['model_output'][:2]), 'should be 0')

    # Test: the output of a cell should not be influenced by the not visible cells
    # Let's try with cell 12,13
    example_img4 = example_img
    example_img4[0, :, 13:, :] = torch.rand(example_img4[0, :, 13:, :].shape)
    example_img4[0, :, 12, 13:] = torch.rand(example_img4[0, :, 12, 13:].shape)
    result4 = transformer_model(example_img4, conditionning)

    print("\nNot visible cells section:")
    print(torch.max(result4['model_output'][0,:,:,:12,:] - result['model_output'][0,:,:,:12,:]), 'should be 0')
    print(torch.max(result4['model_output'][0,:,:,12,:14] - result['model_output'][0,:,:,12,:14]), 'should be 0')
    # Cell 12,13 is supposed to be determined only with the cells before, i.e.  up to 12,12 so it shouldn't change
    # between the two examples
    print(torch.max(result4['model_output'][0,:,:,12, 14] - result['model_output'][0,:,:,12, 14]), 'should not be 0')
    # As cell 12,13 is not the same in the 2 examples, cell 12,14 of the output should change
    print(torch.max(result4['model_output'][0,:,:,13,:] - result['model_output'][0,:,:,13,:]), 'should not be 0')


    # Sampling
    transformer_model.sample(torch.randn((6, 16)))


