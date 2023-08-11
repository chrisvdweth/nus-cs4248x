import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from numpy.random import random



################################################################################################
## 
## Vanilla RNN models
##
################################################################################################ 

    
class VanillaRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.out = nn.LogSoftmax(dim=1)
        
    def forward(self, inputs, hidden):
        hidden = torch.tanh(self.i2h(inputs) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.out(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)
    
    
class VanillaRnnLanguageModel(nn.Module):
    
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(VanillaRnnLanguageModel, self).__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.i2h = nn.Linear(embed_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, vocab_size)
        #self.out = nn.LogSoftmax(dim=1)

        
    def forward(self, inputs, hidden):
        embed = self.emb(inputs)
        hidden = torch.tanh(self.i2h(embed) + self.h2h(hidden))
        logits = self.h2o(hidden)
        #output = self.out(output)
        return logits, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    
    
    
    
################################################################################################
## 
## RNN-based Language Model
##
################################################################################################        


class RnnLanguageModel(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # Embedding layer
        self.embedding = nn.Embedding(self.params.vocab_size, self.params.embed_size)
        
        # RNN layer
        rnn = None
        if self.params.rnn_cell.upper() == "RNN":
            rnn = nn.RNN
        elif self.params.rnn_cell.upper() == "GRU":
            rnn = nn.GRU
        elif self.params.rnn_cell.upper() == "LSTM":
            rnn = nn.LSTM
        else:
            raise Exception("[Error] Unknown RNN Cell. Currently supported: RNN, GRU, LSTM")
        self.rnn = rnn(self.params.embed_size,
                       self.params.rnn_hidden_size,
                       num_layers=self.params.rnn_num_layers,
                       dropout=self.params.rnn_dropout,
                       batch_first=True)
        
        # Fully connected layers (incl. Dropout and Activation)
        linear_sizes = [params.rnn_hidden_size] + params.linear_hidden_sizes
        
        self.linears = nn.ModuleList()
        for i in range(len(linear_sizes)-1):
            # Add Dropout layer if probality > 0
            if params.linear_dropout > 0.0:
                self.linears.append(nn.Dropout(p=params.linear_dropout))
            self.linears.append(nn.Linear(linear_sizes[i], linear_sizes[i+1]))
            self.linears.append(nn.ReLU())
        
        self.out = nn.Linear(linear_sizes[-1], params.vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        
        
    def forward(self, X, hidden):
        # inputs.shape = (batch_size, seq_len)
        batch_size, seq_len = X.shape
        
        # Push through embedding layer ==> X.shape = (batch_size, seq_len, embed_size)
        X = self.embedding(X)
        
        # Push through RNN layer
        outputs, hidden = self.rnn(X, hidden)
        
        for l in self.linears:
            outputs = l(outputs)
        
        # Return outputs
        return self.out(outputs), hidden
        
        
    def generate(self, seed_tokens, vocabulary, max_len=50, start_token='<SOS>', stop_token='<EOS>'):
        
        # Keep track of the predicted word which forms our final result
        tokens = seed_tokens
        
        # Vectorize seed tokens using the vocabulary
        inputs = np.array(vocabulary.lookup_indices([start_token]) + vocabulary.lookup_indices(seed_tokens))

        # Convert input to tensor and move to GPU (if available)
        inputs = torch.Tensor(inputs).long().unsqueeze(0).to(self.params.device)
        
        # Initialize hidden states w.r.t. batch size (batches might not always been full)
        hidden = self.init_hidden(1)     
        
        # Push seed tokens through RNN layer
        outputs, hidden = self(inputs, hidden)
        
        # Get outputs of the last step
        outputs = outputs[:,-1,:]
        
        # Iterate over the time steps to predict the next word step by step
        for k in range(max_len):
            
            # Get index of word with the highest probability (no sampling here to keep it simple)
            _, topi = outputs[-1].topk(1)
            word_index = topi.item()

            # If we predict the EOS token, we can stop
            if word_index == vocabulary.lookup_indices([stop_token])[0]:
                break

            # Get the respective word/token and add it to the result list
            tokens.append(vocabulary.lookup_token(word_index))

            # Create the tensor for the last predicted word
            next_input = torch.tensor([[word_index]]).to(self.params.device)

            # Use last predicted word as input for the next iteration
            outputs, hidden = self(next_input, hidden)
            
        # Return the result words/tokens as a string
        return ' '.join(tokens)            
              
    
    
    def init_hidden(self, batch_size):
        if self.params.rnn_cell.upper() == "LSTM":
            return (torch.zeros(self.params.rnn_num_layers, batch_size, self.params.rnn_hidden_size).to(self.params.device),
                    torch.zeros(self.params.rnn_num_layers, batch_size, self.params.rnn_hidden_size).to(self.params.device))
        else:
            return torch.zeros(self.params.rnn_num_layers, batch_size, self.params.rnn_hidden_size).to(self.params.device)
        
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
    
    
    
    
    
################################################################################################
## 
## RNN-based Text Classification Model 
##
################################################################################################

class RnnTextClassifier(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        
        # We have to memorize this for initializing the hidden state
        self.params = params
        
        # Calculate number of directions
        self.rnn_num_directions = 2 if params.rnn_bidirectional == True else 1
        
        # Calculate scaling factor for first linear (2x the size if attention is used)
        self.scaling_factor = 2 if params.dot_attention == True else 1
        
        #################################################################################
        ### Create layers
        #################################################################################
        
        # Embedding layer
        self.embedding = nn.Embedding(params.vocab_size, params.embed_size)
        
        # Recurrent Layer
        rnn = None
        if self.params.rnn_cell.upper() == "RNN":
            rnn = nn.RNN
        elif self.params.rnn_cell.upper() == "GRU":
            rnn = nn.GRU
        elif self.params.rnn_cell.upper() == "LSTM":
            rnn = nn.LSTM
        else:
            raise Exception("[Error] Unknown RNN Cell. Currently supported: RNN, GRU, LSTM")
        self.rnn = rnn(params.embed_size,
                       params.rnn_hidden_size,
                       num_layers=params.rnn_num_layers,
                       bidirectional=params.rnn_bidirectional,
                       dropout=params.rnn_dropout,
                       batch_first=True)
        
        # Linear layers (incl. Dropout and Activation)
        linear_sizes = [params.rnn_hidden_size * self.rnn_num_directions * self.scaling_factor] + params.linear_hidden_sizes
        
        self.linears = nn.ModuleList()
        for i in range(len(linear_sizes)-1):
            self.linears.append(nn.Linear(linear_sizes[i], linear_sizes[i+1]))
            self.linears.append(nn.ReLU())
            self.linears.append(nn.Dropout(p=params.linear_dropout))
        
        if self.params.dot_attention == True:
            self.attention = DotAttentionClassification()
            
        
        self.out = nn.Linear(linear_sizes[-1], params.output_size)
        
        #################################################################################
        
        
    def forward(self, inputs, hidden):
        
        batch_size, seq_len = inputs.shape

        # Push through embedding layer
        X = self.embedding(inputs)

        # Push through RNN layer
        rnn_outputs, hidden = self.rnn(X, hidden)
        
        # Extract last hidden state
        if self.params.rnn_cell == "LSTM":
            last_hidden = hidden[0].view(self.params.rnn_num_layers, self.rnn_num_directions, batch_size, self.params.rnn_hidden_size)[-1]
        else:
            last_hidden = hidden.view(self.params.rnn_num_layers, self.rnn_num_directions, batch_size, self.params.rnn_hidden_size)[-1]

        # Handle directions
        if self.rnn_num_directions == 1:
            final_hidden = last_hidden.squeeze(0)
        elif self.rnn_num_directions == 2:
            h_1, h_2 = last_hidden[0], last_hidden[1]
            final_hidden = torch.cat((h_1, h_2), 1)  # Concatenate both states
            
        X = final_hidden
        
        # Push through attention layer
        
        if self.params.dot_attention == True:
            #rnn_outputs = rnn_outputs.permute(1, 0, 2)  #
            X, attention_weights = self.attention(rnn_outputs, final_hidden)
        else:
            X, attention_weights = final_hidden, None
        
        # Push through linear layers (incl. Dropout & Activation layers)
        for l in self.linears:
            X = l(X)

        X = self.out(X)
            
        return F.log_softmax(X, dim=1)

    
    def init_hidden(self, batch_size):
        if self.params.rnn_cell == "LSTM":
            return (torch.zeros(self.params.rnn_num_layers * self.rnn_num_directions, batch_size, self.params.rnn_hidden_size),
                    torch.zeros(self.params.rnn_num_layers * self.rnn_num_directions, batch_size, self.params.rnn_hidden_size))
        else:
            return torch.zeros(self.params.rnn_num_layers * self.rnn_num_directions, batch_size, self.params.rnn_hidden_size)
        
        
        









################################################################################################
## 
## RNN-based Sequence-to-Seqence (Seq2Seq) Model 
##
################################################################################################


class Encoder(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # Embedding layer
        self.embedding = nn.Embedding(self.params.vocab_size_encoder, self.params.embed_size)
        
        # Calculate number of directions
        self.num_directions = 2 if self.params.rnn_encoder_bidirectional == True else 1
        
        # Recurrent Layer
        rnn = None
        if self.params.rnn_cell.upper() == "RNN":
            rnn = nn.RNN
        elif self.params.rnn_cell.upper() == "GRU":
            rnn = nn.GRU
        elif self.params.rnn_cell.upper() == "LSTM":
            rnn = nn.LSTM
        else:
            raise Exception("[Error] Unknown RNN Cell. Currently supported: RNN, GRU, LSTM")
        self.rnn = rnn(self.params.embed_size,
                       self.params.rnn_hidden_size,
                       num_layers=self.params.rnn_num_layers,
                       bidirectional=self.params.rnn_encoder_bidirectional,
                       dropout=self.params.rnn_dropout,
                       batch_first=True)
        
        # Initialize weights
        self._init_weights()
        
        
    def forward(self, X):
        # inputs.shape = (batch_size, seq_len)
        batch_size, _ = X.shape
        
        # Initialize hidden states w.r.t. batch size (batches might not always been full)
        self.hidden = self._init_hidden(batch_size)
        
        # Push through embedding layer ==> X.shape = (batch_size, seq_len, embed_size)
        X = self.embedding(X)
        
        # Push through RNN layer
        output, hidden = self.rnn(X, self.hidden)
        
        # Create final hidden state (essentially handles the bidirectionality by concatenating both directions)
        # This is needed as the decoder won't be birectional!
        hidden = self._create_final_hidden(hidden, batch_size)

        return output, hidden        
      
        
    def _create_final_hidden(self, hidden, batch_size):
        # No need to do anything if the RNN is unidirectional
        if self.num_directions == 1:
            return hidden

        if self.params.rnn_cell.upper() == "LSTM":
            h = self._concat_directions(hidden[0], batch_size)
            c = self._concat_directions(hidden[1], batch_size)
            hidden = (h, c)
            pass
        else: # RNN or GRU
            hidden = self._concat_directions(hidden, batch_size)        
        
        return hidden
    
    
    def _concat_directions(self, s, batch_size):
        # s.shape = (num_layers*num_directions, batch_size, hidden_size)
        X = s.view(self.params.rnn_num_layers, self.num_directions, batch_size, self.params.rnn_hidden_size)
        # X.shape = (num_layers, num_directions, batch_size, hidden_size)
        X = X.permute(0, 2, 1, 3)
        # X.shape = (num_layers, batch_size, num_directions, hidden_size)
        return X.contiguous().view(self.params.rnn_num_layers, batch_size, -1)    
    
    
    
    def _init_hidden(self, batch_size):
        if self.params.rnn_cell.upper() == "LSTM":
            return (torch.zeros(self.params.rnn_num_layers * self.num_directions, batch_size, self.params.rnn_hidden_size).to(self.params.device),
                    torch.zeros(self.params.rnn_num_layers * self.num_directions, batch_size, self.params.rnn_hidden_size).to(self.params.device))
        else:
            return torch.zeros(self.params.rnn_num_layers * self.num_directions, batch_size, self.params.rnn_hidden_size).to(self.params.device)
        
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                
                
                
class Decoder(nn.Module):
    
    def __init__(self, params, criterion):
        super().__init__()
        self.params = params
        self.criterion = criterion
        
        # Embedding layer
        self.embedding = nn.Embedding(self.params.vocab_size_decoder, self.params.embed_size)
        
        # Calculate number of directions of the encoder (not for the decoder!)
        self.encoder_num_directions = 2 if self.params.rnn_encoder_bidirectional == True else 1    
    
        # RNN layer
        self.hidden_dim = self.params.rnn_hidden_size * self.encoder_num_directions
        
        rnn = None
        if self.params.rnn_cell.upper() == "RNN":
            rnn = nn.RNN
        elif self.params.rnn_cell.upper() == "GRU":
            rnn = nn.GRU
        elif self.params.rnn_cell.upper() == "LSTM":
            rnn = nn.LSTM
        else:
            raise Exception("[Error] Unknown RNN Cell. Currently supported: RNN, GRU, LSTM")
            
        self.hidden_size = self.params.rnn_hidden_size * self.encoder_num_directions
        
        self.rnn = rnn(self.params.embed_size,
                       self.hidden_size,
                       num_layers=self.params.rnn_num_layers,
                       bidirectional=False,
                       dropout=self.params.rnn_dropout,
                       batch_first=True)

        # Attention (optional)
        if self.params.attention.upper() == "DOT":
            self.attention = DotAttention()
            self.first_linear_factor = 2
        else:
            self.first_linear_factor = 1
        
        # Fully connected layers (incl. Dropout and Activation)
        linear_sizes = [params.rnn_hidden_size * self.encoder_num_directions * self.first_linear_factor] + params.linear_hidden_sizes
        
        self.linears = nn.ModuleList()
        for i in range(len(linear_sizes)-1):
            # Add Dropout layer if probality > 0
            if params.linear_dropout > 0.0:
                self.linears.append(nn.Dropout(p=params.linear_dropout))
            self.linears.append(nn.Linear(linear_sizes[i], linear_sizes[i+1]))
            self.linears.append(nn.ReLU())
        
        self.out = nn.Linear(linear_sizes[-1], params.vocab_size_decoder)
        
        # Initialize weights
        self._init_weights()

        
        
    def forward(self, inputs, hidden, encoder_hidden_states):
        batch_size, num_steps = inputs.shape

        # Create SOS token tensor as first input for decoder
        token = torch.LongTensor([[self.params.special_token_sos]] * batch_size).to(self.params.device)

        # Decide whether to do teacher forcing or not
        use_teacher_forcing = random(1)[0] < self.params.teacher_forcing_prob

        # Initiliaze loss
        loss = 0

        # Go through target sequence step by step
        for i in range(num_steps):
            output, hidden, attention_weights = self._step(token, hidden, encoder_hidden_states)

            loss += self.criterion(output, inputs[:, i])

            if use_teacher_forcing:
                # Use the TRUE token of target sequence
                token = inputs[:, i].unsqueeze(dim=1)
            else:
                # Use the PREDICTED token of the target sequence
                topv, topi = output.topk(1)
                token = topi.detach()
                
        return loss
        

        
    def generate(self, hidden, encoder_hidden_states, max_len=100):

        decoded_sequence = []
        # Create SOS token tensor as first input for decoder
        token = torch.LongTensor([[self.params.special_token_sos]] * 1).to(self.params.device)
        
        decoder_attentions = torch.zeros(max_len, encoder_hidden_states.shape[1])

        # Loop over each item in the target sequences (must have the same length!!!)
        for i in range(max_len):
            output, hidden, attention_weights = self._step(token, hidden, encoder_hidden_states)

            # Update attention weights matrix with the latest values
            decoder_attentions[i] = attention_weights
            
            # Get index of hightest value
            topv, topi = output.data.topk(1)
            if topi.item() == self.params.special_token_eos:
                break
            else:
                decoded_sequence.append(topi.item())
                token = topi.detach()

        return decoded_sequence, decoder_attentions[:i]       
        
        
        
    def _step(self, token, decoder_hidden_state, encoder_hidden_states):
        # encoder_outputs.shape = (B x S x H)
        # Get embedding of current input word:
        X = self.embedding(token)
        # Push input word through rnn layer with current hidden state
        output, hidden = self.rnn(X, decoder_hidden_state)
        # output.shape = (B x S=1 x D)
        # hidden.shape = (L x B x H)
        
        if self.params.rnn_cell.upper() == "LSTM":
            last_hidden = hidden[0][-1]
        else:
            last_hidden = hidden[-1]
        # last_hidden.shape = (B x H)
        
        if self.params.attention.upper() == "DOT":
            output, attention_weights = self.attention(encoder_hidden_states, last_hidden)
        else:
            output, attention_weights = last_hidden, None

        # Push through linear layers
        for l in self.linears:
            output = l(output)
            
        # Push through output layer
        output = self.out(output)
            
        #output = F.log_softmax(output.squeeze(dim=1), dim=1)
        
        return output, hidden, attention_weights        
        
        
        

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

                
                
class DotAttention(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, encoder_hidden_states, decoder_hidden_state):
        # Shape of tensors
        # encoder_hidden_states: (B, S, H)
        # decoder_hidden_state:  (B, H)
        
        # Calculate attention weights
        # (B x S x H) @ (B x H x 1) ==> (B x S x 1)
        attention_weights = torch.bmm(encoder_hidden_states, decoder_hidden_state.unsqueeze(2))
        attention_weights = F.softmax(attention_weights.squeeze(2), dim=1)
        
        # Calculate context vector
        # (B x H x S) @ (B x S x 1) ==> (B x H x 1) ==> (B x H)
        context = torch.bmm(encoder_hidden_states.transpose(1,2), attention_weights.unsqueeze(2)).squeeze(2)
        
        # Concatenate context vector and hidden state of decoder (return also the attention weights)
        return torch.cat((context, decoder_hidden_state), dim=1), attention_weights
    
    
    
    
    
    
    
class RnnAttentionSeq2Seq(nn.Module):

    def __init__(self, params, criterion):
        super().__init__()
        self.params = params
        self.criterion = criterion

        self.encoder = Encoder(params)
        self.decoder = Decoder(params, self.criterion)
        
        
    def forward(self, X, Y):
        
        # Push through encoder
        encoder_outputs, encoder_hidden = self.encoder(X)

        # Push through decoder
        loss = self.decoder(Y, encoder_hidden, encoder_outputs)
        
        return loss
    
    
    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()    
        
        
        
        

        
        
################################################################################################
## 
## RNN-based Sequence Labeling Model 
##
################################################################################################        


class RnnSequenceLabeller(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # Embedding layer
        self.embedding = nn.Embedding(self.params.vocab_size, self.params.embed_size)
        
        # Calculate number of directions
        self.num_directions = 2 if self.params.rnn_bidirectional == True else 1
        
        # RNN layer
        rnn = None
        if self.params.rnn_cell.upper() == "RNN":
            rnn = nn.RNN
        elif self.params.rnn_cell.upper() == "GRU":
            rnn = nn.GRU
        elif self.params.rnn_cell.upper() == "LSTM":
            rnn = nn.LSTM
        else:
            raise Exception("[Error] Unknown RNN Cell. Currently supported: RNN, GRU, LSTM")
        self.rnn = rnn(self.params.embed_size,
                       self.params.rnn_hidden_size,
                       num_layers=self.params.rnn_num_layers,
                       bidirectional=self.params.rnn_bidirectional,
                       dropout=self.params.rnn_dropout,
                       batch_first=True)
        
        # Fully connected layers (incl. Dropout and Activation)
        linear_sizes = [params.rnn_hidden_size] + params.linear_hidden_sizes
        
        self.linears = nn.ModuleList()
        for i in range(len(linear_sizes)-1):
            # Add Dropout layer if probality > 0
            if params.linear_dropout > 0.0:
                self.linears.append(nn.Dropout(p=params.linear_dropout))
            self.linears.append(nn.Linear(linear_sizes[i], linear_sizes[i+1]))
            self.linears.append(nn.ReLU())
        
        self.out = nn.Linear(linear_sizes[-1], params.output_size)
        
        # Initialize weights
        self._init_weights()
        
        
        
    def forward(self, X, hidden):
        # inputs.shape = (batch_size, seq_len)
        batch_size, seq_len = X.shape
        
        # Initialize hidden states w.r.t. batch size (batches might not always been full)
        self.hidden = self._init_hidden(batch_size)        
        
        # Push through embedding layer ==> X.shape = (batch_size, seq_len, embed_size)
        X = self.embedding(X)
        
        # Push through RNN layer
        outputs, hidden = self.rnn(X, hidden)
        
        outputs = outputs.reshape(batch_size, seq_len, self.num_directions, self.params.rnn_hidden_size)

        if self.num_directions > 1:
            outputs = outputs[:,:,0,:] + outputs[:,:,1,:]
        else:
            outputs = outputs.squeeze(2)
        
        for l in self.linears:
            outputs = l(outputs)
        
        # Return outputs
        return self.out(outputs), hidden

        
    def _init_hidden(self, batch_size):
        if self.params.rnn_cell.upper() == "LSTM":
            return (torch.zeros(self.params.rnn_num_layers * self.num_directions, batch_size, self.params.rnn_hidden_size),
                    torch.zeros(self.params.rnn_num_layers * self.num_directions, batch_size, self.params.rnn_hidden_size))
        else:
            return torch.zeros(self.params.rnn_num_layers * self.num_directions, batch_size, self.params.rnn_hidden_size)
        
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                
                
                
                
                
