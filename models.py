import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module): # Scalar output for each sentence (probability of being subjective)

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors) #Convert token into word vector 
        self.fc = nn.Linear(embedding_dim, 1)
        
    def forward(self, x, lengths=None):
        #x = [sentence length, batch size (bs)] <-- x's shape 
        embedded = self.embedding(x) # [sentence length, batch size, embedding_dim]
        average = embedded.mean(0) 
        output = self.fc(average).squeeze(1) #[bs]
        
        return output


class CNN(nn.Module):
    
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes): #(100,vocab, 50 filters/kernels, (2,4))
        super(CNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors) #matrix of word vectors         
        
        self.conv1 = nn.Conv2d(1, n_filters, (filter_sizes[0],embedding_dim)) #(
        self.conv2 = nn.Conv2d(1, n_filters, (filter_sizes[1],embedding_dim))
        # 1 for text, kernels, size of kernels (k,embedding_dim) <-- stride = 1 by default

        self.fc = nn.Linear(len(filter_sizes) * n_filters, 1) # 2 * 50 --> (100,2) 
        self.af = nn.Sigmoid() 


    def forward(self, x, lengths=None):
        # x = [sentence length, batch size] <-- x's shape 
        embedded = self.embedding(x.T) # [batch_size, sentence length, embedding_dim]
        embedded = embedded.unsqueeze(1) # [batch_size, 1, sentence length, embedding_dim] <-- match convlayer dim
        
        conv1_out = self.conv1(embedded) 
        conv1_out = F.relu(conv1_out.squeeze(3)) #[bs, n_filters, sentence length - k1 + 1]
        conv2_out = self.conv2(embedded) 
        conv2_out = F.relu(conv2_out.squeeze(3)) #[bs, n_filters, sentence length - k2 + 1]
        #print(conv1_out.shape)
        #print(conv2_out.shape)
        
        pool1_out = F.max_pool1d(conv1_out, conv1_out.shape[2])
        pool1_out = pool1_out.squeeze(2) #[bs, embedding_dim/2] <--- change shape to be allow concatenation later
        pool2_out = F.max_pool1d(conv2_out, conv2_out.shape[2])
        pool2_out = pool2_out.squeeze(2) #[bs, embedding_dim/2]
        #print(pool1_out.shape)
        #print(pool2_out.shape)
        
        # Concatentate results of 2 poolings together (on top of each other) 
        pool_tot = torch.cat((pool1_out, pool2_out), dim = 1) #[bs, embedding_dim]
        #print(pool_tot.shape)

        fc_outputs = self.fc(pool_tot) # <--- vector of fixed dimension 100 
        outputs = self.af(fc_outputs) # <--- make it a probability
        
        return outputs.flatten()

class RNN(nn.Module):
    
    def __init__(self, embedding_dim, vocab, hidden_dim): #100,100
        super(RNN, self).__init__()

        self.embedding_dim = embedding_dim #100
        self.hidden_dim = hidden_dim #100
        
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors) # Matrix of word vectors         
        self.gru = nn.GRU(embedding_dim, hidden_dim) #instantiate RNN (GRU) 
       
        self.fc = nn.Linear(hidden_dim, 1) # [100,1] Takes in last hidden state to generate output 
        self.af = nn.Sigmoid() # Activation function for probability output
        
    def init_hidden(self, batch_size): # Generate initial hidden state 
        h0 = torch.zeros(1, batch_size, self.hidden_dim)
        return h0

    def forward(self, x,x_len, lengths=None):
        # x = [sentence length, batch size (bs)] <-- x's shape 
        batch_size = (x.T).shape[0] # A single number 

        hidden = self.init_hidden(batch_size) # Create the initial hidden state  
        embedded = self.embedding(x) # embedded = [bs, sentence length, embedding_dim]

        pack_embedded = nn.utils.rnn.pack_padded_sequence(embedded,x_len, batch_first=False)
        
        pack_output, hidden = self.gru(pack_embedded,hidden)  ###

        hidden = hidden.contiguous().view(-1, self.hidden_dim) #[1, bs, 1] --> [bs, 1]

        
        fc_outputs = self.fc(hidden) #[bs,1] <--- use last hidden as linear function's input 
        
        outputs=self.af(fc_outputs) #[bs,1] <--- make it a probability 
                
        return outputs.flatten()  
