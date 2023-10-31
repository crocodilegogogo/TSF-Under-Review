"The implementation of article 'Predictive maintenance for general aviation using convolutional Transformers' (ConvTransformer)"

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import time
from utils.utils import *
import os

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.transpose(1,2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        return self.dropout(x)

class SelfAttention(nn.Module):
    def __init__(self, k, heads = 8, drop_rate = 0):
        super(SelfAttention, self).__init__()
        self.k, self.heads = k, heads
        
        self.tokeys    = nn.Linear(k, k * heads, bias = False)
        self.toqueries = nn.Linear(k, k * heads, bias = False)
        self.tovalues  = nn.Linear(k, k * heads, bias = False)
        
        self.dropout_attention = nn.Dropout(drop_rate)
        
        self.unifyheads = nn.Linear(heads * k, k)
        
    def forward(self, x):
        
        b, t, k = x.size()
        h = self.heads
        queries = self.toqueries(x).view(b, t, h, k)
        keys    = self.tokeys(x).view(b, t, h, k)
        values  = self.tovalues(x).view(b, t, h, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys    = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values  = values.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))
        dot  = torch.bmm(queries, keys.transpose(1,2))
        dot = F.softmax(dot, dim=2)
        dot = self.dropout_attention(dot)
        out = torch.bmm(dot, values).view(b, h, t, k)
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h*k)
        
        return self.unifyheads(out) # (b, t, k)

class TransformerBlock(nn.Module):
    def __init__(self, k, heads, drop_rate):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(k, heads = heads, drop_rate = drop_rate)
        self.norm1 = nn.LayerNorm(k)

        self.mlp = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )
        self.norm2 = nn.LayerNorm(k)
        self.dropout_forward = nn.Dropout(drop_rate)

    def forward(self, x):
        
        attended = self.attention(x)
        
        x = self.norm1(attended + x)
        
        feedforward = self.mlp(x)
        
        return self.dropout_forward(self.norm2(feedforward + x))

class Deep_Conv_Transformer(nn.Module):
    def __init__(self, input_2Dfeature_channel, input_channel, kernel_size, feature_channel,
                 feature_channel_out, multiheads, drop_rate, data_length, num_class):
        
        super(Deep_Conv_Transformer, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_2Dfeature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size//2), 1, (0,kernel_size//4)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        self.transition = nn.Sequential(
            nn.Conv1d(feature_channel*input_channel, feature_channel_out, 1, 1),
            nn.BatchNorm1d(feature_channel_out),
            nn.ReLU()
            )
        
        self.position_encode = PositionalEncoding(feature_channel_out, drop_rate, data_length)
        
        self.transformer_block1 = TransformerBlock(feature_channel_out, multiheads, drop_rate)
        
        self.transformer_block2 = TransformerBlock(feature_channel_out, multiheads, drop_rate)
        
        self.transformer_block3 = TransformerBlock(feature_channel_out, multiheads, drop_rate)
        
        self.transformer_block4 = TransformerBlock(feature_channel_out, multiheads, drop_rate)
        
        self.global_ave_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.linear = nn.Linear(feature_channel_out, num_class)

    def forward(self, x, test_flag=False):
        
        # flops
        if len(x.shape) == 3:
            x           = x.unsqueeze(0)
        # flops
        
        # hidden = None
        batch_size = x.shape[0]
        data_length = x.shape[-1]
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = x.view(batch_size, -1, data_length)
        x = self.transition(x)
        
        x = self.position_encode(x)
        x = x.permute(0,2,1) # b,t,f
        
        x = self.transformer_block1(x)
        x = self.transformer_block2(x)
        x = self.transformer_block3(x)
        x = self.transformer_block4(x)
        x = x.permute(0,2,1) # b,f,t
        x_feature = x
        x = self.global_ave_pooling(x).squeeze()
        
        output = self.linear(x)
        
        return output, x
    
def train_op(network, EPOCH, BATCH_SIZE, LR,
             train_x, train_y, val_x, val_y, X_test, y_test,
             output_directory_models, log_training_duration, test_split):
    
    # prepare training_data
    if train_x.shape[0] % BATCH_SIZE == 1:
        drop_last_flag = True
    else:
        drop_last_flag = False
    torch_dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.tensor(train_y).long())
    train_loader = Data.DataLoader(dataset = torch_dataset,
                                    batch_size = BATCH_SIZE,
                                    shuffle = True,
                                    drop_last = drop_last_flag
                                   )
    
    # init lr&train&test loss&acc log
    lr_results = []
    
    loss_train_results = []
    accuracy_train_results = []
    
    loss_validation_results = []
    accuracy_validation_results = []
    macro_f1_val_results        = []
    
    loss_test_results = []
    accuracy_test_results = []
    macro_f1_test_results       = []
    
    # prepare optimizer&scheduler&loss_function
    parameters = network.parameters()
    optimizer = torch.optim.Adam(parameters,lr = LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
                                                           patience=5,
                                                           min_lr=LR/10, verbose=True)
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    # loss_function = LabelSmoothingCrossEntropy()
    
    # save init model
    output_directory_init = os.path.join(output_directory_models, 'init_model.pkl')
    torch.save(network.state_dict(), output_directory_init)   # save only the init parameters
    
    training_duration_logs = []
    start_time = time.time()
    for epoch in range (EPOCH):
        
        for step, (x,y) in enumerate(train_loader):
            
            batch_x = x.cuda()
            batch_y = y.cuda()
            output_bc = network(batch_x)[0]
            
            # cal the sum of pre loss per batch 
            loss = loss_function(output_bc, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # test per epoch
        network.eval()
        # loss_train:loss of training set; accuracy_train:pre acc of training set
        loss_train, accuracy_train, _ = get_test_loss_acc(network, loss_function, train_x, train_y, test_split)        
        loss_validation, accuracy_validation, macro_f1_val = get_test_loss_acc(network, loss_function, val_x, val_y, test_split) 
        loss_test, accuracy_test, macro_f1_test = get_test_loss_acc(network, loss_function, X_test, y_test, test_split)
        network.train()  
        
        # update lr
        scheduler.step(accuracy_validation)
        lr = optimizer.param_groups[0]['lr']
        
        # log lr&train&validation loss&acc per epoch
        lr_results.append(lr)
        loss_train_results.append(loss_train)    
        accuracy_train_results.append(accuracy_train)
        
        loss_validation_results.append(loss_validation)    
        accuracy_validation_results.append(accuracy_validation)
        macro_f1_val_results.append(macro_f1_val)
        
        loss_test_results.append(loss_test)    
        accuracy_test_results.append(accuracy_test)
        macro_f1_test_results.append(macro_f1_test)
        
        # print training process
        if (epoch+1) % 1 == 0:
            print('Epoch:', (epoch+1), '|lr:', lr,
                  '| train_loss:', loss_train, 
                  '| train_acc:', accuracy_train, 
                  '| validation_loss:', loss_validation, 
                  '| validation_acc:', accuracy_validation)
        
        save_models(network, output_directory_models, 
                    loss_train, loss_train_results, 
                    accuracy_validation, accuracy_validation_results,
                    start_time, training_duration_logs)
    
    # log training time 
    per_training_duration = time.time() - start_time
    log_training_duration.append(per_training_duration)
    
    # save last_model
    output_directory_last = os.path.join(output_directory_models, 'last_model.pkl')
    torch.save(network.state_dict(), output_directory_last)
    
    # log history
    history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
                          loss_validation_results, accuracy_validation_results,
                          loss_test_results, accuracy_test_results,
                          output_directory_models)
    
    plot_learning_history(EPOCH, history, output_directory_models)
    
    return(history, per_training_duration, log_training_duration)