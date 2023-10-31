"The implementation of article 'GlobalFusion: A global attentional deep learning framework for multisensor information fusion' (GlobalFusion)"

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
import time
from utils.utils import *
import os

class SelfAttention(nn.Module):
    def __init__(self, k, heads = 8, drop_rate = 0):
        super(SelfAttention, self).__init__()
        self.k, self.heads = k, heads
        
        self.tokeys    = nn.Linear(k, k * heads, bias = False)
        self.toqueries = nn.Linear(k, k * heads, bias = False)
        self.tovalues  = nn.Linear(k, k * heads, bias = False)
        
        self.dropout_attention = nn.Dropout(drop_rate)
        self.unifyheads = nn.Linear(heads * k, k)
        
    def forward(self, x_local, x_glob):
        
        b, t_l, k       = x_local.size()
        _, t_g, _       = x_glob.size()
        
        h       = self.heads
        queries = self.toqueries(x_glob).view(b, t_g, h, k)
        keys    = self.tokeys(x_local).view(b, t_l, h, k)
        values  = self.tovalues(x_local).view(b, t_l, h, k)
        
        queries = queries.transpose(1, 2).contiguous().view(b * h, t_g, k)
        keys    = keys.transpose(1, 2).contiguous().view(b * h, t_l, k)
        values  = values.transpose(1, 2).contiguous().view(b * h, t_l, k)
        
        queries = queries / (k ** (1/4))
        keys    = keys / (k ** (1/4))
        
        dot     = torch.bmm(queries, keys.transpose(1,2)).transpose(1, 2)
        dot     = F.softmax(dot, dim=1)
        dot     = self.dropout_attention(dot)
        out     = torch.mul(dot, values).view(b, h, t_l, k)
        # swap h, t_l back, unify heads
        out     = out.transpose(1, 2).contiguous().view(b, t_l, h*k)
        out     = torch.sum(out, 1)
        
        return self.unifyheads(out) # (b, t_l, k)

class Individial_Conv(nn.Module):
    def __init__(self, input_2Dfeature_channel, feature_channel, kernel_size):
        super(Individial_Conv, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_2Dfeature_channel, feature_channel, (kernel_size,1), (2,1), (kernel_size//2,0)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (kernel_size,1), 1, (kernel_size//2,0)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (kernel_size,1), 1, (kernel_size//2,0)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return x

class Spatial_Conv(nn.Module):
    def __init__(self, pos_num, spe_interv, feature_channel, kernel_size):
        super(Spatial_Conv, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(feature_channel*pos_num*spe_interv//2, feature_channel, kernel_size, 1, kernel_size//2),
            nn.BatchNorm1d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(feature_channel, feature_channel, kernel_size, 1, kernel_size//2),
            nn.BatchNorm1d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(feature_channel, feature_channel, kernel_size, 1, kernel_size//2),
            nn.BatchNorm1d(feature_channel),
            nn.ReLU(),
            )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return x

class Modality_Conv(nn.Module):
    def __init__(self, modality_num, feature_channel, kernel_size):
        super(Modality_Conv, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(modality_num*feature_channel, feature_channel, kernel_size, 1, kernel_size//2),
            nn.BatchNorm1d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(feature_channel, feature_channel, kernel_size, 1, kernel_size//2),
            nn.BatchNorm1d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(feature_channel, feature_channel, kernel_size, 1, kernel_size//2),
            nn.BatchNorm1d(feature_channel),
            nn.ReLU(),
            )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return x

class TemporalAttention(nn.Module):
    """
    Temporal attention module
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x):
        out = self.fc(x).squeeze(2)
        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 1)
        return context

class GlobalFusion(nn.Module):
    def __init__(self, input_2Dfeature_channel, input_channel, feature_channel,
                 kernel_size, kernel_size_grav, scale_num, feature_channel_out,
                 multiheads, drop_rate, dataset_name, spe_interv, sin_pos_chnnl, num_class):
        
        super(GlobalFusion, self).__init__()
        
        self.feature_channel      = feature_channel
        self.feature_channel_out  = feature_channel_out
        self.input_channel        = input_channel
        self.sin_pos_chnnl        = sin_pos_chnnl
        
        self.Individial_Convs     = []
        for i in range(input_channel//sin_pos_chnnl):
            Ind_Conv = Individial_Conv(input_2Dfeature_channel, feature_channel, kernel_size)
            setattr(self, 'Individial_Convs%i' % i, Ind_Conv)
            self.Individial_Convs.append(Ind_Conv)
        
        self.transition_pos       = nn.Conv1d(feature_channel*spe_interv//(2*sin_pos_chnnl//3), feature_channel, 1, 1)
        
        self.Spa_Conv             = Spatial_Conv(input_channel//sin_pos_chnnl, spe_interv//(sin_pos_chnnl//3),\
                                                  feature_channel, kernel_size)
        
        self.sa_pos               = SelfAttention(feature_channel, heads = 1, drop_rate = drop_rate)
        
        self.Mod_Conv             = Modality_Conv(sin_pos_chnnl//3,feature_channel, kernel_size)
        
        self.sa_mod               = SelfAttention(feature_channel, heads = 1, drop_rate = drop_rate)
        
        self.rnn = nn.GRU(
            input_size  = feature_channel,
            hidden_size = feature_channel_out,
            num_layers  = 2,
            dropout     = drop_rate,
            batch_first = True
        )
        
        self.ta = TemporalAttention(feature_channel_out)
        
        self.linear = nn.Linear(feature_channel_out, num_class)

    def forward(self, x, test_flag=False):
        
        # flops
        if len(x.shape) == 3:
            x           = x.unsqueeze(0)
        # flops
        
        batch_size       = x.shape[0]
        pos_num          = x.shape[1]
        data_length      = x.shape[2]
        feature_chnnl    = x.shape[-1]
        
        x_input          = x.reshape(batch_size, pos_num, data_length,\
                                     self.input_channel//(3*pos_num), -1)
        x_input          = x_input.permute(0,1,4,3,2).reshape(batch_size,\
                                           pos_num, -1, self.input_channel//(3*pos_num)*data_length)
        for i in range(pos_num):
            x_cur_pos    = self.Individial_Convs[i](x_input[:,i,:,:].unsqueeze(1))
            if i == 0:
                x        = x_cur_pos
            else:
                x        = torch.cat((x, x_cur_pos), 3) # [128, 64, 18, 120]
        
        x_pos = x # [128, 64, 18, 120]
        x_pos = x_pos.reshape(batch_size, -1, pos_num*self.input_channel//(3*pos_num)*data_length) # [128, 64*18, 120]
        x_pos = self.transition_pos(x_pos) # [128, 64, 120]
        x_pos = x_pos.reshape(batch_size, self.feature_channel, pos_num, -1, data_length) # [128, 64, 5, 3, 8]
        x_pos = x_pos.permute(0, 3, 4, 2, 1) # [128, 3, 8, 5, 64]
        x_pos = x_pos.reshape(-1, pos_num, self.feature_channel) # [128*3*8, 5, 64]
        
        x_glob_pos = x # [128, 64, 18, 120]
        x_glob_pos = x_glob_pos.reshape(batch_size, self.feature_channel, -1,
                                        pos_num, self.input_channel//(3*pos_num)*data_length) # [128, 64, 18, 5, 3*8]
        x_glob_pos = x_glob_pos.reshape(batch_size, -1, self.input_channel//(3*pos_num)*data_length) # [128, 64*18*5, 3*8]
        x_glob_pos = self.Spa_Conv(x_glob_pos) # [128, 64, 3*8]
        x_glob_pos = x_glob_pos.permute(0, 2, 1) # [128, 3*8, 64]
        x_glob_pos = x_glob_pos.reshape(-1, self.feature_channel).unsqueeze(1) # [128*3*8, 1, 64]
        
        x_pos_out  = x_glob_pos.squeeze(1) + self.sa_pos(x_pos, x_glob_pos)              # [128*3*8, 64]
        x_pos_out  = x_pos_out.reshape(batch_size, -1, data_length, self.feature_channel)  # [128,3,8,64]
        ########### LN ###########
        x_pos_out  = x_pos_out.permute(0,2,1,3).reshape(batch_size, data_length, -1)
        x_pos_out  = F.layer_norm(x_pos_out, (x_pos_out.shape[-1],))
        x_pos_out  = x_pos_out.reshape(batch_size, data_length, -1, self.feature_channel).permute(0,2,1,3)
        ########### LN ###########
        
        # [128*8, 3, 64]
        x_modality       = x_pos_out.permute(0,2,1,3).reshape(batch_size*data_length, -1, self.feature_channel)
        
        x_glob_modality  = x_pos_out  # [128,3,8,64]
        x_glob_modality  = x_glob_modality.permute(0,1,3,2) # [128,3,64,8]
        x_glob_modality  = x_glob_modality.reshape(batch_size, -1, data_length) # [128,3*64,8]
        x_glob_modality  = self.Mod_Conv(x_glob_modality)   # [128, 64, 8]
        x_glob_modality  = x_glob_modality.permute(0,2,1).reshape(-1, self.feature_channel) # [128*8, 64]
        x_glob_modality  = x_glob_modality.unsqueeze(1)     # [128*8,1,64]
        
        x                = x_glob_modality.squeeze(1) + self.sa_mod(x_modality, x_glob_modality)  # [128*8, 64]
        x                = x.reshape(batch_size, data_length, -1) #.permute(0,2,1)
        ########### LN ###########
        x                = F.layer_norm(x, (x.shape[-1],))
        ########### LN ###########
        
        outputs, h       = self.rnn(x)
        
        out = outputs.view(batch_size, data_length, -1)[:,-1,:]
        
        output           = self.linear(out)
        
        return output, out
    
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
