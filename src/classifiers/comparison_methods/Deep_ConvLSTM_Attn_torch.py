"The implementation of article 'On attention models for human activity recognition' (Deep_ConvLSTM_Attn)"

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import time
from utils.utils import *
import os

class Deep_ConvLSTM_Attn(nn.Module):
    def __init__(self, input_2Dfeature_channel, input_channel, kernel_size,
                 feature_channel, hidden_size, drop_rate, num_class):
        
        super(Deep_ConvLSTM_Attn, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_2Dfeature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        self.lstm = nn.LSTM(
            input_size  = input_channel*feature_channel,
            hidden_size = hidden_size,
            num_layers  = 2,
            batch_first = True
            )
        
        self.dropout = nn.Dropout(drop_rate)
        
        self.time_attention1 = nn.Sequential(
            nn.Linear(hidden_size,hidden_size//2),
            nn.Tanh()
            )
        
        self.time_attention2 = nn.Sequential(
            nn.Linear(hidden_size//2,1),
            nn.Tanh()
            )
        
        self.linear = nn.Linear(hidden_size, num_class)
    
    def forward(self, x):
        
        batch_size = x.shape[0]
        feature_channel = x.shape[1]
        input_channel = x.shape[2]
        data_length = x.shape[-1]
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #############
        x = x.view(batch_size, -1, data_length)
        x = x.permute(0,2,1)
        #############
        x, hidden = self.lstm(x, None)
        x = self.dropout(x)
        
        # cal time attention
        time_attention1 = self.time_attention1(x)
        time_attention2 = self.time_attention2(time_attention1).squeeze(2)
        time_attention2 = F.softmax(time_attention2, dim=1)
        
        # weighting time steps
        x = x * (time_attention2.unsqueeze(-1))
        x_feature = x
        x = torch.sum(x, dim=1).squeeze()

        output = self.linear(x)

        return output, x_feature
    
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
