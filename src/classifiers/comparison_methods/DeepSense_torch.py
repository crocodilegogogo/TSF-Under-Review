"The implementation of article 'Deepsense: A unified deep learning framework for time-series mobile sensing data processing' (Deepsense)"

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import time
from utils.utils import *
import os

class Individial_Pos_Convs(nn.Module):
    def __init__(self, input_2Dfeature_channel, feature_channel, kernel_size, drop_rate):
        super(Individial_Pos_Convs, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_2Dfeature_channel, feature_channel, (1,2*3*3), (1,2*3), (0,0)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        self.dropout1 = nn.Dropout(drop_rate)
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,3), 1, (0,0)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        self.dropout2 = nn.Dropout(drop_rate)
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,0)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = x.unsqueeze(dim=4)
        return x

class DeepSense(nn.Module):
    def __init__(self, input_2Dfeature_channel, input_channel, POS_NUM, kernel_size,
                 feature_channel, merge_kernel_size1, merge_kernel_size2, merge_kernel_size3,
                 hidden_size, drop_rate, drop_rate_gru, num_class, datasetname):
        
        super(DeepSense, self).__init__()
        
        self.datasetname   = datasetname
        self.input_channel = input_channel
        self.POS_NUM       = POS_NUM
        
        if datasetname in ['DSADS']:
            kernel_size = 1
        
        self.Acc_Pos_Convs     = []
        self.Mag_Pos_Convs     = []
        self.Grav_Pos_Convs    = []
        self.Gyro_Pos_Convs    = []
        
        for i in range(POS_NUM):
            
            if input_channel//POS_NUM == 12:
                self.mag_convs = Individial_Pos_Convs(input_2Dfeature_channel, feature_channel, kernel_size, drop_rate)
                setattr(self, 'Mag_Pos_Convs%i' % i, self.mag_convs)
                self.Mag_Pos_Convs.append(self.mag_convs)
            
            if input_channel//POS_NUM == 9 or input_channel//POS_NUM == 12:
                self.acc_convs = Individial_Pos_Convs(input_2Dfeature_channel, feature_channel, kernel_size, drop_rate)
                setattr(self, 'Acc_Pos_Convs%i' % i, self.acc_convs)
                self.Acc_Pos_Convs.append(self.acc_convs)
                
            self.grav_convs = Individial_Pos_Convs(input_2Dfeature_channel, feature_channel, kernel_size, drop_rate)
            setattr(self, 'Grav_Pos_Convs%i' % i, self.grav_convs)
            self.Grav_Pos_Convs.append(self.grav_convs)
            
            self.gyro_convs = Individial_Pos_Convs(input_2Dfeature_channel, feature_channel, kernel_size, drop_rate)
            setattr(self, 'Gyro_Pos_Convs%i' % i, self.gyro_convs)
            self.Gyro_Pos_Convs.append(self.gyro_convs)
        
        # all sensor data merging convs
        self.merge_dropout = nn.Dropout(drop_rate)
        self.sensor_conv1 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,2*merge_kernel_size1), (1,2), (0,merge_kernel_size1)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        self.sensor_dropout1 = nn.Dropout(drop_rate)
        self.sensor_conv2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,2*merge_kernel_size2), (1,2), (0,merge_kernel_size2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        self.sensor_dropout2 = nn.Dropout(drop_rate)
        self.sensor_conv3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,2*merge_kernel_size3), (1,2), (0,merge_kernel_size3)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        if self.datasetname in ['Opportunity','RealWorld','SHO']:
            mul = 5
        elif self.datasetname in ['SHL_2018','DSADS','Pamap2']:
            mul = 3
        else:
            mul = 2
        self.gru = nn.GRU(
            input_size=mul*feature_channel,
            hidden_size=hidden_size,
            num_layers = 2,
            batch_first = True
            )
        
        self.gru_dropout = nn.Dropout(drop_rate_gru)
        
        self.linear = nn.Linear(hidden_size, num_class)
        
    def forward(self, x, test_flag=False):
        
        # flops
        if len(x.shape) == 3:
            x           = x.unsqueeze(0)
        # flops
        
        batch_size = x.shape[0]
        
        for i in range(self.POS_NUM):
            x_pos = x[:,i,:,:].unsqueeze(1)
            if self.input_channel//self.POS_NUM == 12:
                inputs = torch.split(x_pos, x_pos.shape[3]//4, dim=3)
                grav_inputs = inputs[0]
                mag_inputs  = inputs[1]
                gyro_inputs = inputs[2]
                acc_inputs  = inputs[3]
                x_mag       = self.Mag_Pos_Convs[i](mag_inputs)
                x_acc       = self.Acc_Pos_Convs[i](acc_inputs)
            if self.input_channel//self.POS_NUM == 9:
                inputs = torch.split(x_pos, x_pos.shape[3]//3, dim=3)
                grav_inputs = inputs[0]
                gyro_inputs = inputs[1]
                acc_inputs  = inputs[2]
                x_acc       = self.Acc_Pos_Convs[i](acc_inputs)
            if self.input_channel//self.POS_NUM == 6:
                inputs = torch.split(x_pos, x_pos.shape[3]//2, dim=3)
                grav_inputs = inputs[0]
                gyro_inputs = inputs[1]
            x_grav          = self.Grav_Pos_Convs[i](grav_inputs)
            x_gyro          = self.Gyro_Pos_Convs[i](gyro_inputs)
            
            if i == 0:
                if self.input_channel//self.POS_NUM == 12:
                    x_all_sensor = torch.cat([x_acc, x_grav, x_gyro, x_mag],4)
                elif self.input_channel//self.POS_NUM == 9:
                    x_all_sensor = torch.cat([x_acc, x_grav, x_gyro],4)
                elif self.input_channel//self.POS_NUM == 6:
                    x_all_sensor = torch.cat([x_grav, x_gyro],4)
            else:
                if self.input_channel//self.POS_NUM == 12:
                    x_all_sensor = torch.cat([x_all_sensor, x_acc, x_grav, x_gyro, x_mag],4)
                elif self.input_channel//self.POS_NUM == 9:
                    x_all_sensor = torch.cat([x_all_sensor, x_acc, x_grav, x_gyro],4)
                elif self.input_channel//self.POS_NUM == 6:
                    x_all_sensor = torch.cat([x_all_sensor, x_grav, x_gyro],4)
        
        x_all_sensor = self.merge_dropout(x_all_sensor)
        x_all_sensor = x_all_sensor.reshape([x_all_sensor.shape[0], x_all_sensor.shape[1], x_all_sensor.shape[2], -1])
        
        x_all_sensor = self.sensor_conv1(x_all_sensor)
        x_all_sensor = self.sensor_dropout1(x_all_sensor)
        x_all_sensor = self.sensor_conv2(x_all_sensor)
        x_all_sensor = self.sensor_dropout2(x_all_sensor)
        x_all_sensor = self.sensor_conv3(x_all_sensor)
        
        x = x_all_sensor.permute(0,1,3,2)
        data_length = x.shape[-1]
        x = x.contiguous().view(batch_size, -1, data_length)
        x = x.permute(0,2,1)
        
        x, hidden = self.gru(x, None)
        x = self.gru_dropout(x)
        
        # get the last hidden state
        x = x.view(batch_size, data_length, -1)[:,-1,:]
        
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
     optimizer  = torch.optim.Adam(parameters,lr = LR)
     scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
                                                            patience=5,
                                                            min_lr=LR/10, verbose=True)
     loss_function = nn.CrossEntropyLoss(reduction='sum')
     # loss_function = LabelSmoothingCrossEntropy()
    
     # save init model    
     output_directory_init = os.path.join(output_directory_models, 'init_model.pkl')
     torch.save(network.state_dict(), output_directory_init)
    
     training_duration_logs = []
     start_time = time.time()
     for epoch in range (EPOCH):
        
         for step, (x,y) in enumerate(train_loader):
            
             # h_state = None      # for initial hidden state
            
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
     torch.save(network.state_dict(), output_directory_last)   # save only the init parameters
    
     # log history
     history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
                           loss_validation_results, accuracy_validation_results,
                           loss_test_results, accuracy_test_results,
                           output_directory_models)
    
     plot_learning_history(EPOCH, history, output_directory_models)
    
     return(history, per_training_duration, log_training_duration)