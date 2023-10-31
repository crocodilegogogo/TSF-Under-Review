"The implementation of article 'Towards a dynamic inter-sensor correlations learning framework for multi-sensor-based wearable human activity recognition' (DynamicWHAR)"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import math
import pandas as pd
import time
from utils.utils import *
import os

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def edge_init(node_num, INFERENCE_DEVICE):
    
    off_diag = np.ones([node_num, node_num]) - np.eye(node_num) # (5, 5)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

    relation_num = node_num - 1
    rel_rec_undirected = np.empty([0, node_num])
    rel_send_undirected = np.empty([0, node_num])
    for k in range(1, relation_num + 1):
        rel_rec_undirected  = np.concatenate( (rel_rec_undirected,rel_rec[((k-1)*relation_num+k-1):(k*relation_num),:] ), axis=0)
        rel_send_undirected  = np.concatenate( (rel_send_undirected,rel_send[((k-1)*relation_num+k-1):(k*relation_num),:] ), axis=0)

    rel_rec_undirected = torch.FloatTensor(rel_rec_undirected)
    rel_send_undirected = torch.FloatTensor(rel_send_undirected)   

    if INFERENCE_DEVICE == 'TEST_CUDA':
        rel_rec_undirected = rel_rec_undirected.cuda()
        rel_send_undirected = rel_send_undirected.cuda()
        
    rel_rec_undirected = Variable(rel_rec_undirected)
    rel_send_undirected = Variable(rel_send_undirected)

    return rel_rec_undirected, rel_send_undirected

class DynamicWHAR (nn.Module):

    def __init__(self, node_num = 5, node_dim = 9, window_size=24, channel_dim=8, 
                 time_reduce_size=10, hid_dim=128, class_num=17, INFERENCE_DEVICE='TEST_CUDA'):
        super(DynamicWHAR, self).__init__()
        self.node_num = node_num * node_dim // 3
        self.node_dim = node_dim
        self.window_size = window_size
        self.channel_dim = channel_dim
        self.time_reduce_size = time_reduce_size
        self.hid_dim = hid_dim
        self.class_num = class_num
        
        self.dropout_prob = 0.6
        self.conv1 = nn.Conv1d(3, self.channel_dim, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(self.channel_dim)
        self.conv2 = nn.Conv1d(self.window_size, self.time_reduce_size, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(self.time_reduce_size)
        self.conv3 = nn.Conv1d(self.channel_dim * self.time_reduce_size * 2, self.channel_dim * self.time_reduce_size * 2, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(self.channel_dim * self.time_reduce_size * 2)        
        self.conv5 = nn.Conv1d(self.channel_dim * self.time_reduce_size * 2, self.channel_dim * self.time_reduce_size * 2, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm1d(self.channel_dim * self.time_reduce_size * 2)
        
        self.msg_fc1 = nn.Linear(self.channel_dim * self.time_reduce_size * 3 * self.node_num, self.hid_dim)
        self.fc_out  = nn.Linear(self.hid_dim, self.class_num)
        
        self.conv4 = nn.Conv1d(self.channel_dim * self.time_reduce_size * 2, 1, kernel_size=1, stride=1)
        
        self.rel_rec, self.rel_send = edge_init(self.node_num, INFERENCE_DEVICE)
        
        self.sigmoid = nn.Sigmoid()

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x) # calculate the features of edges (dim_1)
        senders = torch.matmul(rel_send, x) # calculate the features of edges (dim_2)
        edges = torch.cat([senders, receivers], dim=2)
        return edges
    
    def edge2node(self, x, rel_rec, rel_send, rel_type):
        mask = rel_type.squeeze(-1)
        x = x + x * (mask.unsqueeze(2))
        rel = rel_rec.t() + rel_send.t()
        incoming = torch.matmul(rel, x)
        return incoming / incoming.size(1)

    def forward(self, inputs, test_flag=False):
        
        # flops
        if len(inputs.shape) == 3:
            inputs           = inputs.unsqueeze(0)
        # flops
        
        inputs = inputs.reshape(inputs.shape[0], -1, 3, self.window_size).permute(0,1,3,2)
        x = inputs.reshape(inputs.shape[0]*inputs.shape[1], inputs.shape[2], inputs.shape[3])
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = x.reshape(inputs.shape[0], inputs.shape[1], x.shape[1])
        s_input_1 = x
        
        edge = self.node2edge(s_input_1, self.rel_rec, self.rel_send)
        edge = edge.permute(0, 2, 1)
        edge = F.relu(self.bn3(self.conv3(edge)))
        edge = edge.permute(0, 2, 1)
        
        x = edge.permute(0, 2, 1)
        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        # rel_type = F.sigmoid(x)
        rel_type = self.sigmoid(x)

        s_input_2 = self.edge2node(edge, self.rel_rec, self.rel_send, rel_type)
        s_input_2 = s_input_2.permute(0, 2, 1)
        s_input_2 = F.relu(self.bn5(self.conv5(s_input_2)))
        s_input_2 = s_input_2.permute(0, 2, 1)

        join = torch.cat((s_input_1, s_input_2), dim=2)
        join = join.reshape(join.shape[0], -1)
        join = F.dropout(join, p=self.dropout_prob, training=self.training)
        join = F.relu(self.msg_fc1(join))
        join = F.dropout(join, p=self.dropout_prob, training=self.training)
        preds = self.fc_out(join)        
        
        return preds, join
    
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
    
    # loss_function = nn.CrossEntropyLoss(reduction='sum')
    loss_function = LabelSmoothingCrossEntropy()
    
    # save init model
    output_directory_init = output_directory_models+'init_model.pkl'
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
    output_directory_last = output_directory_models+'last_model.pkl'
    torch.save(network.state_dict(), output_directory_last)
    
    # log history
    history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
                          loss_validation_results, accuracy_validation_results,
                          loss_test_results, accuracy_test_results,
                          output_directory_models)
    
    plot_learning_history(EPOCH, history, output_directory_models)
    
    return(history, per_training_duration, log_training_duration)