"The implementation of article 'ConvBoost: Boosting ConvNets for Sensor-based Activity Recognition' (Boosting Attention Model)"

import torch
import torch.nn as nn
from torch.autograd import Variable
from contiguous_params import ContiguousParams
from utils.utils import *
import time
from utils.constants import INFERENCE_DEVICE

class Attn_Boost_Single(nn.Module):
    def __init__(self, input_dim, n_classes, FILTER_SIZE, NUM_FILTERS, hidden_size, num_layers=2, is_bidirectional=False, dropout=0.2,
                 attention_dropout=0.2):
        super(Attn_Boost_Single, self).__init__()
        self.is_bidirectional = is_bidirectional
        self.num_directions = 2 if is_bidirectional else 1
        self.hidden_size = hidden_size
        hidden_dim = hidden_size * self.num_directions
        self.dropout_val = dropout
        self.attention_dropout_val = attention_dropout
        self.conv2DLayer1 = nn.Conv2d(1, NUM_FILTERS, (FILTER_SIZE, 1), 1, (FILTER_SIZE//2,0))
        self.relu1 = nn.ReLU()
        self.conv2DLayer2 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1), 1, (FILTER_SIZE//2,0))
        self.relu2 = nn.ReLU()
        self.conv2DLayer3 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1), 1, (FILTER_SIZE//2,0))
        self.relu3 = nn.ReLU()
        self.conv2DLayer4 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1), 1, (FILTER_SIZE//2,0))
        self.relu4 = nn.ReLU()
        self.lstm = nn.LSTM(NUM_FILTERS * input_dim, self.hidden_size, num_layers, bidirectional=is_bidirectional,
                            dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dense_layer = nn.Linear(hidden_dim, n_classes)
        self.num_layers  = num_layers
        self.attentionLayer1 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh1 = nn.Tanh()
        self.attentionLayer2 = nn.Linear(hidden_dim, 1)
        self.softmax_attention = torch.nn.Softmax(dim=0)
        if INFERENCE_DEVICE == 'TEST_CUDA':
            self.DEVICE = 'cuda'
        else:
            self.DEVICE = 'cpu'

    def forward(self, input, test_flag = False):
        
        # flops
        if len(input.shape) == 3:
            input           = input.unsqueeze(0)
        # flops
        self.batch_size = input.shape[0]
        input = input.permute(0,1,3,2)
        convout1 = self.conv2DLayer1(input)
        convout1 = self.relu1(convout1)
        convout2 = self.conv2DLayer2(convout1)
        convout2 = self.relu2(convout2)
        convout3 = self.conv2DLayer3(convout2)
        convout3 = self.relu3(convout3)
        convout4 = self.conv2DLayer4(convout3)
        convout4 = self.relu4(convout4)
        
        # reshape to put them in the lstm
        lstm_input = convout4.permute(2, 0, 1, 3)
        lstm_input = lstm_input.contiguous()
        lstm_input = lstm_input.view(lstm_input.shape[0], lstm_input.shape[1], -1)
        
        # put things in lstm
        lstm_input = self.dropout(lstm_input)
        output, hidden = self.lstm(lstm_input, self.initHidden())
        
        # attention stuff
        past_context = output[:-1]
        current = output[-1]
        
        attention_layer1_output = self.attentionLayer1(past_context)
        attention_layer1_output = attention_layer1_output + current

        attention_layer1_output = self.tanh1(attention_layer1_output)
        attention_layer1_output = self.attention_dropout(attention_layer1_output)
        attention_layer2_output = self.attentionLayer2(attention_layer1_output)
        attention_layer2_output = attention_layer2_output.squeeze(2)
        # find weights
        attn_weights = self.softmax_attention(attention_layer2_output)
        
        # the cols represent the weights
        attn_weights = attn_weights.unsqueeze(2)
        new_context_vector = torch.sum(attn_weights * past_context, 0)
        
        # use this new context vector for prediction
        # add a skip connection
        new_context_vector = new_context_vector + current
        logits = self.dense_layer(new_context_vector)
        
        return logits, attn_weights

    def initHidden(self):
        h0 = Variable(torch.mul(torch.randn(self.num_layers * self.num_directions, self.batch_size, self.hidden_size), 0.08)).to(self.DEVICE)
        c0 = Variable(torch.mul(torch.randn(self.num_layers * self.num_directions, self.batch_size, self.hidden_size), 0.08)).to(self.DEVICE)
        return (h0.contiguous(), c0.contiguous())

class MixUpLoss(nn.Module):

    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        self.reduction = reduction

    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1, loss2 = self.crit(output, target[:, 0].long()), self.crit(output, target[:, 1].long())
            d = loss1 * target[:, 2] + loss2 * (1 - target[:, 2])
        else:
            d = self.crit(output, target)
        if self.reduction == 'mean':
            return d.mean()
        elif self.reduction == 'sum':
            return d.sum()
        return d

    def get_old(self):
        if hasattr(self, 'old_crit'):
            return self.old_crit
        elif hasattr(self, 'old_red'):
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit

def mixup_data(x, y, alpha=0.4):

    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha, batch_size)
    
    lam = np.concatenate([lam[:, None], 1 - lam[:, None]], 1).max(1)
    
    # tensor and cuda version of lam
    lam = x.new(lam)

    shuffle = torch.randperm(batch_size).cuda()

    x1, y1 = x[shuffle], y[shuffle]
    out_shape = [lam.size(0)] + [1 for _ in range(len(x1.shape) - 1)]

    # [bs, temporal, sensor]
    mixed_x = (x * lam.view(out_shape) + x1 * (1 - lam).view(out_shape))
    # [bs, 3]
    y_a_y_b_lam = torch.cat([y[:, None].float(), y1[:, None].float(), lam[:, None].float()], 1)

    return mixed_x, y_a_y_b_lam

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
    parameters = ContiguousParams(network.parameters())
    optimizer = torch.optim.Adam(parameters.contiguous(),lr = LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
                                                           patience=5,
                                                           min_lr=LR/10, verbose=True)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss_function_nomixup = LabelSmoothingCrossEntropy()
    
    # save init model
    output_directory_init = os.path.join(output_directory_models, 'init_model.pkl')
    torch.save(network.state_dict(), output_directory_init)   # save only the init parameters
    
    training_duration_logs = []
    
    # super param
    mixup = True
    alpha = 0.8
    #############
    
    start_time = time.time()
    for epoch in range (EPOCH):
        
        epoch_tau = epoch+1
        tau = max(1 - (epoch_tau - 1) / 50, 0.5)
        for m in network.modules():
            if hasattr(m, '_update_tau'):
                m._update_tau(tau)
                # print(a)
        
        for step, (x,y) in enumerate(train_loader):
            
            batch_x = x.cuda()
            batch_y = y.cuda()
            
            for bj in range(batch_x.shape[0]):
                num_change = np.random.randint(0, int(batch_x.shape[2] * 0.2))
                dim_location_change = np.random.randint(0, batch_x.shape[2] - num_change)
                batch_x[bj, :, dim_location_change:dim_location_change + num_change, :] = 0
            
            if mixup == True:
                batch_x, batch_y_mixup = mixup_data(batch_x, batch_y, alpha)
            
            logits, out_attn     = network(batch_x)
            
            # cal the sum of pre loss per batch
            if mixup == True:
                loss_function    = MixUpLoss(criterion)
                loss             = loss_function(logits, batch_y_mixup)
            else:
                loss             = loss_function_nomixup(logits, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if mixup == True:
                loss_function = loss_function.get_old()
        
        # test per epoch
        network.eval()
        test_flag = True
        # loss_train:loss of training set; accuracy_train:pre acc of training set
        loss_train, accuracy_train, _ = get_test_loss_acc(network, loss_function, train_x, train_y, test_split, test_flag)
        loss_validation, accuracy_validation, macro_f1_val = get_test_loss_acc(network, loss_function, val_x, val_y, test_split, test_flag)
        loss_test, accuracy_test, macro_f1_test = get_test_loss_acc(network, loss_function, X_test, y_test, test_split, test_flag)
        test_flag = False
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
