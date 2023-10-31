"The implementation of article 'ConvBoost: Boosting ConvNets for Sensor-based Activity Recognition' (Boosting ConvLSTM)"

import torch
import torch.nn as nn
from torch.autograd import Variable
from contiguous_params import ContiguousParams
from utils.utils import *
import time
from utils.constants import INFERENCE_DEVICE

class ConvBoost_Single(nn.Module):
    """Model for human-activity-recognition."""

    def __init__(self, input_channel, num_classes, cnn_channel, n_hidden, drop_rate):
        super(ConvBoost_Single, self).__init__()
        self.n_layers = 2
        self.num_classes = num_classes
        self.n_hidden = n_hidden

        kernal = (5, 1)

        self.features = nn.Sequential(
            nn.Conv2d(1, cnn_channel, kernal, 1, (kernal[0]//2,0)),
            nn.GroupNorm(4, cnn_channel),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(cnn_channel, cnn_channel, kernal, 1, (kernal[0]//2,0)),
            nn.GroupNorm(4, cnn_channel),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(cnn_channel, cnn_channel, kernal, 1, (kernal[0]//2,0)),
            nn.GroupNorm(4, cnn_channel),
            nn.ReLU(),
            # nn.AdaptiveMaxPool2d((4, input_channel))
        )

        self.lstm1 = nn.LSTM(cnn_channel*input_channel, hidden_size=self.n_hidden, num_layers=self.n_layers)
        self.fc = nn.Linear(self.n_hidden, self.num_classes)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, test_flag=False):

        # flops
        if len(x.shape) == 3:
            x           = x.unsqueeze(0)
        # flops
        x = x.permute(0, 1, 3, 2)
        x = self.features(x)

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)
        x, _ = self.lstm1(x)
        x = x[-1, :, :]

        out = self.fc(x)

        return out, x

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
    # t = max(t, 1-t)
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