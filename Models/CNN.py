import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

#class CNN(nn.Module):
#    def __init__(self, input_channels = 1, fc_input = 576):
#        super(CNN, self).__init__()
#        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5)
#        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
#        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
#        self.fc1 = nn.Linear(fc_input, 256)
#        self.fc2 = nn.Linear(256, 10)
#        self.activation_fc = nn.Linear(256, 1600)
#        for p in self.activation_fc.parameters():
#            p.requires_grad_(False)
#
#    def forward(self, x):
#        x = x.view(-1,1,28,28)
#        x = F.relu(self.conv1(x))
#        x = F.dropout(x, p=0.5, training=self.training)
#        x = F.relu(F.max_pool2d(self.conv2(x), 2))
#        x = F.dropout(x, p=0.5, training=self.training)
#        x = F.relu(F.max_pool2d(self.conv3(x),2))
#        x = F.dropout(x, p=0.5, training=self.training)
#        x = torch.flatten(x, start_dim=1)
#        x = F.relu(self.fc1(x))
#        x = F.dropout(x, training=self.training)
#        x = self.fc2(x)
#        return F.log_softmax(x, dim=1)
#    
#    def get_activations_1(self, x):
#        x = x.view(-1,1,28,28)
#        x= F.relu(self.conv1(x))
#        return torch.flatten(x, start_dim=1)
#
#    def get_activations_2(self, x):
#        x = x.view(-1,1,28,28)
#        x= F.relu(self.conv1(x)) # CHANGE RELU
#        x = F.max_pool2d(self.conv2(x),2) # CHANGE RELU
#        return torch.flatten(x, start_dim=1)
#    
#    def get_activations_3(self, x):
#        x = x.view(-1,1,28,28)
#        x= F.relu(self.conv1(x))
#        x = F.relu(F.max_pool2d(self.conv2(x),2))
#        x = F.max_pool2d(self.conv3(x),2)
#        return torch.flatten(x, start_dim=1)
#    
#    def get_activations_4(self, x):
#        x = x.view(-1,1,28,28)
#        x= F.relu(self.conv1(x))
#        x = F.relu(F.max_pool2d(self.conv2(x),2))
#        x = F.relu(F.max_pool2d(self.conv3(x),2)) 
#        x = torch.flatten(x, start_dim=1)
#        x = self.fc1(x)
#        return torch.flatten(x, start_dim=1)
#    
#    def get_activation_stems(self, x):
#        x = x.view(-1,1,28,28)
#        x= F.relu(self.conv1(x))
#        x = F.relu(F.max_pool2d(self.conv2(x),2))
#        x = F.relu(F.max_pool2d(self.conv3(x),2)) 
#        x = torch.flatten(x, start_dim=1)
#        x = F.relu(self.fc1(x))
#        x = self.activation_fc(x)
#        return torch.flatten(x, start_dim=1)
#    
#    def get_activations(self, x):
#        return self.get_activations_2(x)
#
#class ResidualCNN(nn.Module):
#    def __init__(self, input_channels = 1, fc_input = 576, input_size = 28*28):
#        super(ResidualCNN, self).__init__()
#        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5)
#        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
#        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
#        self.fc1 = nn.Linear(fc_input, 256)
#        #self.fc2 = nn.Linear(256, 10)
#        self.residual1 = nn.Linear(input_size, 3200)
#        self.residual2 = nn.Linear(3200, 576)
#        #self.residual3 = nn.Linear(576, 10)
#        self.class_fc = nn.Linear(256+576,10)
#
#    def forward(self, x):
#        x1 = x.view(-1,1,28,28) 
#        x2 = torch.flatten(x, start_dim=1)
#        # CNN network
#        x = F.relu(self.conv1(x1))
#        x = F.dropout(x, p=0.5, training=self.training)
#        x = F.relu(F.max_pool2d(self.conv2(x), 2))
#        x = F.dropout(x, p=0.5, training=self.training)
#        x = F.relu(F.max_pool2d(self.conv3(x),2))
#        x = F.dropout(x, p=0.5, training=self.training)
#        x = torch.flatten(x, start_dim=1)
#        x = F.relu(self.fc1(x))
#        x = F.dropout(x, training=self.training)
#        #x = F.relu(self.fc2(x))
#        # Residual network
#        z = F.relu(self.residual1(x2))
#        z = F.dropout(z, p=0.5, training=self.training)
#        z = F.relu(self.residual2(z))
#        z = F.dropout(z, p=0.5, training=self.training)
#        #z = F.relu(self.residual3(z))
#        # Combine the two 
#        y = self.class_fc(torch.cat((x,z), dim=1))
#        return F.log_softmax(y, dim=1)
    
    #def get_activations(self, x):
    #    x1 = x.view(-1,1,28,28) 
    #    x2 = torch.flatten(x, start_dim=1)
    #    # CNN network
    #    x = F.relu(self.conv1(x1))
    #    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    #    x = F.relu(F.max_pool2d(self.conv3(x),2))
    #    x = torch.flatten(x, start_dim=1)
    #    x = F.relu(self.fc1(x))
    #    #x = F.relu(self.fc2(x))
    #    # Residual network
    #    z = F.relu(self.residual1(x2))
    #    z = F.relu(self.residual2(z))
    #    #z = F.relu(self.residual3(z))
    #    # Combine the two 
    #    acts = torch.cat((x,z), dim=1)
    #    return torch.flatten(acts, start_dim=1)
    
#    def get_activations(self, x):
#        x = torch.flatten(x, start_dim=1)
#        z = self.residual1(x)
#        #z = self.residual2(z)
#        return torch.flatten(z, start_dim=1)

class CNN(nn.Module):
    def __init__(self, input_channels = 1, fc_input = 576):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(fc_input, 256)
        self.fc2 = nn.Linear(256, 10)
        #self.activation_fc = nn.Linear(256, 1600)
        #for p in self.activation_fc.parameters():
        #    p.requires_grad_(False)

    def forward(self, x):
        x = x.view(-1,1,28,28)
        x = F.tanh(self.conv1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.tanh(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.tanh(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.flatten(x, start_dim=1)
        x = F.tanh(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def get_activations_1(self, x):
        x = x.view(-1,1,28,28)
        x= F.leaky_relu(self.conv1(x))
        return torch.flatten(x, start_dim=1)

    def get_activations_2(self, x):
        x = x.view(-1,1,28,28)
        x= F.leaky_relu(self.conv1(x)) # CHANGE RELU
        x = F.max_pool2d(self.conv2(x),2) # CHANGE RELU
        return torch.flatten(x, start_dim=1)
    
    def get_activations_3(self, x):
        x = x.view(-1,1,28,28)
        x= F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(F.max_pool2d(self.conv2(x),2))
        x = F.max_pool2d(self.conv3(x),2)
        return torch.flatten(x, start_dim=1)
    
    def get_activations_4(self, x):
        x = x.view(-1,1,28,28)
        x= F.tanh(self.conv1(x))
        x = F.tanh(F.max_pool2d(self.conv2(x),2))
        x = F.tanh(F.max_pool2d(self.conv3(x),2)) 
        x = torch.flatten(x, start_dim=1)
        x = F.tanh(self.fc1(x))
        return x
    
    def get_activations(self, x):
        return self.get_activations_4(x)

class GuardCNN(nn.Module):
    def __init__(self):
        super(GuardCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1,1,28,28)
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, padding=1))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, padding=1))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = x.view((-1,3136))
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def get_activations(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, padding=1))
        #x = F.dropout(x, p=0.5, training=self.training)
        return torch.flatten(x, start_dim=1)
    
    
class CifarCNN(nn.Module):
    def __init__(self, input_channels = 3):
        super(CifarCNN, self).__init__()
        # Conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(8192, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(-1,3,32,32)
        x = self.conv1(x)
        x = self.conv2(x)
        #x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        #return F.softmax(x, dim=1)
        return x

    def get_activations(self, x):
        x = self.conv1(x)
        return torch.flatten(x, start_dim=1)

class Net(nn.Module):  
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class CifarCNN2(nn.Module):
    def __init__(self, input_channels=3, fc_input=1024):
        super(CifarCNN2, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(fc_input, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = F.relu(self.conv1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_activations(self, x):
        x = x.view(-1, 3, 32, 32)
        x = F.relu(self.conv1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.max_pool2d(self.conv2(x), 2)
        return torch.flatten(x, start_dim=1)

    # def get_activations_2(self, x):
    #    x = F.relu(self.conv1(x))
    #    #x = F.dropout(x, p=0.5, training=self.training)
    #    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    #    x = F.dropout(x, p=0.5, training=self.training)
    #    x = F.max_pool2d(self.conv3(x),2)
    #    return x
