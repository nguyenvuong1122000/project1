import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

use_cuda = False

class backbone_CNN(nn.Module):
    def __init__(self, in_channels = 3):
        super(backbone_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding = 1, stride=1)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size=3, padding = 1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1, stride=2)
        self.linear1 = nn.Linear(in_features=2048, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=5)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)

        return F.softmax(self.linear2(self.linear1(x)))



class LSTM_Cell(nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super(LSTM_Cell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_features = 4

        self.W = nn.Linear(in_features=self.input_channels, out_features=self.input_channels)
        self.Wy = nn.Linear(int(self.input_channels + self.hidden_channels), self.hidden_channels)
        self.Wi = nn.Linear(self.hidden_channels, self.hidden_channels,   bias=False)
        self.Wbi = nn.Linear(self.hidden_channels, self.hidden_channels,  bias=False)
        self.Wbf = nn.Linear(self.hidden_channels, self.hidden_channels,  bias=False)
        self.Wbc = nn.Linear(self.hidden_channels, self.hidden_channels,  bias=False)
        self.Wbo = nn.Linear(self.hidden_channels, self.hidden_channels,  bias=False)

        self.relu = nn.ReLU6()

        # print("Initializing weights of lstm by Xavier Init")
        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, h, c):
        print(x)
        x = self.W(x)
        y = torch.cat((x, h), 1)
        i = self.Wy(y)
        b = self.Wi(i)
        ci = torch.sigmoid(self.Wbi(b))
        cf = torch.sigmoid(self.Wbf(b))
        cc = cf * c + ci * self.relu(self.Wbc(b))
        co = torch.sigmoid(self.Wbo(b))
        ch = co * self.relu(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, feauture_size):
        #tao hidden state khi bat dau
        if use_cuda :
            return (Variable(torch.zeros(batch_size, feauture_size)).cuda(),
                    Variable(torch.zeros(batch_size, hidden)).cuda()
                    )
        else:
            return (Variable(torch.zeros(batch_size,  feauture_size)),
                    Variable(torch.zeros(batch_size, hidden, ))
                    )

class LSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, feauture_size, batch_size):
        super(LSTM, self).__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.cell = LSTM_Cell(self.input_channels, self.hidden_channels)
        (h, c) = self.cell.init_hidden(batch_size, hidden=self.hidden_channels, feauture_size=feauture_size)

        self.hidden_state = h
        self.cell_state = c

    def forward(self, input):
        new_h, new_c = self.cell(input, self.hidden_state, self.cell_state)
        self.hidden_state = new_h
        self.cell_state = new_c
        return self.hidden_state

    def detach_hidden(self):
        """
        Detaches hidden state and cell state of all the LSTM layers from the graph
        """
        self.hidden_state.detach_()
        self.cell_state.detach_()
class G2RL(nn.Module):
    #input to LSTM = 256 theo paper
    def __init__(self, input_embedding = 256):
        super(G2RL, self).__init__()
        self.backbone_CNN = backbone_CNN()
        # self.RNN_model = LSTM(input_embedding, feauture_size=256, batch_size=1, hidden_channels=256)
        #
        # self.MLP = nn.Sequential(
        #     nn.Linear(in_features=256, out_features=512),
        #     nn.ReLU(),
        #     nn.Linear(in_features=512, out_features=5),
        # )
    def forward(self, x):
        x = self.backbone_CNN(x)
        # print(x)
        # x = self.MLP(self.RNN_model(x))

        return x
    def init_xavier(self):
        pass

