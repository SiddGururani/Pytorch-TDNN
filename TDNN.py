import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import math

"""Time Delay Neural Network as mentioned in the 1989 paper by Waibel et al. (Hinton) and the 2015 paper by Peddinti et al. (Povey)"""

class TDNN(nn.Module):
    def __init__(self, context, input_dim, output_dim, full_context = True):
        """
        Definition of context is the same as the way it's defined in the Peddinti paper. It's a list of integers.
        """
        super(TDNN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_width, self.context = self.get_kernel_width(context,full_context)
        self.full_context = full_context
        stdv = 1./math.sqrt(input_dim)
        self.kernel = nn.Parameter(torch.Tensor(output_dim, input_dim, self.kernel_width).normal_(0,stdv))
        self.bias = nn.Parameter(torch.Tensor(output_dim).normal_(0,stdv))

    def forward(self,x):
        """
        x is one batch of data
        x.size(): [batch_size, sequence_length, input_dim]
        sequence length is the length of the input spectral data (number of frames) or if already passed through the convolutional network, it's the number of learned features
        """
        context = Variable(torch.LongTensor(self.context))
        conv_out = self.special_convolution(x, self.kernel, context, self.bias)
        return F.relu(conv_out)

    @staticmethod
    def special_convolution(x, kernel, context, bias):

        input_size = x.size()
        assert len(input_size) == 3, 'Input tensor dimensionality is incorrect. Should be a 3D tensor'
        [batch_size, input_sequence_length, input_dim] = input_size
        x = x.transpose(1,2).contiguous()

        # Allocate memory for output
        valid_steps = range(-1*context.data[0], input_sequence_length - context.data[-1])
        xs = Variable(torch.Tensor(batch_size, kernel.size()[0], len(valid_steps)))

        for c, i in enumerate(valid_steps):
            features = torch.index_select(x, 2, context+i)
            xs[:,:,c] = F.conv1d(features, kernel, bias = bias)[:,:,0]
        return xs

    @staticmethod
    def get_kernel_width(context, full_context):
        if full_context:
            context = range(context[0],context[-1]+1)
        return len(context), context