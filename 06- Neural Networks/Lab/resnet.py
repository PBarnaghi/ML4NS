import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ResNet block that will be used multiple times in the resnet model
class ResBlock(nn.Module):
    def __init__(self,
        input_dim:int, 
        input_channels:int,
        out_channels:int,
        out_dim:int,
        kernel_size:int=3,
        dropout_rate:float=0.2,
        ):
        super(ResBlock, self).__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.out_dim = out_dim
        
        self.x1 = nn.Sequential(
            nn.Conv1d(
                input_channels, 
                out_channels,
                kernel_size=kernel_size,
                bias=False,
                padding='same',
                ),
            nn.BatchNorm1d(
                num_features=out_channels,
                affine=True,
                ),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            )

        self.x2 = nn.Sequential(
            nn.Conv1d(
                in_channels=out_channels, 
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=False,
                stride=input_dim//out_dim,
                )
            )
        
        self.y1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding='same',
                bias=False,
                )
            )
        
        self.xy1 = nn.Sequential(
            nn.BatchNorm1d(
                num_features=out_channels,
                affine=False,
                ),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            )

        return

    # resizing the skip connection if needed, and then using 1d Convolution
    def _skip_connection(self, y):
        downsample = self.input_dim//self.out_dim
        if downsample > 1:

            same_pad = np.ceil(
                0.5*((y.size(-1)//self.out_dim)*(self.out_dim-1) - y.size(-1) + downsample)
                )
            if same_pad < 0:
                same_pad = 0
            y = nn.functional.pad(y, (int(same_pad), int(same_pad)), "constant", 0)
            y = nn.MaxPool1d(
                kernel_size=downsample,
                stride=downsample,
                )(y)
        
        elif downsample == 1:
            pass
        else:
            raise ValueError("Size of input should always decrease.")
        y = self.y1(y)
        
        return y

    def forward(self, inputs):
        x, y = inputs

        # y
        y = self._skip_connection(y)

        # x
        x = self.x1(x)
        same_pad = np.ceil(
            0.5*((x.size(-1)//self.out_dim)*(self.out_dim-1) - x.size(-1) + self.kernel_size)
            )
        if same_pad < 0:
            same_pad = 0
        x = nn.functional.pad(x, (int(same_pad), int(same_pad)), "constant", 0)
        x = self.x2(x)

        # xy
        xy = x + y
        y = x
        xy = self.xy1(xy)

        return [xy, y]

# main resnet model, made of 4 resnet blocks
class ResNet(nn.Module):
    def __init__(
        self,
        input_dim:int=4096,
        input_channels:int=64,
        n_output:int=10,
        kernel_size:int=16,
        dropout_rate:float=0.2,
        ):
        '''
        Model with 4 :code:`ResBlock`s, in which
        the number of channels increases linearly
        and the output dimensions decreases
        exponentially. This model will
        require the input dimension to be of at least 
        256 in size. This model is designed for sequences,
        and not images. The expected input is of the type::

            [n_batches, n_filters, sequence_length]


        Examples
        ---------
        
        .. code-block::
        
            >>> model = ResNet(
                    input_dim=4096,
                    input_channels=64,
                    kernel_size=16,
                    n_output=5,
                    dropout_rate=0.2,
                    )
            >>> model(
                    torch.rand(1,64,4096)
                    )
            tensor([[0.3307, 0.4782, 0.5759, 0.5214, 0.6116]], grad_fn=<SigmoidBackward0>)
        
        
        Arguments
        ---------
        
        - input_dim: int, optional:
            The input dimension of the input. This
            is the size of the final dimension, and 
            the sequence length.
            Defaults to :code:`4096`.
        
        - input_channels: int, optional:
            The number of channels in the input.
            This is the second dimension. It is the
            number of features for each sequence element.
            Defaults to :code:`64`.
        
        - n_output: int, optional:
            The number of output classes in 
            the prediction. 
            Defaults to :code:`10`.
        
        - kernel_size: int, optional:
            The size of the kernel filters
            that will act over the sequence. 
            Defaults to :code:`16`.       
        
        - dropout_rate: float, optional:
            The dropout rate of the ResNet
            blocks. This should be a value
            between :code:`0` and  :code:`1`.
            Defaults to :code:`0.2`.     
        
        '''
        super(ResNet, self).__init__()

        self.x1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                stride=1,
                padding='same',
                bias=False,
                ),
            nn.BatchNorm1d(
                num_features=input_channels,
                affine=False,
                ),
            nn.ReLU(),
        )

        self.x2 = nn.Sequential(
            ResBlock(
                input_dim=input_dim, # 4096
                input_channels=input_channels, # 64
                out_channels=2*input_channels//1, # 128
                kernel_size=kernel_size, # 16
                out_dim=input_dim//4, # 1024,
                dropout_rate=dropout_rate,
                ),

            ResBlock(
                input_dim=input_dim//4, # 1024
                input_channels=2*input_channels//1, # 128
                out_channels=3*input_channels//1, # 192
                kernel_size=kernel_size, # 16
                out_dim=input_dim//16, # 256
                dropout_rate=dropout_rate,
                ),

            ResBlock(
                input_dim=input_dim//16, # 256
                input_channels=3*input_channels//1, # 192
                out_channels=4*input_channels//1, # 256
                kernel_size=kernel_size, # 16
                out_dim=input_dim//64, # 64
                dropout_rate=dropout_rate,
                ),

            ResBlock(
                input_dim=input_dim//64, # 64
                input_channels=4*input_channels//1, # 256
                out_channels=5*input_channels//1, # 320
                kernel_size=kernel_size, # 16
                out_dim=input_dim//256, # 16
                dropout_rate=dropout_rate,
                ),
            )

        self.x3 = nn.Flatten() # flattens the data
        self.x4 = nn.Sequential(        
            nn.Linear(
                (input_dim//256) * (5*input_channels//1),
                n_output,
                )
            )

    def forward(self, x):
        
        x = self.x1(x)
        x, _ = self.x2([x,x])
        x = self.x3(x)
        x = self.x4(x)

        return x