import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, "imgH has to be a multiple of 16"

        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1),  # conv1
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),          # 32x128 -> 16x64

            nn.Conv2d(64, 128, 3, 1, 1), # conv2
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),          # 16x64 -> 8x32

            nn.Conv2d(128, 256, 3, 1, 1), # conv3
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), # conv4
            nn.ReLU(True),
            nn.MaxPool2d((2,2), (2,1), (0,1)), # 8x32 -> 4x33

            nn.Conv2d(256, 512, 3, 1, 1), # conv5
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), # conv6
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2,2), (2,1), (0,1)), # 4x33 -> 2x34

            nn.Conv2d(512, 512, 2, 1, 0),  # conv7 kernel=2 no padding
            nn.ReLU(True)
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=nh,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.embedding = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        # x: (batch, channel=1, height, width)
        conv = self.cnn(x)  # [batch, 512, 1, width']
        b, c, h, w = conv.size()
        assert h == 1, "height after conv must be 1"
        conv = conv.squeeze(2)  # [batch, 512, width]
        conv = conv.permute(0, 2, 1)  # [batch, width, 512]

        rnn_out, _ = self.rnn(conv)  # [batch, width, nh*2]
        output = self.embedding(rnn_out)  # [batch, width, nclass]

        # output: logit sequence for CTC loss
        return output.log_softmax(2)  # for CTC loss: log prob on dim=2
