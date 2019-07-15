from mxnet import nd
from mxnet.gluon import nn


class ShuffleLayer(nn.HybridBlock):
    def __init__(self, groups, **kwargs):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def hybrid_forward(self, F, x):
        x = F.reshape(x, (0, -4, self.groups, -1, 0, 0))
        x = F.swapaxes(x, 1, 2)
        x = F.reshape(x, (0, -3, -2))
        return x

class ShuffleUnitC(nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        super(ShuffleUnitC, self).__init__()
        branch_channels = channels//2
        with self.name_scope():
            self.branch = nn.HybridSequential()
            self.branch.add(nn.Conv2D(channels=branch_channels, kernel_size=(1, 1)))
            self.branch.add(nn.BatchNorm())
            self.branch.add(nn.Activation(activation='relu'))
            self.branch.add(nn.Conv2D(channels=branch_channels, kernel_size=(3, 3),
                                      strides=(1, 1), padding=(1, 1), groups=branch_channels))
            self.branch.add(nn.BatchNorm())
            self.branch.add(nn.Conv2D(channels=branch_channels, kernel_size=(1, 1)))
            self.branch.add(nn.BatchNorm())
            self.branch.add(nn.Activation(activation='relu'))
            
            self.channel_shuffle = ShuffleLayer(groups=2)

    def hybrid_forward(self, F, x):
        b1, b2 = F.split(x, axis=1, num_outputs=2)
        b1 = self.branch(b1)
        out = F.concat(b1, b2, dim=1)
        out = self.channel_shuffle(out)
        return out


class ShuffleUnitD(nn.HybridBlock):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ShuffleUnitD, self).__init__()
        branch_channels = out_channels//2
        with self.name_scope():
            # branch 1
            self.branch1 = nn.HybridSequential()
            self.branch1.add(nn.Conv2D(channels=branch_channels, kernel_size=(1, 1)))
            self.branch1.add(nn.BatchNorm())
            self.branch1.add(nn.Activation(activation='relu'))
            self.branch1.add(nn.Conv2D(channels=branch_channels, kernel_size=(3, 3),
                                       strides=(2, 2), padding=(1, 1), groups=branch_channels))
            self.branch1.add(nn.BatchNorm())
            self.branch1.add(nn.Conv2D(channels=branch_channels, kernel_size=(1, 1)))
            self.branch1.add(nn.BatchNorm())
            self.branch1.add(nn.Activation(activation='relu'))

            # branch 2
            self.branch2 = nn.HybridSequential()
            self.branch2.add(nn.Conv2D(channels=branch_channels, kernel_size=(3, 3),
                                       strides=(2, 2), padding=(1, 1), groups=branch_channels))
            self.branch2.add(nn.BatchNorm())
            self.branch2.add(nn.Conv2D(channels=branch_channels, kernel_size=(1, 1)))

            self.channel_shuffle = ShuffleLayer(groups=2)

    def hybrid_forward(self, F, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = F.concat(b1, b2, dim=1)
        out = self.channel_shuffle(out)
        
        return out


class ShuffleNetV2(nn.HybridBlock):
    def __init__(self, num_classes, channels, **kwargs):
        super(ShuffleNetV2, self).__init__()
        self.num_classes = num_classes

        with self.name_scope():
            self.stage1 = nn.HybridSequential()
            self.stage1.add(nn.Conv2D(channels[0], kernel_size=(3, 3), strides=(2, 2),
                                      padding=(1, 1)))
            self.stage1.add(nn.Conv2D(channels[0], kernel_size=(3, 3), strides=(2, 2),
                                      padding=(1, 1)))

            self.stage2 = nn.HybridSequential()
            self.stage2.add(ShuffleUnitD(in_channels=channels[0], out_channels=channels[1]))
            for _ in range(3):
                self.stage2.add(ShuffleUnitC(channels=channels[1]))

            self.stage3 = nn.HybridSequential()
            self.stage3.add(ShuffleUnitD(in_channels=channels[1], out_channels=channels[2]))
            for _ in range(7):
                self.stage3.add(ShuffleUnitC(channels=channels[2]))

            self.stage4 = nn.HybridSequential()
            self.stage4.add(ShuffleUnitD(in_channels=channels[2], out_channels=channels[3]))
            for _ in range(3):
                self.stage4.add(ShuffleUnitC(channels=channels[3]))

            self.last_conv = nn.Conv2D(channels=channels[4], kernel_size=(1, 1))

            self.flatten = nn.Flatten()
            self.dense = nn.Dense(num_classes)

    def hybrid_forward(self, F, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.last_conv(x)
        x = self.flatten(x)
        out = self.dense(x)
            
        return out


complexity_channels = {
    0.5: [24, 48, 96, 192, 1024],
    1: [24, 116, 232, 464, 1024],
    1.5: [24, 176, 352, 704, 1024],
    2: [24, 244, 488, 976, 2048]
}


def shufflenetv2_0_5(num_classes):
    complexity = 0.5
    channels = complexity_channels[complexity]
    net = ShuffleNetV2(num_classes=1000, channels=channels)
    return net


if __name__ == '__main__':
    net = shufflenetv2_0_5(1000)
    net.initialize()

    x = nd.random.uniform(0, 1, (1, 24, 56, 56)) 
    output = net(x)
    print (output.shape)

