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

class ShuffleUnitA(nn.HybridBlock):
    pass


class ShuffleUnitB(nn.HybridBlock):
    def __init__(self, channels, groups, **kwargs):
        super(ShuffleUnitB, self).__init__()
        bottlenet_channels = channels//4
        with self.name_scope():
            self.gconv1 = nn.Conv2D(channels=bottlenet_channels, kernel_size=(1, 1), groups=groups)
            self.bn1 = nn.BatchNorm()
            self.relu1 = nn.Activation(activation='relu')
            self.channel_shuffle = ShuffleLayer(groups=groups)
            self.dwconv = nn.Conv2D(channels=bottlenet_channels, kernel_size=(3, 3),
                                    strides=(1, 1), padding=(1, 1), groups=bottlenet_channels)
            self.bn2 = nn.BatchNorm()
            self.gconv2 = nn.Conv2D(channels=channels, kernel_size=(1, 1), groups=groups)
            self.bn3 = nn.BatchNorm()

    def hybrid_forward(self, F, x):
        out = self.gconv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.channel_shuffle(out)
        out = self.dwconv(out)
        out = self.bn2(out)
        out = self.gconv2(out)
        out = self.bn3(out)
        out = F.elemwise_add(x, out)
        
        return F.relu(out)


class ShuffleUnitC(nn.HybridBlock):
    def __init__(self, in_channels, out_channels, groups, small_channels=False, **kwargs):
        super(ShuffleUnitC, self).__init__()
        bottlenet_channels = out_channels//4
        mid_channels = out_channels - in_channels
        with self.name_scope():
            if small_channels:
                self.gconv1 = nn.Conv2D(channels=bottlenet_channels, kernel_size=(1, 1))
            else:
                self.gconv1 = nn.Conv2D(channels=bottlenet_channels, kernel_size=(1, 1), groups=groups)
            self.bn1 = nn.BatchNorm()
            self.relu1 = nn.Activation(activation='relu')
            self.channel_shuffle = ShuffleLayer(groups=groups)
            self.dwconv = nn.Conv2D(channels=bottlenet_channels, kernel_size=(3, 3),
                                    strides=(2, 2), padding=(1, 1), groups=bottlenet_channels)
            self.bn2 = nn.BatchNorm()
            self.gconv2 = nn.Conv2D(channels=mid_channels, kernel_size=(1, 1), groups=groups)
            self.bn3 = nn.BatchNorm()

            self.avg_pool = nn.AvgPool2D(pool_size=(3, 3), strides=(2, 2), padding=(1, 1))

    def hybrid_forward(self, F, x):
        out = self.gconv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.channel_shuffle(out)
        out = self.dwconv(out)
        out = self.bn2(out)
        out = self.gconv2(out)
        out = self.bn3(out)
        x = self.avg_pool(x)
        out = F.concat(x, out, dim=1)
        
        return F.relu(out)


class ShuffleNet(nn.HybridBlock):
    def __init__(self, num_classes, groups, channels, multiplier=1.0, **kwargs):
        super(ShuffleNet, self).__init__()
        self.num_classes = num_classes
        self.channels = [channel*multiplier for channel in channels]

        with self.name_scope():
            self.stage1 = nn.HybridSequential()
            self.stage1.add(nn.Conv2D(channels[0], kernel_size=(3, 3), strides=(2, 2),
                                      padding=(1, 1)))
            self.stage1.add(nn.Conv2D(channels[0], kernel_size=(3, 3), strides=(2, 2),
                                      padding=(1, 1)))

            self.stage2 = nn.HybridSequential()
            self.stage2.add(ShuffleUnitC(in_channels=channels[0], out_channels=channels[1],
                                         groups=groups, small_channels=True))
            for _ in range(3):
                self.stage2.add(ShuffleUnitB(channels=channels[1], groups=groups))

            self.stage3 = nn.HybridSequential()
            self.stage3.add(ShuffleUnitC(in_channels=channels[1], out_channels=channels[2],
                                         groups=groups))
            for _ in range(7):
                self.stage3.add(ShuffleUnitB(channels=channels[2], groups=groups))

            self.stage4 = nn.HybridSequential()
            self.stage4.add(ShuffleUnitC(in_channels=channels[2], out_channels=channels[3],
                                         groups=groups))
            for _ in range(3):
                self.stage4.add(ShuffleUnitB(channels=channels[3], groups=groups))

            self.flatten = nn.Flatten()
            self.dense = nn.Dense(num_classes)

    def hybrid_forward(self, F, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.flatten(x)
        out = self.dense(x)
            
        return out


group_channels = {
    1: [24, 144, 288, 576],
    2: [24, 200, 400, 800],
    3: [24, 240, 480, 960],
    4: [24, 272, 544, 1088],
    8: [24, 384, 768, 1536]
}


def shufflenet0_5_g3(num_classes):
    multiplier = 0.5
    groups = 3
    channels = group_channels[groups]
    net = ShuffleNet(num_classes=1000, groups=groups, channels=channels, multiplier=multiplier)
    return net


if __name__ == '__main__':
    groups = 3
    channels = group_channels[groups]
    net = get_shufflenet0_5(1000)
    net.initialize()

    x = nd.random.uniform(0, 1, (1, 3, 224, 224)) 
    output = net(x)
    print (output.shape)

