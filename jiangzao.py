from torch import nn

from torchvision import transforms
import numpy as np

# #另一个剑招代码
# class DenoiseAutoEncoder(nn.Module):
#     def __init__(self):
#         super(DenoiseAutoEncoder, self).__init__()
#         self.Encoder = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),#[,64,96,96]
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1,  padding=11),#[,64,96,96]
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1,  padding=11),#[,64,96,96]
#             nn.ReLU(),
#             nn.MaxPool2d(2,2),#[,64,48,48]
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64,128,3,1,1),#[,128,48,48]
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128,128,3,1,1),#[,128,48,48]
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128,256,3,1,1),#[,256,48,48]
#             nn.ReLU(),
#             nn.MaxPool2d(2,2),#[,256,24,24]
#             nn.BatchNorm2d(256)
#         )
#         self.Decoder=nn.Sequential(
#             nn.ConvTranspose2d(256,128,3,1,1),#[,256,24,24]
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.ConvTranspose2d(128,128,3,2,1,1),#[,128,48,48]
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.ConvTranspose2d(128,64,3,1,1),#[,64,48,48]
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.ConvTranspose2d(64,32,3,1,1),#[,32,48,48]
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.ConvTranspose2d(32,32,3,1,1),#[,32,48,48]
#             nn.ConvTranspose2d(32,16,3,2,1,1),#[,16,96,96]
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.ConvTranspose2d(16,3,3,1,1),#[,3,96,96]
#             nn.Sigmoid()
#         )
#     def forward(self,x):
#         encoder=self.Encoder(x)
#         decoder=self.Decoder(encoder)
#         return encoder,decoder

from torch.nn import *
class jiangzaonet(nn.Module):
    def __init__(self, channels=3):
        super(jiangzaonet, self).__init__()
        # self.conv_layer_1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=(3, 3), padding='same')
        # self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same')

        self.conv_layer_1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=(2, 2), padding='same')
        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2), padding='same')

        self.conv_layer_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same')
        self.conv_layer_4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same')
        self.conv_layer_5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding='same')

        self.deconv_layer_5 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        # deconv_layer_5 = Add(name="add_1")([conv_layer_4, deconv_layer_5])
        self.deconv_layer_4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.deconv_layer_3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        # #deconv_layer_3 = Add(name="add_2")([conv_layer_2, deconv_layer_3])
        self.deconv_layer_2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.deconv_layer_1 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(3, 3), padding=(1, 1))
        # out = Add(name="add_3")([input_0, deconv_layer_1])

    def forward(self, x):
        #编码
        x1 = self.conv_layer_1(x)
        x2 = self.conv_layer_2(x1)
        x3 = self.conv_layer_3(x2)
        x4 = self.conv_layer_4(x3)
        x5 = self.conv_layer_5(x4)
        #解码
        d5 = self.deconv_layer_5(x5)
        d5 = torch.add(x4, d5)
        d4 = self.deconv_layer_4(d5)
        d3 = self.deconv_layer_3(d4)
        d3 = torch.add(x2, d3)
        d2 = self.deconv_layer_2(d3)
        d1 = self.deconv_layer_1(d2)
        d1 = torch.add(d1, x)
        return d1

class zong(nn.Module):
    def __init__(self,jiangzao, mainmodel, device):
        super(zong,self).__init__()
        self.jiangzao = jiangzao.to(device)
        self.mainmodel = mainmodel.to(device)
    def forward(self,x):

        x = self.jiangzao(x)
        x = self.mainmodel(x)
        return x

import torch
if __name__ == "__main__":
    # model = jiangzaonet(channels=3)
    # input = torch.ones((1,3,224,224))
    # print(input.shape)
    # print(model(input).shape)

    import torch
    from PIL import Image

    # Load input image
    input_image = Image.open('C:\\zlg\\2.jpg')
    input_tensor = transforms.ToTensor()(input_image).unsqueeze(0)

    # # Instantiate the model
    model = jiangzaonet()
    #
    # # Reconstruct the image
    output_tensor = model(input_tensor)
    # print(input_tensor.shape)
    print(output_tensor.shape)

    # Convert output tensor to numpy array
    output_array = input_tensor.squeeze(0).detach().cpu().numpy()
    output_array = np.transpose(output_array, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)

    # Convert numpy array to PIL image
    output_image = Image.fromarray((output_array * 255).astype(np.uint8))

    # Save or display the reconstructed image
    # output_image.save('output_image.jpg')
    output_image.show()
