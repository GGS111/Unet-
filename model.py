import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

#First class for double convolutions
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList() #Module for decoding with enhancement
        self.downs = nn.ModuleList() #Module for encoding with lowering
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #Max pool for lowering

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature #Re-assigning the size of the feature map

        # Up part of UNET
        for feature in reversed(features):
            #The first part is responsible for Upsampling, reverse maxpulling. Because of we will have concatenation
            #of feature maps from skip.connection, then multiply the first value by 2
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        #The downest string
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        #Final convolution with a 1x1 core
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        #do all the transformations to the lower level, and collect the skip.connection
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        #Transformations of the lowest level
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        #Step = 2, becuase of we have double convolution and upsampling
        for idx in range(0, len(self.ups), 2):
            #Upsampling
            x = self.ups[idx](x)
            #Take the skip connection and make the index by 2 to take it correctly
            skip_connection = skip_connections[idx//2]
            
            '''
            Checking for the coincidence of the size of the encoding with the decoding before contacting.
            It is important to understand here that in the classical model, the size of the image on decoding is less than on encoding
            Therefore, we make a resize of 3 4 encoding channels, that is, we resize the image.
            PS: here did not reduce the size of the image in the double convolution, so there is comparison decoding and encoding 
            for same size here, because of that the output it does not lose the size of the image as a result
            '''
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            #Concatenating feature maps
            concat_skip = torch.cat((skip_connection, x), dim=1)
            #Aapply a doublconv to the concatenated element. Note: take the indices 1/3/5, etc.
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)