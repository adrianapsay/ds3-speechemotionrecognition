from torch import nn 

# the class needs to inherit from PyTorch's nn.Module
# use a CNN to perform classification on our MelSpectogram
class CNNNetwork(nn.Module):
    # NOTE: a better improvement would be to pass the paramters like
    # stride, padding, kernel_size in the constructor 
    
    def __init__(self):
        # we are making a VGG net
        # can perhaps make a better architecture better for your project
        super().__init__()
        
        # 4 conv blocks / flatten / linear / softmax
        # pytorch will pass the data through layers in Sequential
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=3,
                stride=1, 
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=3,
                stride=1, 
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
            
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=3,
                stride=1, 
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
            
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, 
                out_channels=128, 
                kernel_size=3,
                stride=1, 
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # compress the output from the high channels we have
        self.flatten = nn.Flatten()
        # below is the flattened length of the output 
        # from the last convolutional channel
        
        n_classes = 6
        self.linear = nn.Linear(128 * 5 * 4, n_classes)
        self.softmax = nn.Softmax(dim=1)
        
    # need to define how sequential should pass 
    # the data forawrd between layers that we just defined
    # in the sequential architecture
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits) 
        return predictions

    if __name__ == "__main__":
        cnn = CNNNetwork() # we can add arugments to the constructor
        # 1 -- channel (grayscale)
        # 64 -- frequency axis
        # 44 -- the time axis 
        print('testing')
        summary(cnn.cuda(), (1, 64, 44))
            
       
        