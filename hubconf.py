from torch import nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1,int(x.nelement() / x.shape[0]))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

def lenet5(pretrained=False,**kwargs):
    lenet=LeNet5()
    if pretrained:
        lenet.load_state_dict(torch.load("./lenet5_model.pt"))
    return lenet

def load_model(model_name,pretrained=False,**kwargs):
    if model_name=="Lenet5":
        return lenet5(pretrained=pretrained,**kwargs)
    elif model_name=="vgg16":
        raise ValueError("Can't find vgg16")
    else:
        raise ValueError("Can't find model you input")