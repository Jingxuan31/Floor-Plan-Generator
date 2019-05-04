import torch
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from models.networks import ResnetBlock

model = torch.load('./checkpoints/test_01/latest_net_G.pth')
#for k,v in model.items():
#    print k
#print model['model.16.conv_block.1.weight']

net0 = torch.nn.Sequential(OrderedDict([
    ('model.0', torch.nn.ReflectionPad2d((3, 3, 3, 3))),
#    ('model.1', torch.nn.Conv2d (3, 64, kernel_size=(7, 7), stride=(1, 1))),
#    ('model.2', torch.nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False)),
#    ('model.3', torch.nn.ReLU(inplace=True)),
#    ('model.4', torch.nn.Conv2d (64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
#    ('model.5', torch.nn.InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False)),
#    ('model.6', torch.nn.ReLU(inplace=True)),
#    ('model.7', torch.nn.Conv2d (128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
#    ('model.8', torch.nn.InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False)),
#    ('model.9', torch.nn.ReLU(inplace=True)),
#    ('model.10', torch.nn.Conv2d (256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
#    ('model.11', torch.nn.InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False)),
#    ('model.12', torch.nn.ReLU(inplace=True)),
#    ('model.13', torch.nn.Conv2d (512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
#    ('model.14', torch.nn.InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False)),
#    ('model.15', torch.nn.ReLU(inplace=True)),
#    ('model.16', ResnetBlock(1024, 'reflect', torch.nn.InstanceNorm2d)),
#    ('model.17', ResnetBlock(1024, 'reflect', torch.nn.InstanceNorm2d)),
#    ('model.18', ResnetBlock(1024, 'reflect', torch.nn.InstanceNorm2d)),
#    ('model.19', ResnetBlock(1024, 'reflect', torch.nn.InstanceNorm2d)),
#    ('model.20', ResnetBlock(1024, 'reflect', torch.nn.InstanceNorm2d)),
#    ('model.21', ResnetBlock(1024, 'reflect', torch.nn.InstanceNorm2d)),
#    ('model.22', ResnetBlock(1024, 'reflect', torch.nn.InstanceNorm2d)),
#    ('model.23', ResnetBlock(1024, 'reflect', torch.nn.InstanceNorm2d)),
#    ('model.24', ResnetBlock(1024, 'reflect', torch.nn.InstanceNorm2d)),
#    ('model.25', torch.nn.ConvTranspose2d (1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))),
#    ('model.26', torch.nn.InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False)),
#    ('model.27', torch.nn.ReLU(inplace=True)),
#    ('model.28', torch.nn.ConvTranspose2d (512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))),
#    ('model.29', torch.nn.InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False)),
#    ('model.30', torch.nn.ReLU(inplace=True)),
#    ('model.31', torch.nn.ConvTranspose2d (256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))),
#    ('model.32', torch.nn.InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False)),
#    ('model.33', torch.nn.ReLU(inplace=True)),
#    ('model.34', torch.nn.ConvTranspose2d (128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))),
#    ('model.35', torch.nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False)),
#    ('model.36', torch.nn.ReLU(inplace=True)),
#    ('model.37', torch.nn.ReflectionPad2d((3, 3, 3, 3))),
#    ('model.38', torch.nn.Conv2d (64, 3, kernel_size=(7, 7), stride=(1, 1))),
#    ('model.39', torch.nn.Tanh()),
    ]))

#params = net0.state_dict()
#print params['model.19.conv_block.1.weight']
net0.load_state_dict(model,False)
#params = net0.state_dict()
#for k,v in params.items():
#    print k
#print params['model.19.conv_block.1.weight']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
testset = torchvision.datasets.ImageFolder(root='./datasets/test01/lm_test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

for data in testloader:
    images, labels = data
    outputs = net0(Variable(images))

for i in range(outputs.size(1)):
    torchvision.utils.save_image(torch.split(outputs,1,1)[i].data, './layer/layer0/'+str(i)+'.jpg', normalize=False, padding=0)
print torch.split(outputs,1,1)[0].data
