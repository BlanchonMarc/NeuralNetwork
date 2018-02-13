import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage
import torch.optim as optim
from segnet import *
from dataload import *


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor,
                          torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


input_transform = Compose([
    CenterCrop(256),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform = Compose([
    CenterCrop(256),
    ToLabel(),
    Relabel(255, 21),
])

workers = 20
batch_size = 1
n_classes = 22

loader = DataLoader(DatasetLoader('/data/scene-segmentation/CamVid/', input_transform=input_transform,
                               target_transform=target_transform),
                    num_workers=workers,
                    batch_size=batch_size, shuffle=False)


model = segnet(in_channels=3, n_classes=n_classes)
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
model.cuda()
learning_rate = 0.0001

opt = optim.Adam(params=model.parameters(), lr=learning_rate)

#weight = nn.init.xavier_normal(torch.Tensor(n_classes, 1).cuda(),
#gain=nn.init.calculate_gain('relu'))
weight = torch.ones(n_classes).cuda()
weight[0] = 0

criterion = nn.NLLLoss2d(weight)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

cuda_activated = True

for epoch in range(100):

    running_loss = 0.0
    for step, (images, targets) in enumerate(loader):
        if cuda_activated:
            images = images.cuda()
            targets = targets.cuda()

        images = autograd.Variable(images)
        targets = autograd.Variable(targets)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(F.log_softmax(outputs, dim=1), targets[:, 0])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = loss.data[0]
        print(running_loss)
        if step % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')
