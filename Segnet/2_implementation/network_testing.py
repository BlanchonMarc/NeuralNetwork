from segnet import *

# Execution
batch_size = 1
input_size = 8
num_classes = 8
learning_rate = 0.0001
nb = 64

input = autograd.Variable(torch.rand(batch_size, input_size, nb, nb))
target = autograd.Variable(torch.rand(batch_size, num_classes, nb, nb)).long()


model = segnet(in_channels=input_size, n_classes=num_classes)

opt = optim.Adam(params=model.parameters(), lr=learning_rate)


for epoch in range(2):
    out = model(input)

    # Loss definition - cross entropy
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, target[:, 0])

    # nll loss
    # loss = F.nll_loss(out, target[:, 0])

    print ('Loss : ' + str(loss.data))

    model.zero_grad()
    loss.backward()

    opt.step()
