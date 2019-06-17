import torch
from torchvision import models,transforms,datasets
from torch import nn
from torch import optim

def load_data(path):
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    vt_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=vt_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=vt_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    return trainloader, validloader, testloader, train_data.class_to_idx

def train(trainloader, validloader, class_to_idx , epochs, lr, arch, hidden_units):
    # Build and train your network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arch == 'densenet121' :
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif arch == 'densenet161' :
        model = models.densenet161(pretrained=True)
        input_size = 2208
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    else:
        return Null

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.3),
                                    nn.Linear(hidden_units,len(class_to_idx)),
                                    nn.LogSoftmax(dim=1))
    
    # Criterion
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= lr)
    model.to(device)

    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print(f'Epoch {epoch+1}/{epochs}.. '
                          f'Validation loss: {running_loss/print_every:.3f}.. '
                          f'Validation loss: {valid_loss/len(validloader):.3f}.. '
                          f'Validation accuracy: {accuracy/len(validloader):.3f}')
                running_loss = 0
                model.train()

    # TODO: Save the checkpoint 

    model.class_to_idx = class_to_idx
    torch.save(model, 'checkpoint.pth')
    #torch.save(model.state_dict(), filepath)
    #torch.save(checkpoint, 'checkpoint.pth')
    print("Model saved to 'checkpoint.pth'")
    
def test(testloader):
    print('Testing model')
    print('Loading checkpoint.pth')
    
    model_c = load_checkpoint('checkpoint.pth')

    # TODO: Do validation on the test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss = 0
    accuracy = 0
    model_c.to(device)
    model_c.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model_c.forward(inputs)

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print(f'Test accuracy: {accuracy/len(testloader):.3f}')
    
def load_checkpoint(filepath):

    return torch.load(filepath)