import torch
from torchvision import datasets, transforms
import torch.nn as nn

# Define a transform to convert grayscale images to RGB and resize them to 32x32
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize((32, 32)),  # Resize images to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Assuming that you have your test data in a folder named 'test_data'
test_data = datasets.ImageFolder(r'data\mnist_USPS\testB', transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

if torch.cuda.is_available:
    device = torch.device("cuda")
else:
     device = torch.device( "cpu")
     
device = "cpu"

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class LeNet(nn.Module): #for 32
    def __init__(self, input_nc):  # Changed 'init' to '__init__'
        super(LeNet, self).__init__()

        sequence = [
            nn.Conv2d(input_nc, 20, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            Flatten(),

            nn.Linear(50*5*5, 500),
            #nn.Linear(50, 500),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(500, 10)
        ]
        self.net = nn.Sequential(*sequence)
        
    def forward(self, input, d_feat=False):
        """Standard forward."""
        self.out = self.net(input)
        return self.out


# Load your pre-trained model
model = LeNet(3)
#model.load_state_dict(torch.load(r'models\pretrain\lenet_mnist_acc_97.5000.pt', map_location = device))

#For trained function
model_state = torch.load('80_net_C_B.pth' , map_location = "cpu")
model.load_state_dict(model_state)


model.eval()


correct = 0
total = 0

# No need to track gradients for evaluation, saves memory and computations
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        #print(images.shape)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on test images: %.2f %%' % (100 * correct / total))
