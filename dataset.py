import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define the directory where you want to save the dataset
data_dir = 'mnist_data'

# Define the transformation to convert the images to tensors
transform = transforms.ToTensor()

# Download the training set
train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)

# Download the test set
test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

print("MNIST dataset downloaded successfully!")