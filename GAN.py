# GAN for Image Generation using CIFAR-10 Dataset

"""
- This project implements a Generative Adversarial Network (GAN)
- GAN consists of two models:
    1. Generator → creates fake images
    2. Discriminator → distinguishes real vs fake images
- Both models compete with each other (adversarial training)
- Dataset used: CIFAR-10 (32x32 color images)
"""
"""
- This project uses the CIFAR-10 dataset for training the GAN model
- Dataset will be automatically downloaded when you run the code
- Make sure you have an active internet connection
- The dataset will be stored in the './data' folder

Note:
If you already have the dataset, set:
download=False
"""
#========================================================================================================
#Load Libraries
#========================================================================================================
import torch                                           # PyTorch framework
import torch.nn as nn                                  # neural network layers
import torch.optim as optim                            # optimizers
import torchvision                                     # datasets & utilities
from torchvision import datasets, transforms
import matplotlib.pyplot as plt                        # visualization
import numpy as np                                     # numerical operations 
#========================================================================================================
#Device Configuration
""" 
Explanation:
- Uses GPU if available (faster training)
- Otherwise falls back to CPU
"""
#========================================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#========================================================================================================
#Hyperparameters
""" 
Explanation:
- latent_dim → size of random noise vector
- lr → learning rate
- beta1, beta2 → Adam optimizer parameters
- num_epochs → number of training iterations
"""
#========================================================================================================
latent_dim = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_epochs = 10
#========================================================================================================
#Data Preprocessing
""" 
Explanation:
- Converts images to tensor
- Normalizes pixel values to range [-1, 1]
- Required because Generator uses Tanh activation
"""
#========================================================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
#========================================================================================================
#Load Dataset
""" 
Explanation:
- CIFAR-10 dataset:
    - 60,000 images (32×32 RGB)
    - 10 classes (airplane, car, bird, etc.)
- Automatically downloads dataset
"""
#========================================================================================================
train_dataset = datasets.CIFAR10(root =  './data', \
    train = True, download = True, transform = transform)
#========================================================================================================
#DataLoader
""" 
Explanation:
- Loads data in batches
- shuffle=True → improves training
"""
#========================================================================================================
dataloader = torch.utils.data.DataLoader(train_dataset, \
                                         batch_size = 32, shuffle = True)
#========================================================================================================
#Generator Model
""" 
Input → Random noise (latent vector)

Steps:
1. Linear layer → expands noise
2. Unflatten → converts to feature map
3. Upsampling → increases image size
4. Conv layers → refine image
5. BatchNorm → stabilizes training
6. Tanh → outputs image in [-1,1]

Output → Fake image (3 × 32 × 32)
"""
#========================================================================================================
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear( latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128,8,8)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128, momentum = 0.78),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.78),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    def forward(self, z):
        return self.model(z)
##========================================================================================================
#Discriminator Model
""" 
Input → Image (real or fake)

Steps:
1. Conv layers → extract features
2. LeakyReLU → better gradient flow
3. Dropout → prevents overfitting
4. BatchNorm → stabilizes learning
5. Flatten → convert to vector
6. Linear → output single value
7. Sigmoid → probability (real or fake)

Output:
0 → Fake
1 → Real
"""
#========================================================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.25),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ZeroPad2d((0, 1, 0, 1)),
        nn.BatchNorm2d(64, momentum=0.82),
        nn.LeakyReLU(0.25),
        nn.Dropout(0.25),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(128, momentum=0.82),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.25),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256, momentum=0.8),
        nn.LeakyReLU(0.25),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(256 * 5 * 5, 1),
        nn.Sigmoid()
    )

    def forward(self, img):
        validity = self.model(img)
        return validity
#========================================================================================================
#Model Initialization
#========================================================================================================
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
#========================================================================================================
#Loss Function
""" 
Explanation:
- Binary Cross Entropy Loss
- Used for real vs fake classification
"""
#========================================================================================================
adversarial_loss = nn.BCELoss()

#========================================================================================================
#Optimizers
#========================================================================================================
optimizer_G = optim.Adam(generator.parameters()\
                         , lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters()\
                         , lr=lr, betas=(beta1, beta2))

#========================================================================================================
#Training Loop
""" 
- real_images → actual dataset images
- valid → label = 1
- fake → label = 0
"""
#========================================================================================================
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
       
        real_images = batch[0].to(device) 
       
        valid = torch.ones(real_images.size(0), 1, device=device)
        fake = torch.zeros(real_images.size(0), 1, device=device)
       
        real_images = real_images.to(device)

        optimizer_D.zero_grad()
       
        z = torch.randn(real_images.size(0), latent_dim, device=device)
      
        fake_images = generator(z)

        real_loss = adversarial_loss(discriminator\
                                     (real_images), valid)
        fake_loss = adversarial_loss(discriminator\
                                     (fake_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
    
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
      
        gen_images = generator(z)
        
        g_loss = adversarial_loss(discriminator(gen_images), valid)
        g_loss.backward()
        optimizer_G.step()
       
        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}]\
                        Batch {i+1}/{len(dataloader)} "
                f"Discriminator Loss: {d_loss.item():.4f} "
                f"Generator Loss: {g_loss.item():.4f}"
            )
#========================================================================================================
#Image Generation
#========================================================================================================
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            generated = generator(z).detach().cpu()
            grid = torchvision.utils.make_grid(generated,\
                                        nrow=4, normalize=True)
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.axis("off")
            plt.show()
#========================================================================================================
#Summary
#========================================================================================================
"""
- This project implements a GAN using PyTorch
- Generator creates fake images from noise
- Discriminator classifies real vs fake images
- Both models improve through adversarial training
- Uses CIFAR-10 dataset for training
- Outputs realistic synthetic images over time
"""