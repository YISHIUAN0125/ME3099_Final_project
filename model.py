import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)
    

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks=9):
        super().__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_dim, 64, kernel_size=7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True),
                 
                 nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                 nn.InstanceNorm2d(128),
                 nn.ReLU(inplace=True),

                 nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                 nn.InstanceNorm2d(256),
                 nn.ReLU(inplace=True),
            ]
        
        for _ in range(num_blocks):
            model += [ResnetBlock(256)]
        
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_dim, kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        model = [nn.Conv2d(input_dim, 64, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True),
                 
                 nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                 nn.InstanceNorm2d(128),
                 nn.LeakyReLU(0.2, inplace=True),

                 nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                 nn.InstanceNorm2d(256),
                 nn.LeakyReLU(0.2, inplace=True),

                 nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
                 nn.InstanceNorm2d(512),
                 nn.LeakyReLU(0.2, inplace=True),

                 nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    gen = Generator(input_dim=3, output_dim=3)
    disc = Discriminator(input_dim=3)
    x = torch.randn(1, 3, 256, 256)
    y = gen(x)
    d = disc(y)

    print(f'Generator shape: {y.shape}')
    print(f'Discriminator shape: {d.shape}')