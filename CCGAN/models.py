import torch.nn as nn
import torch
import torch.nn.utils as utils


class Generator(nn.Module):
    def __init__(self, args, n_atoms_total, n_label_features, label_dim):
        super(Generator, self).__init__()
        self.n_label_features = n_label_features
        self.label_dim = label_dim
        self.latent_dim = args.latent_dim
        self.gen_channels_1 = args.gen_channels_1
        self.n_atoms_total = n_atoms_total

        # self.label_embedding = nn.Embedding(n_label_features, self.label_dim)

        self.label_proj = nn.Sequential(
            nn.Linear(self.n_label_features, self.label_dim),
            nn.ReLU(inplace=True),
        )
        

        self.l1 = nn.Sequential(nn.Linear(self.latent_dim + self.label_dim, self.gen_channels_1*self.n_atoms_total),nn.ReLU(True))
        self.map1 = nn.Sequential(nn.ConvTranspose2d(self.gen_channels_1,256,(1,3),stride = 1,padding=0),nn.ReLU(True))
        self.map2 = nn.Sequential(nn.ConvTranspose2d(256,512,(1,1),stride = 1,padding=0),nn.ReLU(True))
        self.map3 = nn.Sequential(nn.ConvTranspose2d(512,256,(1,1),stride = 1,padding=0),nn.ReLU(True)) 
        self.map4 = nn.Sequential(nn.ConvTranspose2d(256,1,(1,1),stride=1,padding=0)) 
        self.sigmoid = nn.Sigmoid() # NOTE: Removed sigmoid activation for the generator output

    def forward(self, noise, labels):

        label_proj = self.label_proj(labels)  # shape: (batch_size, label_dim)
        
        # Concatenate noise and label embedding
        gen_input = torch.cat([noise, label_proj], dim=1)

        # gen_input = noise # NOTE: OLD
        h = self.l1(gen_input)
        h = h.view(h.shape[0], self.gen_channels_1, self.n_atoms_total, 1)   # h.shape[0] is the current batch size 
        h = self.map1(h)
        h = self.map2(h)
        h = self.map3(h)
        h = self.map4(h)
        pos = self.sigmoid(h) # NOTE: Removed sigmoid activation for the generator output
        
        return pos  # torch.Size is (current_batch_size, 1, n_atoms_total, 3)


class CoordinateDiscriminator(nn.Module):
    def __init__(self, args, n_atoms_elements, n_label_features, label_dim):
        super(CoordinateDiscriminator, self).__init__()
        self.n_elements = len(n_atoms_elements)
        self.label_dim = label_dim
        self.n_label_features = n_label_features
        self.n_atoms_elements = n_atoms_elements
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (1,1), stride = 1, padding = 0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
            nn.LeakyReLU(0.2,inplace=True),
            )
        
        self.avgpool_elements = []
        for i in range(self.n_elements):
            self.avgpool_elements.append(nn.AvgPool2d(kernel_size = (self.n_atoms_elements[i],1)))

        # Add a label embedding for categorical labels:
        # self.label_proj = nn.Embedding(n_label_features, label_dim)

        # Use a linear layer to project the label to the same dimension as the output
        self.label_proj = nn.Sequential(
            nn.Linear(self.n_label_features, self.label_dim),
            nn.LeakyReLU(0.2,inplace=True),
        )

        self.feature_layer = nn.Sequential(
            nn.Linear(256*self.n_elements + self.label_dim, 1000),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1000,200),
            nn.LeakyReLU(0.2,inplace=True),
        )
        self.output = nn.Sequential(
            nn.Linear(200,10)
            )

    def forward(self, x, labels):
        B = x.shape[0]
        output = self.model(x)
        
        output_elements = []   # Stores the output that has been sliced based on the element type
        start = 0
        for i in range(self.n_elements):
            stop = start + self.n_atoms_elements[i]
            output_slice = output[:,:,start:stop,:]
            output_slice = self.avgpool_elements[i](output_slice)
            output_elements.append(output_slice)
            start += self.n_atoms_elements[i]
        
        output_all = torch.cat(output_elements, dim=-2)
        output_all = output_all.view(B, -1)   # Flatten all channels

        # Get the label embedding
        label_embed = self.label_proj(labels)  # shape: (B, label_dim) 

        # Concatenate the features and the label embedding
        combined_features = torch.cat([output_all, label_embed], dim=1)

        feature = self.feature_layer(combined_features)  # torch.Size is (current_batch_size, 200)
        return feature, self.output(feature)   # output(feature) has size (current_batch_size, 10)


class DistanceDiscriminator(nn.Module):
    def __init__(self, args, n_atoms_elements, n_label_features, label_dim):
        super(DistanceDiscriminator, self).__init__()
        self.n_elements = len(n_atoms_elements)
        self.n_atoms_elements = n_atoms_elements
        self.n_neighbors = args.n_neighbors
        self.label_dim = label_dim
        self.n_label_features = n_label_features
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,self.n_neighbors), stride = 1, padding = 0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (1,1), stride = 1, padding = 0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
            nn.LeakyReLU(0.2,inplace=True),
            )
        
        self.avgpool_elements = []
        for i in range(self.n_elements):
            self.avgpool_elements.append(nn.AvgPool2d(kernel_size = (self.n_atoms_elements[i],1)))

        # Use a linear layer to project the label to the same dimension as the output
        self.label_proj = nn.Sequential(
            nn.Linear(self.n_label_features, self.label_dim),
            nn.LeakyReLU(0.2,inplace=True),
        )

        self.feature_layer = nn.Sequential(
            nn.Linear(256*self.n_elements + self.label_dim, 1000),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1000,200),
            nn.LeakyReLU(0.2,inplace=True),
        )
        self.output = nn.Sequential(
            nn.Linear(200,10)
            )

    def forward(self, x, labels):
        B = x.shape[0]
        output = self.model(x)

        output_elements = []   # Stores the output that has been sliced based on the element type
        start = 0
        for i in range(self.n_elements):
            stop = start + self.n_atoms_elements[i]
            output_slice = output[:,:,start:stop,:]
            output_slice = self.avgpool_elements[i](output_slice)
            output_elements.append(output_slice)
            start += self.n_atoms_elements[i]
        
        output_all = torch.cat(output_elements, dim=-2)
        output_all = output_all.view(B, -1)   # Flatten all channels

        # Get the label embedding
        label_embed = self.label_proj(labels)  # shape: (B, label_dim) 

        # Concatenate the features and the label embedding
        combined_features = torch.cat([output_all, label_embed], dim=1)

        feature = self.feature_layer(combined_features)  # torch.Size is (current_batch_size, 200)
        return feature, self.output(feature)   # output(feature) has size (current_batch_size, 10)
        




