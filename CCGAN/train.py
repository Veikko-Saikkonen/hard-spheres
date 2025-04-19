import argparse
import os
import numpy as np
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.init as init
from models import Generator, CoordinateDiscriminator, DistanceDiscriminator
from tools import BatchDistance
import csv
import sys
import shutil
import time
from ase.io import read

# In case a remote cluster does not print output in real-time, this flushes the output
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
print('Python output is being flushed...')


def weights_init(m):
    "Initializes the weights of a model"
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d : 
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0.0)

def calc_gradient_penalty(netD, real_data, real_labels, fake_data, cuda, mps):
    "Calculates the WGAN gradient penalty"
    batch_size = real_data.size(0)
    # Uniform random number from the interval [0,1). This is used to give the interpolation point.
    alpha = torch.rand(batch_size, 1)   
    # Duplicates and expands alpha to the size of the real data 
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous().view(
            batch_size, 1, real_data.size(-2), real_data.size(-1))
    
    # Move alpha to the GPU if available
    if cuda:
        alpha = alpha.cuda()
    elif mps:
        # Move alpha to the MPS device
        alpha = alpha.to(torch.device("mps"))
    # Interpolates between real and fake data
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if cuda:
        interpolates = interpolates.cuda()
    elif mps:
        # Move interpolates to the MPS device
        interpolates = interpolates.to(torch.device("mps"))
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    feature, disc_interpolates = netD(interpolates, real_labels)

    grad_outputs = torch.ones(disc_interpolates.size())
    if cuda:
        grad_outputs = grad_outputs.cuda()
    elif mps:
        # Move grad_outputs to the MPS device
        grad_outputs = grad_outputs.to(torch.device("mps"))

    # Calculate the sum of gradients of outputs with respect to the inputs
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=grad_outputs,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10   # Lambda set to 10 here (same as WGAN paper)
    return gradient_penalty

class AverageMeter(object):
    "Computes and stores the average and current value"

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', type=str, default='', help='path to training data (.extxyz file)')
    parser.add_argument('--n_epochs', type=int, default=10000001, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--g_lr', type=float, default=0.00005, help='adam: generator learning rate')
    parser.add_argument('--coord_lr', type=float, default=0.00005, help='adam: coordinate discriminator learning rate')
    parser.add_argument('--dist_lr', type=float, default=0.00005, help='adam: distance discriminator learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: beta_1')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: beta_2')
    parser.add_argument('--step_size', type=int, default=100000, help='step size of learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.95, help='multiplicative factor of learning rate decay')
    parser.add_argument('--weight_dist', type=float, help='weight of distance discriminator loss')
    parser.add_argument('--latent_dim', type=int, default=64, help='number of random numbers used for generator input')
    parser.add_argument('--gen_channels_1', type=int, default=128, help='number of channels after the first layer of the generator')
    parser.add_argument('--n_neighbors', type=int, default=6, help='number of nearest neighbors when calculating bond distances')
    parser.add_argument('--gen_int', type=int, default=5, help='interval between generator trainings, the generator is trained once every "gen_int" batches')
    parser.add_argument('--load_generator', type=str, default='', help='path to generator model to be loaded')
    parser.add_argument('--load_coord_disc', type=str, default='', help='path to coordinate discriminator model to be loaded')
    parser.add_argument('--load_dist_disc', type=str, default='', help='path to distance discriminator model to be loaded')
    parser.add_argument('--load_checkpoint', type=str, default='', help='path to checkpoint to be loaded')
    parser.add_argument('--msave_dir', type = str, default = './model_saves/', help='directory to save model files')
    parser.add_argument('--msave_freq', type=int, default=1000, help='frequency of saving models, saved every "msave_freq" epochs')
    parser.add_argument('--gsave_dir', type=str, default='./gen_data/', help='directory to save generated data')
    parser.add_argument('--gsave_freq', type=int, default=1000, help='frequency of saving generated data, saved every "gsave_freq" epochs')
    parser.add_argument('--n_save', type=int, default=100, help='maximum number of structures to save in each saved file')
    parser.add_argument('--print_freq', type=int, default=20, help='print frequency of output, printed every "print_freq" batches')
    parser.add_argument('--gen_label_dim', type=int, default=64, help='conditioning label laten dimension for generator')
    parser.add_argument('--disc_label_dim', type=int, default=64, help='conditioning label laten dimension for discriminator')
    parser.add_argument('--disable_cuda', action='store_true', help='disables CUDA when called')
    
    args = parser.parse_args()
    print(args)
    
    ## Determine whether to use GPU
    cuda = not args.disable_cuda and torch.cuda.is_available()
    mps = not args.disable_cuda and torch.backends.mps.is_available() and torch.backends.mps.is_built()

    print('cuda is', cuda)
    print('mps is', mps)
    
    ## Initialize best distance and starting epoch
    best_distance = 1e10
    start_epoch = 0

    ## Create directories for saving generated data and trained models
    if not os.path.isdir(args.gsave_dir):
        os.makedirs(args.gsave_dir)
    if not os.path.isdir(args.msave_dir):
        os.makedirs(args.msave_dir)

    ## Read and prepare training data
    print("Reading training data...")
    ase_atoms = []
    if os.path.isdir(args.training_data):
        for root, _, files in os.walk(args.training_data):
            print("Reading folder: ", root)
            for file in files:
                if file.endswith('.extxyz'):
                    file_path = os.path.join(root, file)
                    ase_atoms.extend(read(file_path, index=':', format='extxyz'))
    else:
        ase_atoms = read(args.training_data, index=':', format='extxyz')
    lattice = ase_atoms[0].get_cell()[:]   # Lattice vectors, array of shape (3,3)
    n_atoms_total = len(ase_atoms[0])   # Total number of atoms in each structure
    _, idx, n_atoms_elements = np.unique(ase_atoms[0].numbers, return_index=True, return_counts=True)
    n_atoms_elements = n_atoms_elements[np.argsort(idx)]   # Array of number of atoms per element in each structure
    train_coords_all = []   # Stores the fractional coordinates of all structures in ase_atoms
    for i in range(len(ase_atoms)):
        train_coords_all.append(ase_atoms[i].get_scaled_positions())
    train_coords_all = torch.FloatTensor(np.array(train_coords_all))
    # Append bond distances to the coordinates
    print("Appending bond distances...")
    prep_dataloader = torch.utils.data.DataLoader(train_coords_all, batch_size = 256, shuffle = False)
    train_data = []   # Stores the fractional coordinates and bond distances of all structures
    for i, batch_coords in enumerate(prep_dataloader):
        batch_coords = batch_coords.view(batch_coords.shape[0], 1, n_atoms_total, 3).float()
        if cuda:
            batch_coords = batch_coords.cuda()
        elif mps:
            batch_coords = batch_coords.to(device='mps')
            
        batch_dataset = BatchDistance(batch_coords, n_neighbors=args.n_neighbors, lat_matrix=lattice)
        batch_coords_with_dist = batch_dataset.append_dist()
        train_data.append(batch_coords_with_dist.cpu())
    train_data = torch.cat(train_data)
    torch.save(train_data, 'train_data.pt')
    print("Training data shape is ", train_data.shape)
    # Remove unneeded objects to free up memory
    
    # Get the labels (only phi for now)
    train_labels = np.array([ase_atoms[i].info["phi"] for i in range(len(ase_atoms))])
    train_labels = torch.FloatTensor(train_labels).unsqueeze(1)

    # Create custom dataset that includes the labels
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    dataset = CustomDataset(train_data, train_labels)
    
    
    del ase_atoms, train_coords_all, prep_dataloader, batch_coords, batch_dataset, batch_coords_with_dist
    print("=> Training data prepared.")

	## Configure data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

	## Initialize generator and discriminators
    # generator = Generator(args, n_atoms_total)
    generator = Generator(args, n_atoms_total, n_label_features=train_labels.shape[1], label_dim=args.gen_label_dim)
    # coord_disc = CoordinateDiscriminator(args, n_atoms_elements)
    coord_disc = CoordinateDiscriminator(args, n_atoms_elements, n_label_features=train_labels.shape[1], label_dim=args.disc_label_dim)
    dist_disc = DistanceDiscriminator(args, n_atoms_elements, n_label_features=train_labels.shape[1], label_dim=args.disc_label_dim)
    if cuda:
        generator.cuda()
        coord_disc.cuda()
        dist_disc.cuda()
    elif mps:
        generator.to(device='mps')
        coord_disc.to(device='mps')
        dist_disc.to(device='mps')

    print("=> Models initialized.")

	## Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.g_lr, betas=(args.b1, args.b2))
    optimizer_CD = torch.optim.Adam(coord_disc.parameters(), lr=args.coord_lr, betas=(args.b1, args.b2))
    optimizer_DD = torch.optim.Adam(dist_disc.parameters(), lr=args.dist_lr, betas=(args.b1, args.b2))

    ## Schedulers
    ## Learning rate is multiplied by a factor of 'gamma' every 'step_size' epochs
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=args.step_size, gamma=args.gamma)
    scheduler_CD = torch.optim.lr_scheduler.StepLR(optimizer_CD, step_size=args.step_size, gamma=args.gamma)
    scheduler_DD = torch.optim.lr_scheduler.StepLR(optimizer_DD, step_size=args.step_size, gamma=args.gamma)
    
    
	## Load model or initialize
    if args.load_generator:
        print("Loading generator...")
        assert os.path.exists(args.load_generator), "Cannot find generator model to load!"
        generator.load_state_dict(torch.load(args.load_generator))
        print("=> Loaded '{}'.".format(args.load_generator))
    else:
        generator.apply(weights_init)
        print("Generator weights are initialized.")
    
    if args.load_coord_disc:
        print("Loading coordinate discriminator...")
        assert os.path.exists(args.load_coord_disc), "Cannot find coordinate discriminator model to load!"
        coord_disc.load_state_dict(torch.load(args.load_coord_disc))
        print("=> Loaded '{}'.".format(args.load_coord_disc))
    else:
        coord_disc.apply(weights_init)
        print("Coordinate discriminator weights are initialized.")
        
    if args.load_dist_disc:
        print("Loading distance discriminator...")
        assert os.path.exists(args.load_dist_disc), "Cannot find distance discriminator model to load!"
        dist_disc.load_state_dict(torch.load(args.load_dist_disc))
        print("=> Loaded '{}'.".format(args.load_dist_disc))
    else:
        dist_disc.apply(weights_init)
        print("Distance discriminator weights are initialized.")
    
    ## Load checkpoint to restart training
    if args.load_checkpoint:
        print("Loading checkpoint...")
        assert os.path.exists(args.load_checkpoint), "Cannot find checkpoint to load!"
        checkpoint = torch.load(args.load_checkpoint)
        start_epoch = checkpoint['epoch']
        best_distance = checkpoint['best_distance']
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_CD.load_state_dict(checkpoint['optimizer_CD'])
        optimizer_DD.load_state_dict(checkpoint['optimizer_DD'])
        scheduler_G.load_state_dict(checkpoint['scheduler_G'])
        scheduler_CD.load_state_dict(checkpoint['scheduler_CD'])
        scheduler_DD.load_state_dict(checkpoint['scheduler_DD'])
        print("=> Loaded checkpoint. Checkpoint epoch is {}.".format(checkpoint['epoch']))


    for epoch in range(start_epoch, args.n_epochs):
        w_dis = AverageMeter()      # Stores the Wasserstein distance of the discriminators
        cost_dis = AverageMeter()   # Stores the loss of the discriminators
        cost_gen = AverageMeter()   # Stores the loss of the generator
        meter_D_real = AverageMeter()   # Stores D_real
        meter_D_fake = AverageMeter()   # Stores D_fake
        meter_D_dist_real = AverageMeter()   # Stores D_dist_real
        meter_D_dist_fake = AverageMeter()   # Stores D_dist_fake
        data_time_fake = AverageMeter()   # Stores the time for data of fake structures to complete prep
        batch_time = AverageMeter()  # Stores the time for batch to complete
        end = time.time()   # time stamp
        
        for i, (real_coords_with_dis, real_labels) in enumerate(dataloader):
            for p in coord_disc.parameters():
                p.requires_grad = True
            for p in dist_disc.parameters():
                p.requires_grad = True
            
            ## Prepare tensor of real coordinates
            current_batch_size = real_labels.shape[0]
            if cuda:
                real_coords_with_dis = real_coords_with_dis.cuda()
                real_labels = real_labels.cuda()
            elif mps:
                real_coords_with_dis = real_coords_with_dis.to(device='mps')
                real_labels = real_labels.to(device='mps')

            ## Prepare tensor of real distances
            real_coords = real_coords_with_dis[:,:,:,:3]
            real_distances = real_coords_with_dis[:,:,:,3:]
            
            ## Feed real coordinates into Coordinate Discriminator
            real_feature, D_real = coord_disc(real_coords, real_labels) # real_feature is tensor of (current_batch_size, 200). D_real is the real_feature fed into linear layer to reduce from 200 to 10 values.
            D_real = D_real.mean()
            
            ## Feed real distances into Distance Discriminator
            real_dist_feature, D_dist_real = dist_disc(real_distances, real_labels) # real_dist_feature is tensor of (current_batch_size, 200). D_dist_real is the real_dist_feature fed into linear layer to reduce from 200 to 10 values.
            D_dist_real = D_dist_real.mean()
            
            
            ## Generate fake coordinates
            z = torch.FloatTensor(np.random.normal(0,1,(current_batch_size, args.latent_dim)))   # torch.Size([current_batch_size, args.latent_dim])
            if cuda:
                z = z.cuda()  
            elif mps:
                z = z.to(device='mps')
            ## Feed fake coordinates into Coordinate Discriminator
            fake_coords = generator(z, real_labels)   # size is (current_batch_size, 1, n_atoms_total, 3)
            fake_feature, D_fake = coord_disc(fake_coords.detach(), real_labels.detach())  # fake feature has size (current_batch_size, 200), D_fake has size (current_batch_size, 10)
            D_fake = D_fake.mean()
            ## Feed fake distances into Distance Discriminator
            end_fake = time.time()   # time stamp for fake structures
            fake_dataset = BatchDistance(fake_coords, n_neighbors=args.n_neighbors, lat_matrix=lattice)
            fake_distances = fake_dataset.append_dist()[:,:,:,3:]
            data_time_fake.update(time.time() - end_fake)   # measure data prep time
            fake_dist_feature, D_dist_fake = dist_disc(fake_distances.detach(), real_labels.detach())
            D_dist_fake = D_dist_fake.mean()
            

            ## Compute gradient and do optimizer step. Save losses. 
            optimizer_CD.zero_grad()
            optimizer_DD.zero_grad()

            gradient_penalty_D = calc_gradient_penalty(coord_disc, real_coords, real_labels, fake_coords, cuda, mps)
            gradient_penalty_dist = calc_gradient_penalty(dist_disc, real_distances, real_labels, fake_distances, cuda, mps)
            
            D_cost = D_fake - D_real + gradient_penalty_D + args.weight_dist*(D_dist_fake - D_dist_real + 
                                                                              gradient_penalty_dist)
            D_cost.backward()
            cost_dis.update(D_cost.detach().clone().item(), n=current_batch_size)
            Wasserstein_D = D_real - D_fake + args.weight_dist*(D_dist_real - D_dist_fake)
            w_dis.update(Wasserstein_D.detach().clone().item(), n=current_batch_size)
            
            meter_D_real.update(D_real.detach().clone().item(), n=current_batch_size)
            meter_D_fake.update(D_fake.detach().clone().item(), n=current_batch_size)
            meter_D_dist_real.update(D_dist_real.detach().clone().item(), n=current_batch_size)
            meter_D_dist_fake.update(D_dist_fake.detach().clone().item(), n=current_batch_size)
            
            optimizer_CD.step()
            optimizer_DD.step()
            
            
            ## Train Generator every "gen_int" batches with new noise z
            if i % args.gen_int == 0 :		
                for p in coord_disc.parameters():
                    p.requires_grad = False
                for p in dist_disc.parameters():
                    p.requires_grad = False
                optimizer_G.zero_grad()
                
                ## Generate fake coordinates
                z = torch.FloatTensor(np.random.normal(0,1,(current_batch_size, args.latent_dim)))
                if cuda :
                    z = z.cuda()
                elif mps:
                    z = z.to(device='mps')
                fake_coords = generator(z, real_labels)   # size is (current_batch_size, 1, n_atoms_total, 3)
                ## Feed fake coordinates into Coordinate Discriminator
                fake_feature_G, D_fake_G = coord_disc(fake_coords, real_labels)
                D_fake_G = D_fake_G.mean()
                ## Feed fake distances into Distance Discriminator
                fake_dataset = BatchDistance(fake_coords, n_neighbors=args.n_neighbors, lat_matrix=lattice)
                fake_distances = fake_dataset.append_dist()[:,:,:,3:]
                fake_dist_feature_G, D_dist_fake_G = dist_disc(fake_distances, real_labels)
                D_dist_fake_G = D_dist_fake_G.mean()       
 
                
                ## Compute gradient and do optimizer step. Save loss.
                G_cost = -D_fake_G - args.weight_dist*D_dist_fake_G
                G_cost.backward()
                cost_gen.update(G_cost.detach().clone().item(), n=current_batch_size)
                optimizer_G.step()
            
            
            ## Measure batch completion time and print output
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Fake data time {data_time_fake.val:.3f} ({data_time_fake.avg:.3f})\t'
                      'Distance {w_dis.val:.6f} ({w_dis.avg:.6f})\t'
                      'Memory used {mem_used:.3f}\t'
                      'Memory reserved {mem_res:.3f}'.format(
                    epoch, i, len(dataloader)-1, 
                    batch_time=batch_time, data_time_fake=data_time_fake,
                    w_dis=w_dis, mem_used=torch.cuda.max_memory_allocated()/2**30, 
                    mem_res=torch.cuda.max_memory_reserved()/2**30)
                    )

            ## Store the fake coordinates that were generated. Only saves up to "n_save" fake structures. 
            n_save = args.n_save   # Maximum number of fake structures to save
            if i == 0:
                gen_coords = fake_coords.detach().clone()[:n_save]
            else:
                n_current = len(gen_coords)   # Current number of fake structures saved
                if n_current < n_save:
                    n_need = n_save - n_current   # Number of fake structures needed
                    gen_coords = torch.cat((gen_coords, fake_coords.detach().clone()[:n_need]), dim = 0)
            
            ## ---------------------- END OF BATCHES -----------------------------
        
        ## Check if current Wasserstein distance is less than best distance
        is_best = w_dis.avg < best_distance
        best_distance = min(w_dis.avg, best_distance)

        ## Update learning rates
        scheduler_G.step()
        scheduler_CD.step()
        scheduler_DD.step()
        
        ## Save models and checkpoint file
        if epoch % args.msave_freq == 0:
            if os.path.exists('checkpoint.pth.tar'):
                shutil.copyfile('checkpoint.pth.tar', 'checkpoint_previous.pth.tar')
            torch.save(generator.state_dict(), args.msave_dir+'generator_'+str(epoch))
            torch.save(coord_disc.state_dict(), args.msave_dir+'coord_disc_'+str(epoch))
            torch.save(dist_disc.state_dict(), args.msave_dir+'dist_disc_'+str(epoch))
            torch.save({
                'epoch': epoch + 1,
                'best_distance': best_distance,
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_CD': optimizer_CD.state_dict(),
                'optimizer_DD': optimizer_DD.state_dict(),
                'scheduler_G': scheduler_G.state_dict(),
                'scheduler_CD': scheduler_CD.state_dict(),
                'scheduler_DD': scheduler_DD.state_dict(),
                'args': vars(args)
                }, 'checkpoint.pth.tar') 
        if is_best:
            torch.save(generator.state_dict(), args.msave_dir+'best_generator')
            torch.save(coord_disc.state_dict(), args.msave_dir+'best_coord_disc')
            torch.save(dist_disc.state_dict(), args.msave_dir+'best_dist_disc')
            torch.save({
                'epoch': epoch,
                'best_distance': best_distance,
                'args': vars(args)
                }, args.msave_dir+'best_epoch.pth.tar') 
        
        ## Save fake coordinates every (args.gsave_freq) epochs.
        if epoch % args.gsave_freq == 0:		
            gen_name = args.gsave_dir+'gen_coords_'+str(epoch)
            gen_file = gen_coords.cpu().numpy()
            np.save(gen_name, gen_file)
        
        ## Write losses and learning rate files
        with open(args.msave_dir+"losses.csv", "a", newline='') as csvfile: 
            csvwriter = csv.writer(csvfile)
            if epoch == 0:
                csvwriter.writerow(['epoch', 'distance_dis', 'cost_dis', 'cost_gen',
                                    'D_real', 'D_fake', 'D_dist_real', 'D_dist_fake'])
            csvwriter.writerow([epoch, w_dis.avg, cost_dis.avg, cost_gen.avg,
                                meter_D_real.avg, meter_D_fake.avg,
                                meter_D_dist_real.avg, meter_D_dist_fake.avg])
        with open(args.msave_dir+"learning_rate.csv", "a", newline='') as csvfile: 
            csvwriter = csv.writer(csvfile)
            if epoch == 0:
                csvwriter.writerow(['epoch', 'generator_lr', 'coord_disc_lr', 'dist_disc_lr'])
            csvwriter.writerow([epoch, optimizer_G.param_groups[0]['lr'],
                                optimizer_CD.param_groups[0]['lr'], optimizer_DD.param_groups[0]['lr']])
        
        # CSV files to be used when restarting training from last saved model
        if epoch % args.msave_freq == 0:
            shutil.copyfile(args.msave_dir+'losses.csv', args.msave_dir+'losses_cp.csv')
            shutil.copyfile(args.msave_dir+'learning_rate.csv', args.msave_dir+'learning_rate_cp.csv')
 
 

if  __name__ == '__main__':
    main()

