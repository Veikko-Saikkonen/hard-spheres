import argparse
import os
import numpy as np
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.init as init
from models import Generator, CoordinateDiscriminator
import csv
import sys
from torchinfo import summary
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
    parser.add_argument('--b1', type=float, default=0.5, help='adam: beta_1')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: beta_2')
    parser.add_argument('--step_size', type=int, default=100000, help='step size of learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.95, help='multiplicative factor of learning rate decay')
    parser.add_argument('--latent_dim', type=int, default=64, help='number of random numbers used for generator input')
    parser.add_argument('--gen_channels_1', type=int, default=128, help='number of channels after the first layer of the generator')
    parser.add_argument('--gen_int', type=int, default=5, help='interval between generator trainings, the generator is trained once every "gen_int" batches')
    parser.add_argument('--load_generator', type=str, default='', help='path to generator model to be loaded')
    parser.add_argument('--load_coord_disc', type=str, default='', help='path to coordinate discriminator model to be loaded')
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
    #lattice = ase_atoms[0].get_cell()[:]   # Lattice vectors, array of shape (3,3)
    n_atoms_total = len(ase_atoms[0])   # Total number of atoms in each structure
    _, idx, n_atoms_elements = np.unique(ase_atoms[0].numbers, return_index=True, return_counts=True)
    n_atoms_elements = n_atoms_elements[np.argsort(idx)]   # Array of number of atoms per element in each structure        
    train_coords_all = []   # Stores the fractional coordinates of all structures in ase_atoms
    for i in range(len(ase_atoms)):
        train_coords_all.append(ase_atoms[i].get_scaled_positions())
    train_coords_all = torch.FloatTensor(np.array(train_coords_all))
    train_coords_all = train_coords_all.unsqueeze(1)

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
    dataset = CustomDataset(train_coords_all, train_labels)

    # Remove unneeded object to free up memory
    del ase_atoms
    print("=> Training data prepared.")

	## Configure data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = True)
    print("=> Data loader configured.")
	## Initialize generator and discriminators
    generator = Generator(args, n_atoms_total, n_label_features=train_labels.shape[1], label_dim=args.gen_label_dim)
    coord_disc = CoordinateDiscriminator(args, n_atoms_elements, n_label_features=train_labels.shape[1], label_dim=args.disc_label_dim)
    if cuda:
        generator.cuda()
        coord_disc.cuda()
    
    if mps:
        generator.to(device='mps')
        coord_disc.to(device='mps')

    print("=> Models initialized.")

    ## Print model summary
    sample_batch_size = 32
    z = torch.FloatTensor(np.random.normal(0,1,(sample_batch_size, args.latent_dim)))

    summary(generator, input_size=(z[0:sample_batch_size].size(), train_labels[0:sample_batch_size].size()), col_names=["input_size", "output_size", "num_params"])
    summary(coord_disc, input_size=(train_coords_all[0:sample_batch_size].size(), train_labels[0:sample_batch_size].size()), col_names=["input_size", "output_size", "num_params"])
    print("=> Model summary printed.")

    exit()
	## Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.g_lr, betas=(args.b1, args.b2))
    optimizer_CD = torch.optim.Adam(coord_disc.parameters(), lr=args.coord_lr, betas=(args.b1, args.b2))

    ## Schedulers
    ## Learning rate is multiplied by a factor of 'gamma' every 'step_size' epochs
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=args.step_size, gamma=args.gamma)
    scheduler_CD = torch.optim.lr_scheduler.StepLR(optimizer_CD, step_size=args.step_size, gamma=args.gamma)
    
	## Load model or initialize
    if args.load_generator:
        print("Loading generator...")
        assert os.path.exists(args.load_generator), "Cannot find generator model to load!"
        generator.load_state_dict(torch.load(args.load_generator, weights_only=False,))
        print("=> Loaded '{}'.".format(args.load_generator))
    else:
        generator.apply(weights_init)
        print("Generator weights are initialized.")
    
    if args.load_coord_disc:
        print("Loading coordinate discriminator...")
        assert os.path.exists(args.load_coord_disc), "Cannot find coordinate discriminator model to load!"
        coord_disc.load_state_dict(torch.load(args.load_coord_disc, weights_only=False))
        print("=> Loaded '{}'.".format(args.load_coord_disc))
    else:
        coord_disc.apply(weights_init)
        print("Coordinate discriminator weights are initialized.")
    
    ## Load checkpoint to restart training
    if args.load_checkpoint:
        print("Loading checkpoint...")
        assert os.path.exists(args.load_checkpoint), "Cannot find checkpoint to load!"
        checkpoint = torch.load(args.load_checkpoint, weights_only=False)
        start_epoch = checkpoint['epoch']
        best_distance = checkpoint['best_distance']
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_CD.load_state_dict(checkpoint['optimizer_CD'])
        scheduler_G.load_state_dict(checkpoint['scheduler_G'])
        scheduler_CD.load_state_dict(checkpoint['scheduler_CD'])
        print("=> Loaded checkpoint. Checkpoint epoch is {}.".format(checkpoint['epoch']))
    

    for epoch in range(start_epoch, args.n_epochs):
        w_dis = AverageMeter()      # Stores the Wasserstein distance of the discriminator
        cost_dis = AverageMeter()   # Stores the loss of the discriminator
        cost_gen = AverageMeter()   # Stores the loss of the generator
        meter_D_real = AverageMeter()   # Stores D_real
        meter_D_fake = AverageMeter()   # Stores D_fake
        batch_time = AverageMeter()  # Stores the time for batch to complete
        end = time.time()   # time stamp
        
        for i, (real_coords, real_labels) in enumerate(dataloader):
            for p in coord_disc.parameters():
                p.requires_grad = True
            
            ## Prepare tensor of real coordinates
            current_batch_size = real_coords.shape[0] 
            if cuda:
                real_coords = real_coords.cuda()
                real_labels = real_labels.cuda()
            if mps:
                real_coords = real_coords.to(device='mps')
                real_labels = real_labels.to(device='mps')
            
            ## Feed real coordinates into Coordinate Discriminator
            real_feature, D_real = coord_disc(real_coords, real_labels) # real_feature is tensor of (current_batch_size, 200). D_real is the real_feature fed into linear layer to reduce from 200 to 10 values.
            D_real = D_real.mean()
            
            
            ## Generate fake coordinates
            z = torch.FloatTensor(np.random.normal(0,1,(current_batch_size, args.latent_dim)))   # torch.Size([current_batch_size, args.latent_dim])
            if cuda :
                z = z.cuda()  
            if mps:
                z = z.to(device='mps')
            ## Feed fake coordinates into Coordinate Discriminator
            fake_coords = generator(z, real_labels)   # size is (current_batch_size, 1, n_atoms_total, 3)
            fake_feature, D_fake = coord_disc(fake_coords.detach(), real_labels)  # fake feature has size (current_batch_size, 200), D_fake has size (current_batch_size, 10)
            D_fake = D_fake.mean()
            

            ## Compute gradient and do optimizer step. Save losses. 
            optimizer_CD.zero_grad()

            # Print devices

            gradient_penalty_D = calc_gradient_penalty(coord_disc, real_coords, real_labels, fake_coords, cuda)
            
            D_cost = D_fake - D_real + gradient_penalty_D
            D_cost.backward()
            cost_dis.update(D_cost.detach().clone().item(), n=current_batch_size)
            Wasserstein_D = D_real - D_fake
            w_dis.update(Wasserstein_D.detach().clone().item(), n=current_batch_size)
            
            meter_D_real.update(D_real.detach().clone().item(), n=current_batch_size)
            meter_D_fake.update(D_fake.detach().clone().item(), n=current_batch_size)
            
            optimizer_CD.step()
            
            ## Train Generator every "gen_int" batches with new noise z
            if i % args.gen_int == 0 :		
                for p in coord_disc.parameters():
                    p.requires_grad = False
                optimizer_G.zero_grad()
                
                ## Generate fake coordinates
                z = torch.FloatTensor(np.random.normal(0,1,(current_batch_size, args.latent_dim)))
                if cuda :
                    z = z.cuda()
                if mps:
                    z = z.to(device='mps')
                fake_coords = generator(z, real_labels)   # size is (current_batch_size, 1, n_atoms_total, 3)
                ## Feed fake coordinates into Coordinate Discriminator
                fake_feature_G, D_fake_G = coord_disc(fake_coords, real_labels)
                D_fake_G = D_fake_G.mean()
 
                
                ## Compute gradient and do optimizer step. Save loss.
                G_cost = -D_fake_G
                G_cost.backward()
                cost_gen.update(G_cost.detach().clone().item(), n=current_batch_size)
                optimizer_G.step()
            
            
            ## Measure batch completion time and print output
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Distance {w_dis.val:.6f} ({w_dis.avg:.6f})\t'
                      'Memory used {mem_used:.3f}\t'
                      'Memory reserved {mem_res:.3f}'.format(
                    epoch, i, len(dataloader)-1, batch_time=batch_time,
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
        
        ## Save models and checkpoint file
        if epoch % args.msave_freq == 0:
            if os.path.exists('checkpoint.pth.tar'):
                shutil.copyfile('checkpoint.pth.tar', 'checkpoint_previous.pth.tar')
            torch.save(generator.state_dict(), args.msave_dir+'generator_'+str(epoch))
            torch.save(coord_disc.state_dict(), args.msave_dir+'coord_disc_'+str(epoch))
            torch.save({
                'epoch': epoch + 1,
                'best_distance': best_distance,
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_CD': optimizer_CD.state_dict(),
                'scheduler_G': scheduler_G.state_dict(),
                'scheduler_CD': scheduler_CD.state_dict(),
                'args': vars(args)
                }, 'checkpoint.pth.tar') 
        if is_best:
            torch.save(generator.state_dict(), args.msave_dir+'best_generator')
            torch.save(coord_disc.state_dict(), args.msave_dir+'best_coord_disc')
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
        with open("losses.csv", "a", newline='') as csvfile: 
            csvwriter = csv.writer(csvfile)
            if epoch == 0:
                csvwriter.writerow(['epoch', 'distance_dis', 'cost_dis', 'cost_gen',
                                    'D_real', 'D_fake'])
            csvwriter.writerow([epoch, w_dis.avg, cost_dis.avg, cost_gen.avg,
                                meter_D_real.avg, meter_D_fake.avg])
        with open("learning_rate.csv", "a", newline='') as csvfile: 
            csvwriter = csv.writer(csvfile)
            if epoch == 0:
                csvwriter.writerow(['epoch', 'generator_lr', 'coord_disc_lr'])
            csvwriter.writerow([epoch, optimizer_G.param_groups[0]['lr'],
                                optimizer_CD.param_groups[0]['lr']])
        
        # CSV files to be used when restarting training from last saved model
        if epoch % args.msave_freq == 0:
            shutil.copyfile('losses.csv', 'losses_cp.csv')
            shutil.copyfile('learning_rate.csv', 'learning_rate_cp.csv')
 

if  __name__ == '__main__':
    main()

