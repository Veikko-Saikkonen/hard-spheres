import torch
from models import Generator
from ase.io import read, write
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_generator', type=str, default='/path/to/generator', help='path to generator model to be loaded')
    parser.add_argument('--n_struc', type=int, default=1000, help='number of structures to generate')
    parser.add_argument('--ref_struc', type=str, default='/path/to/ref_struc.extxyz', help='path to reference structure (.extxyz file)')
    parser.add_argument('--latent_dim', type=int, default=64, help='number of random numbers used for generator input')
    parser.add_argument('--gen_channels_1', type=int, default=128, help='number of channels after the first layer of the generator')
    parser.add_argument('--write_fname', type=str, default='gen', help='filename to write generated structures (.extxyz file)')
    parser.add_argument('--n_labels', type=int, help='conditioning label dimension')
    parser.add_argument('--gen_label_dim', type=int, default=64, help='conditioning label latent dimension for generator')
    parser.add_argument('--label_phis', type=float, default=0.84, help='phi value for the generator')
    

    args = parser.parse_args()

    args.write_fname = args.write_fname + "-phi-" + str(args.label_phis) + ".extxyz"

    struc_template = read(args.ref_struc, index=0, format='extxyz')  
    n_atoms_total = len(struc_template)
    
    # Load generator
    generator = Generator(args, n_atoms_total, n_label_features=1, label_dim=args.gen_label_dim)
    print("Loading generator...")
    if torch.cuda.is_available():
        generator.load_state_dict(torch.load(args.load_generator, weights_only=False, map_location=torch.device("cpu") if not torch.cuda.is_available() else None))
    elif torch.backends.mps.is_available():
        generator.load_state_dict(torch.load(args.load_generator, weights_only=False, map_location=torch.device("mps")))
    else:
        generator.load_state_dict(torch.load(args.load_generator, weights_only=False, map_location=torch.device("cpu")))
    print("=> Loaded '{}'.".format(args.load_generator))
    generator.eval()

    conditioned_labels = torch.zeros(args.n_struc, args.n_labels)
    conditioned_labels[:, 0] = args.label_phis
    
    # Generate fake coordinates
    z = torch.randn(args.n_struc, args.latent_dim)
    fake_coords = generator(z, conditioned_labels).detach()

    # Save structures by replacing the coordinates of the structure template with the generated coordinates
    fake_struc_all = []
    for i in tqdm(range(len(fake_coords))):
        coords = fake_coords[i][0]
        struc = struc_template.copy()
        struc.set_scaled_positions(coords)
        struc.wrap()
        fake_struc_all.append(struc)
    print("Writing generated structures to file...")
    print("File name: ", args.write_fname)
    write(args.write_fname, fake_struc_all, format='extxyz')


if  __name__ == '__main__':
    main()

