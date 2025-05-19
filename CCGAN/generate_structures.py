import torch
from models import Generator
from ase.io import read, write
import argparse
from tqdm import tqdm
from pathlib import Path

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
    

    args = parser.parse_args()

    struc_template = read(args.ref_struc, index=0, format='extxyz')  
    n_atoms_total = len(struc_template)
    
    ref_phi = struc_template.info['phi']
    # ref_L = struc_template.info['L']

    # ref_label = torch.tensor([ref_phi, ref_L],dtype=torch.float32).unsqueeze(0)
    ref_label = torch.tensor([ref_phi],dtype=torch.float32).unsqueeze(0)
    ref_label = ref_label.repeat(args.n_struc, 1)  # Repeat for the number of structures to generate


    Path(args.write_fname).mkdir(parents=True, exist_ok=True)

    args.write_fname = args.write_fname + "phi-" + str(ref_phi) + ".extxyz"

    
    # Load generator
    generator = Generator(args, n_atoms_total, n_label_features=ref_label.shape[1], label_dim=args.gen_label_dim)
    print("Loading generator...")
    if torch.cuda.is_available():
        generator.load_state_dict(torch.load(args.load_generator, weights_only=False, map_location=torch.device("cuda")))
    elif torch.backends.mps.is_available():
        generator.load_state_dict(torch.load(args.load_generator, weights_only=False, map_location=torch.device("mps")))
    else:
        generator.load_state_dict(torch.load(args.load_generator, weights_only=False, map_location=torch.device("cpu")))
    print("=> Loaded '{}'.".format(args.load_generator))
    generator.eval()

    # Generate fake coordinates
    z = torch.randn(args.n_struc, args.latent_dim)
    fake_coords = generator(z, ref_label).detach()

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

