import torch
from torch.utils.data import Dataset
from pytorch3d.ops import knn_points

def apply_pbc_3x3x3(frac_coords, device):
    """
    Adds atom images based on periodic boundary condition to a batch of crystals,  
    by forming a 3x3x3 supercell. 
    This is to allow for finding neighbors that includes the images.
    
    Parameters
    ----------
    frac_coords: torch.FloatTensor shape (n_structures, 1, n_atoms, 3)
        Fractional coordinates of a batch of crystals
    device: 'cpu' or 'cuda'
        Indicates to run the function on cpu or gpu
    
    Returns
    ----------
    pbc_frac_coords: torch.FloatTensor shape (n_structures, 1, n_atoms*27, 3)
        Fractional coordinates of all crystal atoms and images 
    
    """
    n_structures = frac_coords.shape[0]
    n_atoms = frac_coords.shape[2]
    # Flatten coordinates of each structure
    frac_coords_flat = torch.clone(frac_coords).to(device)   
    frac_coords_flat = frac_coords_flat.view(n_structures, 1, n_atoms*3)
    # Get pbc images 
    pbc_list = []   # For storing fractional coordinates of all atoms and pbc images
    pbc_list.append(frac_coords_flat)   # First set of coordinates are the original coordinates
    for x in [-1.0, 0.0, 1.0]:
        for y in [-1.0, 0.0, 1.0]:
            for z in [-1.0, 0.0, 1.0]:
                if x == 0.0 and y == 0.0 and z == 0.0:
                    pass
                else:
                    pbc_list.append(frac_coords_flat + torch.tensor([x, y, z], device=device).repeat(n_atoms))
    pbc_frac_coords = torch.cat(pbc_list, dim=2)
    pbc_frac_coords = pbc_frac_coords.view(n_structures, 1, n_atoms*27, 3)      

    return pbc_frac_coords


class BatchDistance(Dataset):
    """
    The BatchDistance dataset is a wrapper for a dataset of fractional coordinates.
    The append_dist function appends the bond distances of the n nearest neighbors 
    to the fractional coordinates of each atom.

    Parameters
    ----------

    coords: Tensor shape (current_batch_size, 1, n_atoms, 3)
        A set of fractional coordinates 
    n_neighbors: int
        The number of nearest neighbors when determining bond distances.
    lat_matrix: array shape (3, 3)
        The lattice vectors of the structures

    Returns
    -------

    Using the append_dist function,
    coords_with_dis: Tensor shape (current_batch_size, 1, n_atoms, 3+n_neighbors)
        Fractional coordinates with bond distances of n nearest neighbors appended
    
    """
    def __init__(self, coords, n_neighbors, lat_matrix):
        self.coords = coords
        self.device = coords.device
        self.n_neighbors = n_neighbors
        self.lat_matrix = torch.FloatTensor(lat_matrix).to(self.device)

    def __len__(self):
        return self.coords.size()[0]

    def __getitem__(self, idx):
        return self.coords[idx]
    
    def append_dist(self):
        frac_coords = self.coords   # size is (current_batch_size, 1, n_atoms, 3)
        cart_coords = torch.matmul(frac_coords, self.lat_matrix)
        cart_coords = cart_coords.view(cart_coords.shape[0],cart_coords.shape[2],3)   # size is (current_batch_size, n_atoms, 3)
        # Add periodic boundary condition images for finding neighbors
        pbc_frac_coords = apply_pbc_3x3x3(frac_coords, device=self.device)
        pbc_cart_coords = torch.matmul(pbc_frac_coords, self.lat_matrix)
        pbc_cart_coords = pbc_cart_coords.view(pbc_cart_coords.shape[0],pbc_cart_coords.shape[2],3)   # size is (current_batch_size, n_atoms*27, 3)
        # Finds the (n_neighbors+1) nearest neighbors, first neighbor will be itself
        lengths1 = torch.tensor(cart_coords.shape[1], device=self.device).repeat(cart_coords.shape[0])   # number of atoms in each structure 
        lengths2 = torch.tensor(pbc_cart_coords.shape[1], device=self.device).repeat(pbc_cart_coords.shape[0])   # number of atoms in each pbc_structure 
        bond_dis, _, _ = knn_points(cart_coords, pbc_cart_coords, 
                                    lengths1=lengths1, lengths2=lengths2,
                                    K=self.n_neighbors+1, return_sorted=True)   # Size is (current_batch_size, n_atoms, n_neighbors+1)
        # Remove bond distance with itself
        bond_dis = bond_dis.view(bond_dis.shape[0], 1, bond_dis.shape[1], bond_dis.shape[2])[:,:,:,1:]   # Size is (current_batch_size, 1, n_atoms, n_neighbors)
        bond_dis = torch.sqrt(bond_dis)   # sqrt because knn_points gives squared distances
        # Append bond distance to fractional coordinates
        coords_with_dis = torch.cat((frac_coords, bond_dis), dim=3)
        
        return coords_with_dis

    

