import torch

class Body:
    def __init__(self, name, init_pos, init_vel, G_mass, color, dtype=torch.float64):
        self.dtype = dtype

        if type(init_pos) == torch.Tensor:
            self.position = init_pos
        else:
            self.position = torch.tensor(init_pos, dtype=self.dtype)     # km, km, km


        if type(init_vel) == torch.Tensor:
            self.velocity = init_vel
        else:
            self.velocity = torch.tensor(init_vel, dtype=self.dtype)     # km/s, km/s, km/s

        if type(G_mass) == torch.Tensor: 
            self.G_mass = G_mass
        else:
            self.G_mass = torch.tensor(G_mass, dtype=self.dtype)      # km^3/s^2
            
        self.name = name
        self.color = color

    def calc_accel(self, other):
        displacement = other.position - self.position
        disp_mag = displacement.norm()
        disp_dir = displacement / disp_mag
        acceleration = other.G_mass / (disp_mag*disp_mag) * disp_dir

        return acceleration