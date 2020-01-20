import torch

class Body:
    def __init__(self, name, init_pos, init_vel, G_mass, color, dtype=torch.float64):
        self.dtype = dtype
        self.position = torch.tensor(init_pos, dtype=self.dtype)     # km, km, km
        self.velocity = torch.tensor(init_vel, dtype=self.dtype)     # km/s, km/s, km/s
        self.G_mass = torch.tensor(G_mass, dtype=self.dtype)      # km^3/s^2
        self.name = name
        self.color = color

    def calc_accel(self, other):
        displacement = other.position - self.position
        disp_mag = displacement.norm()
        disp_dir = displacement / disp_mag
        acceleration = other.G_mass / (disp_mag*disp_mag) * disp_dir

        return acceleration