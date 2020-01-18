#!/usr/bin/env python3
import torch
import timeit

torch.set_num_threads(1)

class Planet:
    def __init__(self, name, init_pos, init_vel, G_mass, dtype=torch.float64):
        self.dtype = dtype
        self.position = torch.tensor(init_pos, dtype=self.dtype)     # km, km, km
        self.velocity = torch.tensor(init_vel, dtype=self.dtype)     # km/s, km/s, km/s
        self.G_mass = torch.tensor(G_mass, dtype=self.dtype)      # km^3/s^2
        self.name = name

    def calc_accel(self, other):
        displacement = other.position - self.position
        disp_mag = displacement.norm()
        acceleration = other.G_mass / (disp_mag*displacement)

        return acceleration

class Universe:
    def __init__(self, planets):
        self.planets = planets
        self.num_planets = len(planets)
        self.num_dims = len(planets[0].position)

        # This will be used to fix the divide-by-zero problem the arrises when
        # calculating a planet's acceleration on itself. It should be 0 in those
        # cases, so we add infinity to the distance so that we get GM/inf = 0
        self.inf_diagonal = torch.zeros([len(planets), len(planets), 1], dtype=planets[0].dtype)
        for i in range(len(planets)):
            self.inf_diagonal[i][i][0] = float('inf')

        self.gm_matrix = torch.tensor([[planet.G_mass] for planet in planets])
        self.positions = torch.stack([planet.position for planet in planets])

    def calc_accelerations(self):
        pos_matrix = self.positions.expand(self.num_planets, self.num_planets, self.num_dims)

        dist_matrix = self.inf_diagonal + pos_matrix - pos_matrix.transpose(0, 1)

        # thought: it could end up being more accurate to avoid reducing
        # accel_components, and use it to accumulate the velocity components
        # due to each individual interaction
        accel_components = self.gm_matrix / ((dist_matrix * dist_matrix.norm(dim=2, keepdim=True)))

        accelerations = accel_components.sum(dim=1)

        return accelerations

    # Naive implementation of universal acceleration calculation.
    # This is only meant to be a double-checking mechanism to verify
    # the matrix version, as well as a demonstration of the superior
    # performance that the matrix version gives. This simple version
    # is about 4x slower if there are 3 planets, and it becomes
    # worse with more planets
    def calc_accelerations_no_matrix(self):
        accelerations = []
        for planet_idx, planet in enumerate(planets):
            acceleration = torch.zeros(self.num_dims, dtype=planet.dtype)
            cur_accel = 0
            for other_idx, other in enumerate(planets):
                if planet_idx != other_idx:
                    acceleration += planet.calc_accel(other)

            accelerations.append(
                acceleration
            )

        return torch.stack(accelerations)



# Using NASA Horizons data on A.D. 2020-Jan-17 00:00:00.0000 TDB
sun = Planet(
    'Sun',
    [0,0,0],
    [0,0,0],
    132712440041.93938
)

earth = Planet(
    'Earth',
    [-6.460466571450332E+07,1.322145017754471E+08,-6.309428925409913E+03],
    [-2.725423398965551E+01,-1.317899134460998E+01,8.656734598035953E-04],
    398600.435436
)

jupiter = Planet(
    'Jupiter',
    [9.645852849809307E+07,-7.751822294647597E+08,1.061595873793304E+06],
    [1.281982873892912E+01,2.230656764808971E+00,-2.962161287606510E-01],
    126686534.911
)


planets = [
    sun,
    earth,
    jupiter,
]

universe = Universe(planets)


calc_method_diff = (universe.calc_accelerations() - universe.calc_accelerations_no_matrix()).norm()

if calc_method_diff >= 1e-12:
    print("Error: either matrix or naive acceleration method is wrong")
    exit(1)


for calc_method in ['universe.calc_accelerations()', 'universe.calc_accelerations_no_matrix()']:
    num_iters = 10000
    time_per_iter = torch.mean(torch.tensor(timeit.repeat(
        calc_method,
        'from __main__ import universe',
        repeat = 3,
        number=num_iters
    )[1:])) / num_iters

    print(calc_method)
    print('time per iter: %f s' % time_per_iter)
    print('iters per second: %f fps' % (1/time_per_iter))
    print()

