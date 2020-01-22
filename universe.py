import torch
import copy


class Universe:
    def __init__(self, planets):
        self.planets = copy.deepcopy(planets)
        self.num_planets = len(planets)
        self.num_dims = len(planets[0].position)

        # This will be used to fix the divide-by-zero problem the arrises when
        # calculating a planet's acceleration due to its own gravity. It should
        # be 0 in those cases, but with Newton's formula, we would get infinity
        # since a planet's distance to itself is 0. So we add # infinity to the
        # denominator of the calculations so that we get GM/inf = 0
        self.inf_diagonal = torch.zeros([len(planets), len(planets), 1], dtype=planets[0].dtype)
        for i in range(len(planets)):
            self.inf_diagonal[i][i][0] = float('inf')

        self.gm_matrix = torch.tensor([[planet.G_mass] for planet in planets])
        self.positions = torch.stack([planet.position for planet in planets])
        # self.pos_matrix = self.positions.expand(self.num_planets, self.num_planets, self.num_dims)
        self.velocities = torch.stack([planet.velocity for planet in planets])

        # Some integrators expect to have an existing accelerations matrix
        self.accelerations = self.calc_accelerations()

    # Calculate accelerations on all planets in the universe.
    # The calculations are performed as combined tensor operations in PyTorch,
    # rather than performing python loops over all the planet pairs. This allows
    # us to attain significantly higher performance because the loops and
    # math are offloaded to compiled C++ code, and because PyTorch Tensors are
    # arranged optimally in memory
    def calc_accelerations(self, positions=None):
        if positions is None:
            positions = self.positions

        pos_matrix = positions.expand(self.num_planets, self.num_planets, self.num_dims)

        dist_matrix = pos_matrix - pos_matrix.transpose(0, 1)

        # thought: it could end up being more accurate to avoid reducing
        # accel_components, and use it to accumulate the velocity components
        # due to each individual interaction

        dist_magnitude = dist_matrix.norm(dim=2, keepdim=True) + self.inf_diagonal
        # dist_direction = dist_matrix / dist_magnitude
        dist_matrix /= dist_magnitude

        # denom = 1 / dist_magnitude
        # denom *= denom

        # acceleration_components = self.gm_matrix / (dist_magnitude * dist_magnitude) * dist_direction
        # acceleration_components = self.gm_matrix * denom * dist_matrix

        acceleration_components = self.gm_matrix / (dist_magnitude * dist_magnitude) * dist_matrix

        # acceleration_components = self.gm_matrix

        accelerations = acceleration_components.sum(dim=1)

        return accelerations

    # Naive implementation of universal acceleration calculation.
    # This is only meant to be a double-checking mechanism to verify
    # the matrix version, as well as a demonstration of the superior
    # performance that the matrix version gives. This simple version
    # is about 4x slower if there are 3 planets, and it becomes
    # worse with more planets
    def calc_accelerations_no_matrix(self):
        accelerations = []
        for planet_idx, planet in enumerate(self.planets):
            acceleration = torch.zeros(self.num_dims, dtype=planet.dtype)
            for other_idx, other in enumerate(self.planets):
                if planet_idx != other_idx:
                    acceleration += planet.calc_accel(other)

            # acceleration[2] = 0

            accelerations.append(
                acceleration
            )

        return torch.stack(accelerations)

    # http://www.physics.udel.edu/~bnikolic/teaching/phys660/numerical_ode/node2.html
    def step_euler_cromer(self, time_step):
        accelerations = self.calc_accelerations()
        self.velocities += accelerations * time_step
        self.positions += self.velocities * time_step

    def step_euler(self, time_step):
        accelerations = self.calc_accelerations()
        self.positions += self.velocities * time_step
        self.velocities += accelerations * time_step

    # http://www.physics.udel.edu/~bnikolic/teaching/phys660/numerical_ode/node5.html

    # Experiementation shows that precision can often be increased
    def step_verlet(self, time_step):
        accelerations_1 = self.accelerations
        self.positions += self.velocities * time_step + accelerations_1 * (time_step*time_step*0.5)
        self.accelerations = self.calc_accelerations()

        accelerations_2 = self.accelerations
        self.velocities += 0.5 * (accelerations_2 + accelerations_1) * time_step

    # gives the same exact result as step_verlet
    def step_verlet_leapfrog(self, time_step):
        velocities_halfstep = self.velocities + self.accelerations * (time_step * 0.5)

        self.positions += velocities_halfstep * time_step

        self.accelerations = self.calc_accelerations()
        self.velocities = velocities_halfstep + self.accelerations * (time_step * 0.5)



    # Using method in http://spiff.rit.edu/richmond/nbody/OrbitRungeKutta4.pdf
    # https://github.com/nicokuijpers/SolarSystemSimulator/blob/master/src/main/java/particlesystem/Particle.java
    def step_runge_kutta_4(self, time_step):
        time_step = torch.tensor(time_step, dtype=self.planets[0].dtype)
        half_time_step = time_step * 0.5

        k1_velocities = self.calc_accelerations()
        k1_positions = self.velocities.clone()

        k2_velocities = self.calc_accelerations(self.positions + k1_positions * half_time_step)
        k2_positions = self.velocities * k1_velocities * half_time_step


        k3_velocities = self.calc_accelerations(self.positions + k2_positions * half_time_step)
        k3_positions = self.velocities * k2_velocities * half_time_step

        k4_velocities = self.calc_accelerations(self.positions + k3_positions * time_step)
        k4_positions = self.velocities * k3_velocities * time_step

        self.velocities += (k1_velocities + 2*k2_velocities + 2*k3_velocities + k4_velocities) * (time_step / 6.0)
        self.positions += (k1_positions + 2*k2_positions + 2*k3_positions + k4_positions) * (time_step / 6.0)


