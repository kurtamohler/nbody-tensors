#!/usr/bin/env python3
import torch
import timeit
import time
from body import Body
from universe import Universe
import math

torch.no_grad()
torch.set_num_threads(1)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')


#https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/exm/chapters/orbits.pdf

# Using NASA Horizons data on day 2458865.500000000 A.D. 2020-Jan-17 00:00:00.0000 TDB
sun = Body(
    'Sun',
    [0,0,0],
    [0,0,0],
    # 132712440041.93938,
    132712440041.93938,
    (255, 255, 0)
)

earth = Body(
    'Earth',
    [-6.460466571450332E+07,1.322145017754471E+08,-6.309428925409913E+03],
    [-2.725423398965551E+01,-1.317899134460998E+01,8.656734598035953E-04],
    398600.435436,
    (0, 0, 255)
)

jupiter = Body(
    'Jupiter',
    [9.645852849809307E+07,-7.751822294647597E+08,1.061595873793304E+06],
    [1.281982873892912E+01,2.230656764808971E+00,-2.962161287606510E-01],

    126686534.911,

    # mars does crazy shit if you do this:
    # 126686534.911*10000,
    (255, 128, 0)
)

moon = Body(
    'Moon',
    [-6.495364069029525E+07, 1.320947069062943E+08, 2.704999488616735E+04],
    [-2.693992140731323E+01, -1.418789199462139E+01, -1.479866760897597E-02],
    4902.800066,
    (128, 128, 128)
)

mercury = Body(
    'Mercury',
    [4.022180492977770E+07, -4.846664361672945E+07, -7.650162494227177E+06],
    [2.779612781778818E+01, 3.345304308542049E+01, 1.837186214409776E-01],
    22031.78,
    (128, 128, 128)
)

venus = Body(
    'Venus',
    [9.405851148876747E+07, 5.358598615005913E+07, -4.692568261448300E+06],
    [-1.744537890570094E+01, 3.027869039102691E+01, 1.422189780971635E+00],
    324858.592,
    (255, 170, 40)
)

mars = Body(
    'Mars',
    [-1.757523324502042E+08, -1.561139983441896E+08, 1.040907987535134E+06],
    [1.699996457362255E+01, -1.604014612625873E+01, -7.532127316314190E-01],
    42828.375214,
    (200, 20, 20)
)

saturn = Body(
    'Saturn',
    [5.796789943794842E+08, -1.384408860689929E+09, 9.953620049176812E+05],
    [8.389521747848969E+00, 3.706927582928515E+00, -3.988074318732124E-01],
    37931207.8,
    (200, 200, 100)
)

G = 6.67408e-11/1e9

spaceship = Body(
    'spaceship',
    [-6.460466571450332E+07+200_000,1.322145017754471E+08,-6.309428925409913E+03],
    [-2.725423398965551E+01,-1.317899134460998E+01+4e-1+2e-2,8.656734598035953E-04],
    10*G,
    (255, 0, 255)
)


planets = [
    sun,
    moon,
    earth,
    mercury,
    venus,
    mars,
    jupiter,
    saturn,
]

# for i in range(100):
#     planets.append(Body(
#         'random_planet',
#         torch.rand(3, dtype=torch.float64),
#         torch.rand(3, dtype=torch.float64),
#         torch.rand(1, dtype=torch.float64),
#         (255,255,155)
#     ))


universe = Universe(planets)



# calc_method_diff = (universe.calc_accelerations() - universe.calc_accelerations_no_matrix()).norm()

# if calc_method_diff >= 1e-12:
#     print("Error: either matrix or naive acceleration method is wrong")
#     exit(1)



# exit()
import pygame

def measure_forward_back_offset(universe):
    init_positions = universe.positions.clone().detach()

    # Run for 27,397 years at a timestep of 1 day
    time_step = 86400
    num_steps = 10000_000#_000

    for i in range(num_steps):
        universe.step_verlet(time_step)

    # reverse velocities
    universe.velocities *= -1.0

    for i in range(num_steps):
        universe.step_verlet(time_step)

    final_positions = universe.positions.clone().detach()

    diff = final_positions - init_positions

    print(diff.norm(dim=1))


screen_width = 800
screen_height = 800
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
planet_disp_paths_surface = pygame.Surface((screen_width, screen_height))
planet_disp_paths_array = pygame.PixelArray(planet_disp_paths_surface)

# time_step = 86400/100
# time_step = 86400/(101 + 0.01 + 0.001)
# time_step = 86400/(101 + 0.01 + 0.001) # + 0.001)
# time_step = 8640/2
# time_step = 512
time_step = 10000

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how wide each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


steps_per_second = float('inf')
# steps_per_second = 100
seconds_per_step = 1.0/steps_per_second
last_step_time = -seconds_per_step

fps_target = 10
spf = 1.0 / fps_target
last_draw_time = -spf

disp_radius = 1.5e9
# disp_radius = .5e6
# disp_radius = .3e6

# Keep the center on a specific planet
center_pos = universe.positions[0]
# center_pos = universe.positions[1]

iterations = 0

time_accumulated = 0


total_sim_time_seconds =  (2458865.5 - 2422340.5) * 24 * 60 * 60

total_sim_steps = int(total_sim_time_seconds / time_step)

# 2422340.500000000 = A.D. 1920-Jan-17 00:00:00.0000 TDB
expected_positions = torch.tensor([
    # sun
    [0.000000000000000E+00, 0.000000000000000E+00, 0.000000000000000E+00],

    # moon
    [-6.626849562695473E+07, 1.310922227163226E+08, 3.256617483337969E+04],

    # earth
    [-6.613033666313083E+07, 1.314732964370219E+08, 2.271202175004780E+04],

    # mercury
    [-1.533175836729829E+07,-6.798573108924401E+07, -4.136194215143695E+06],

    # venus
    [-1.048231527374212E+08, -2.473914604257485E+07, 5.724416127873755E+06],

    # mars
    [-2.436889511232330E+08, 4.847655732679014E+07, 7.045020416478494E+06],

    # jupiter
    [-5.407427542990354E+08, 5.822943817195526E+08, 9.759206708556920E+06],

    # saturn
    [-1.289002461574291E+09, 5.231024067539215E+08, 4.192350378270259E+07]
])


while iterations < total_sim_steps:
# while True:
    cur_time = time.time()

    # universe.step_verlet_leapfrog(time_step)
    if (cur_time - last_step_time) > seconds_per_step:
        universe.step_verlet(-time_step)

        # accelerate the spaceship toward its velocity direction for each step
        # universe.velocities[-1] += (0.0000001 * time_step)* torch.tensor([1,1,0])
        # # universe.planets[-1].color = (int(255*math.sin(time_accumulated)+127), 255, 0)
        # universe.planets[-1].color = (
        #     int(127*math.sin(time_accumulated/10000000)+128),
        #     int(127*math.sin(time_accumulated/10000000 + 2*0.333*3.14159)+128),
        #     int(127*math.sin(time_accumulated/10000000 + 2*0.666*3.14159)+128)
        # )
        # time_accumulated += time_step

        last_step_time = cur_time
        iterations += 1


        if iterations % 10 == 0:
            if True:
                # Draw orbit paths
                for planet_idx in range(universe.num_planets):
                    x = universe.positions[planet_idx][0] - center_pos[0]
                    y = universe.positions[planet_idx][1] - center_pos[1]

                    x_disp = int(round(translate(x, -disp_radius, disp_radius, 0, screen_height)))
                    y_disp = int(round(translate(y, -disp_radius, disp_radius, 0, screen_height)))

                    if x_disp >= 0 and x_disp < screen_width and y_disp >= 0 and y_disp < screen_height:
                        planet_disp_paths_array[x_disp, y_disp] = universe.planets[planet_idx].color

    # Update display
    if (cur_time - last_draw_time) > spf:
        pygame.pixelcopy.array_to_surface(screen, planet_disp_paths_array)

        for planet_idx in range(universe.num_planets):
            x = universe.positions[planet_idx][0] - center_pos[0]
            y = universe.positions[planet_idx][1] - center_pos[1]

            x_disp = int(round(translate(x, -disp_radius, disp_radius, 0, screen_height)))
            y_disp = int(round(translate(y, -disp_radius, disp_radius, 0, screen_height)))

            if x_disp >= 0 and x_disp < screen_width and y_disp >= 0 and y_disp < screen_height:
                pygame.draw.circle(screen, planets[planet_idx].color, (x_disp, y_disp), 4)

        pygame.display.flip()
        last_draw_time = cur_time

    # time.sleep(min(seconds_per_step, spf))

# add the sun's position to expected values
# expected_positions += universe.positions[0]

universe.positions -= universe.positions[0].clone()
# print(universe.positions)
# print(expected_positions)

# print((universe.positions - expected_positions).norm(dim=1))
# print((universe.positions - expected_positions) / expected_positions)
print(100*(universe.positions - expected_positions).norm(dim=1) / expected_positions.norm(dim=1))

# pygame.quit()
