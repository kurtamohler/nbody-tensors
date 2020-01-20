#!/usr/bin/env python3
import torch
import timeit
import pygame
import time
from body import Body
from universe import Universe

# torch.set_num_threads(1)

#https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/exm/chapters/orbits.pdf

# Using NASA Horizons data on A.D. 2020-Jan-17 00:00:00.0000 TDB
sun = Body(
    'Sun',
    [0,0,0],
    [0,0,0],
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


planets = [
    sun,
    mercury,
    venus,
    moon,
    earth,
    mars,
    jupiter,
    saturn,

]

universe = Universe(planets)


calc_method_diff = (universe.calc_accelerations() - universe.calc_accelerations_no_matrix()).norm()

if calc_method_diff >= 1e-12:
    print("Error: either matrix or naive acceleration method is wrong")
    exit(1)


def compare_acceleration_calc_performance(universe):
    times = []

    for calc_method in ['universe.calc_accelerations()', 'universe.calc_accelerations_no_matrix()']:
        num_iters = 1000
        time_per_iter = torch.mean(torch.tensor(timeit.repeat(
            calc_method,
            'from __main__ import universe',
            repeat = 3,
            number=num_iters
        )[1:])) / num_iters

        times.append(time_per_iter)

        print(calc_method)
        print('time per iter: %f s' % time_per_iter)
        print('iters per second: %f fps' % (1/time_per_iter))
        print()


    print('speedup: %f' % (times[1] / times[0]))

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

time_step = 86400/10

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how wide each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


steps_per_second = float('inf')
# steps_per_second = 40
seconds_per_step = 1.0/steps_per_second
last_step_time = -seconds_per_step

fps_target = 10
spf = 1.0 / fps_target
last_draw_time = -spf

disp_radius = 1.5e9

# Keep the center on a specific planet
center_pos = universe.positions[0]

iterations = 0

while True:
    cur_time = time.time()

    # universe.step_verlet_leapfrog(time_step)
    if (cur_time - last_step_time) > seconds_per_step:
        universe.step_verlet(time_step)
        last_step_time = cur_time
        iterations += 1


        if iterations % 10 == 0:
            # Draw orbit paths
            for planet_idx in range(universe.num_planets):
                x = universe.positions[planet_idx][0] - center_pos[0]
                y = universe.positions[planet_idx][1] - center_pos[1]

                x_disp = int(round(translate(x, -disp_radius, disp_radius, 0, screen_height)))
                y_disp = int(round(translate(y, -disp_radius, disp_radius, 0, screen_height)))

                if x_disp >= 0 and x_disp < screen_width and y_disp >= 0 and y_disp < screen_height:
                    planet_disp_paths_array[x_disp, y_disp] = planets[planet_idx].color

    # Update display
    if (cur_time - last_draw_time) > spf:
        pygame.pixelcopy.array_to_surface(screen, planet_disp_paths_array)

        for planet_idx in range(universe.num_planets):
            x = universe.positions[planet_idx][0] - center_pos[0]
            y = universe.positions[planet_idx][1] - center_pos[1]

            x_disp = int(round(translate(x, -disp_radius, disp_radius, 0, screen_height)))
            y_disp = int(round(translate(y, -disp_radius, disp_radius, 0, screen_height)))

            pygame.draw.circle(screen, planets[planet_idx].color, (x_disp, y_disp), 3)

        pygame.display.flip()
        last_draw_time = cur_time

    time.sleep(min(seconds_per_step, spf))


pygame.quit()