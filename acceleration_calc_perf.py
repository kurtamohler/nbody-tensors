#!/usr/bin/env python3
import torch
from torch import nn
import time
import copy
from body import Body
from universe import Universe

torch.set_num_threads(1)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

dtype = torch.float64

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
    moon,
    earth,
    mercury,
    venus,
    mars,
    jupiter,
    saturn,
]

# would be nice if i could jit it, but norm() prevents that
# accel_calc = torch.jit.script(AccelCalculator())
# accel_calc = AccelCalculator(planets, dtype=dtype)
universe = Universe(planets)
# accel_calc = AccelCalculator(planets, dtype=dtype)
print('Result with 8 actual planets:')
print(universe.calc_accelerations())


for i in range(100):
    planets.append(Body(
        'random_planet',
        torch.rand(3, dtype=dtype),
        torch.rand(3, dtype=dtype),
        torch.rand(1, dtype=dtype),
        (255,255,155)
    ))

# accel_calc = AccelCalculator(planets, dtype=dtype)
universe = Universe(planets)

warmup_iters = 1000
timed_iters = 2000

for i in range(warmup_iters):
    universe.calc_accelerations()

start_time = time.time()
for i in range(timed_iters):
    universe.calc_accelerations()
total_time = time.time() - start_time

time_per_iter = total_time / timed_iters

print()
print('%e s for %d planets' % (time_per_iter, len(planets)))
print('%f iters per second' % (1/time_per_iter))
