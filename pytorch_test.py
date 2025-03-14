import torch
import random
if torch.backends.mps.is_available():
    mps_device = torch.device('mps')
    x = torch.ones(2, 2, device=mps_device)
    print(x)
else:
    print("MPS device not found")


import timeit

x = torch.ones(50000000, device='mps')
print('mps')
print(timeit.timeit(lambda:x*random.randint(0,100), number=1))


x = torch.ones(50000000, device='cpu')
print('cpu')
print(timeit.timeit(lambda:x*random.randint(0,100), number=1))

