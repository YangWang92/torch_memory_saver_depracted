import logging
import sys
import time

import torch

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

from torch_memory_saver import TorchMemorySaver

memory_saver = TorchMemorySaver()

num_devices = torch.cuda.device_count()

normal_tensors = []
pauseable_tensors = []

for device_id in range(num_devices):
    device = f'cuda:{device_id}'
    normal_tensor = torch.full((40_000_000_000,), 100, dtype=torch.uint8, device=device)
    print(f'{normal_tensor=}')
    normal_tensors.append(normal_tensor)
    with memory_saver.region():
        pauseable_tensor = torch.full((40_000_000_000,), 100, dtype=torch.uint8, device=device)
        print(f'{pauseable_tensor=}')
        pauseable_tensors.append(pauseable_tensor)

print(f'{normal_tensors=} {pauseable_tensors=}')

print('before sleep..., wait 10 seconds')
time.sleep(10)

memory_saver.pause()
print('after sleep..., wait 10 seconds')
time.sleep(10)

memory_saver.resume()
print('resume from sleep..., wait 10 seconds')
time.sleep(10)

print(f'{normal_tensor=} {pauseable_tensor=}')
