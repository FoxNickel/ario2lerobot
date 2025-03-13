import numpy as np

proprioception = np.arange(1, 59)
print(proprioception)
if proprioception.shape[0] < 60:
    proprioception = np.insert(proprioception, -1, 0)
    proprioception = np.insert(proprioception, 28, 0)
print(proprioception)


proprioception_floating_base = np.arange(1, 4)
print(proprioception_floating_base)
if proprioception_floating_base.shape[0] < 4:
    proprioception_floating_base = np.insert(
        proprioception_floating_base,
        -1,
        np.zeros(4 - proprioception_floating_base.shape[0]),
    )
print(proprioception_floating_base)

action = np.arange(1, 16)
if action.shape[0] < 16:
    action = np.insert(action, 2, np.zeros(16 - action.shape[0]))
print(action)
