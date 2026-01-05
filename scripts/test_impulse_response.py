# python -m scripts.test_impulse_response



import numpy as np
from src.math.state_space import StateSpaceModel

T = 500
ssm = StateSpaceModel(state_dim=16, input_dim=1)

x = np.zeros((T, 1))
x[10, 0] = 1.0

states = ssm.run(x)

print(states[10:30, 0])
