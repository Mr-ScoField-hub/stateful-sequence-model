import numpy as np

class StateSpaceModel:
    def __init__(self, state_dim: int, input_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)

        self.state_dim = state_dim
        self.input_dim = input_dim

        eigenvalues = rng.uniform(low=-0.9, high=0.9, size=state_dim)
        self.Lambda = np.diag(eigenvalues)

        self.B = rng.normal(scale=0.1, size=(state_dim, input_dim))

        self.h = np.zeros(state_dim)

    def reset(self):
        self.h = np.zeros(self.state_dim)

    def step(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.input_dim:
            raise ValueError("Input dimension mismatch")

        self.h = self.Lambda @ self.h + self.B @ x
        return self.h

    def run(self, inputs: np.ndarray) -> np.ndarray:
        T = inputs.shape[0]
        states = np.zeros((T, self.state_dim))

        for t in range(T):
            states[t] = self.step(inputs[t])

        return states
