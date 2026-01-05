import numpy as np

def eigenvalues(Lambda: np.ndarray) -> np.ndarray:
    return np.linalg.eigvals(Lambda)

def is_stable(Lambda: np.ndarray) -> bool:
    eigs = eigenvalues(Lambda)
    return np.all(np.abs(eigs) < 1.0)

def memory_time_constants(Lambda: np.ndarray) -> np.ndarray:
    eigs = np.abs(eigenvalues(Lambda))
    eps = 1e-8
    return 1.0 / np.maximum(1.0 - eigs, eps)

def stability_report(Lambda: np.ndarray) -> dict:
    eigs = eigenvalues(Lambda)
    return {
        "eigenvalues": eigs,
        "max_abs_eigenvalue": np.max(np.abs(eigs)),
        "stable": np.all(np.abs(eigs) < 1.0),
        "memory_time_constants": memory_time_constants(Lambda)
    }
