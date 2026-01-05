import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.models.gated_ssm import GatedSSM
from src.data.long_context import copy_task

device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters
input_dim = 1
hidden_dim = 16
output_dim = 1
batch_size = 4
sequence_lengths = [20, 50, 100, 200, 500]

# Model
model = GatedSSM(input_dim, hidden_dim, output_dim).to(device)
model.eval()

def evaluate_memory(model, seq_lengths, batch_size):
    results = {}
    for L in seq_lengths:
        print(f"Evaluating sequence length: {L}")
        X_test, Y_test = copy_task(batch_size, L, input_dim)
        X_test, Y_test = X_test.to(device), Y_test.to(device)

        with torch.no_grad():
            out_test = model(X_test)
            mse = nn.MSELoss()(out_test, Y_test).item()
            results[L] = mse

    return results

# Run evaluation
memory_results = evaluate_memory(model, sequence_lengths, batch_size)

# Print results
print("\nMemory performance (MSE) across sequence lengths:")
for L, mse in memory_results.items():
    print(f"Length {L}: MSE {mse:.6f}")

# Plot
plt.figure(figsize=(8,5))
plt.plot(list(memory_results.keys()), list(memory_results.values()), marker='o')
plt.xlabel("Sequence Length")
plt.ylabel("MSE")
plt.title("Memory Decay Performance")
plt.grid(True)
plt.show()
