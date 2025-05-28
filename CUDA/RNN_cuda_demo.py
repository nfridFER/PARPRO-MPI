import torch
import torch.nn as nn
import numpy as np
import time

def generate_data(seq_length, num_sequences):
    x = np.linspace(0, num_sequences * 2 * np.pi, (seq_length + 1) * num_sequences)
    y = np.sin(x)
    data = []
    for i in range(num_sequences):
        start = i * (seq_length + 1)
        seq_x = y[start:start + seq_length]
        seq_y = y[start + 1:start + seq_length + 1]
        if len(seq_x) == seq_length and len(seq_y) == seq_length:
            data.append((seq_x, seq_y))
    return data

class SineRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(SineRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)

def run_training(device, x_data, y_data, hidden_size=32, num_layers=1, epochs=50, lr=0.01):
    x_tensor = torch.tensor(x_data).unsqueeze(-1).to(device)
    y_tensor = torch.tensor(y_data).unsqueeze(-1).to(device)
    model = SineRNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    duration = time.time() - start_time

    model.eval()
    with torch.no_grad():
        final_output = model(x_tensor)
        final_loss = criterion(final_output, y_tensor).item()

    return duration, final_loss

def main():
    seq_length = 50
    num_sequences = 100
    hidden_size = 32
    epochs = 50

    data = generate_data(seq_length, num_sequences)
    x_data = np.array([x for x, _ in data], dtype=np.float32)
    y_data = np.array([y for _, y in data], dtype=np.float32)

    print("Izvodenje na CPU...")
    cpu_device = torch.device("cpu")
    cpu_time, cpu_loss = run_training(cpu_device, x_data, y_data, hidden_size, epochs=epochs)
    print(f"CPU vrijeme: {cpu_time:.2f}s\tLoss: {cpu_loss:.6f}\n")

    if torch.cuda.is_available():
        print("Izvodenje na GPU...")
        gpu_device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()
        gpu_time, gpu_loss = run_training(gpu_device, x_data, y_data, hidden_size, epochs=epochs)
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
    else:
        gpu_time, gpu_loss, peak_mem = None, None, None

    
    if gpu_time is not None:
        print(f"GPU vrijeme: {gpu_time:.2f}s\tLoss: {gpu_loss:.6f}")
        print(f"GPU memorija: {peak_mem:.2f} MB")
        speedup = cpu_time / gpu_time
        print(f"\nUbrzanje: {speedup:.2f}x")
    else:
        print("Nema GPU")

if __name__ == "__main__":
    main()
