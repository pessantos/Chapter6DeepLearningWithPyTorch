import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CyclicLR
import numpy as np
import matplotlib.pyplot as plt

# Gerar dados simulados
torch.manual_seed(42)
X = torch.rand(100, 1) * 10
y = 3 * X + torch.randn(100, 1) * 2

# Dividir em treino e validação
train_X, val_X = X[:80], X[80:]
train_y, val_y = y[:80], y[80:]

# Modelo simples de regressão linear
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearModel()

# Função de perda
criterion = nn.MSELoss()

# Otimizadores
adam_optimizer = optim.Adam(model.parameters(), lr=0.01)
sgd_optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Schedulers
adam_scheduler = StepLR(adam_optimizer, step_size=10, gamma=0.5)
sgd_scheduler = CyclicLR(sgd_optimizer, base_lr=0.01, max_lr=0.1, step_size_up=5, mode="triangular")

# Treinamento
def train_model(optimizer, scheduler, epochs, use_scheduler=True):
    for epoch in range(epochs):
        # Fase de treinamento
        model.train()
        optimizer.zero_grad()
        preds = model(train_X)
        loss = criterion(preds, train_y)
        loss.backward()
        optimizer.step()
        
        # Fase de validação
        model.eval()
        with torch.no_grad():
            val_preds = model(val_X)
            val_loss = criterion(val_preds, val_y)
        
        # Atualização do scheduler
        if use_scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Exibir progresso
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    return model

# Treinamento com Adam
print("Treinando com Adam e StepLR Scheduler:")
trained_model_adam = train_model(adam_optimizer, adam_scheduler, epochs=20)

# Treinamento com SGD
print("\nTreinando com SGD e CyclicLR Scheduler:")
trained_model_sgd = train_model(sgd_optimizer, sgd_scheduler, epochs=20)

# Visualização
plt.figure(figsize=(10, 6))
plt.scatter(X.numpy(), y.numpy(), label="Dados")
plt.plot(X.numpy(), trained_model_adam(train_X).detach().numpy(), label="Modelo Adam", color="orange")
plt.plot(X.numpy(), trained_model_sgd(train_X).detach().numpy(), label="Modelo SGD", color="green")
plt.legend()
plt.title("Resultados do Modelo")
plt.show()
