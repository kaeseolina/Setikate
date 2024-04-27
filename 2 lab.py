import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import random


input_size = 28 * 28  # Размер изображения
output_size = 4  # Количество символов
# Описание работы
print("Количество входных сигналов =", input_size)
print("Количество выходных сигналов =", output_size)
print("Алгоритм обучения: Однослойный персептрон, метод стохастического градиентного спуска")
print("")

# Загрузка данных MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Определение архитектуры нейронной сети
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, 10)  # Входной слой: 28*28 пикселей, выходной слой: 10 классов

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

# Инициализация нейронной сети
net = NeuralNetwork()

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Обучение нейронной сети
print("\nОбучение нейронной сети:")
for epoch in range(5):  # Количество эпох
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Эпоха {epoch + 1}, Loss: {running_loss / len(trainloader)}")

# Оценка точности на тестовом наборе
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nТочность на тестовом наборе: {100 * correct / total}%")

# Тестирование на случайном изображении
print("\nТестирование на случайном изображении:")
random_index = random.randint(0, len(testset))
image, label = testset[random_index]
output = net(image.unsqueeze(0))  # Размерность изображения: (1, 1, 28, 28)
_, predicted = torch.max(output, 1)
print(f"Истинная метка: {label}, Предсказанная метка: {predicted.item()}")
