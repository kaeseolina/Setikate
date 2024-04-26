import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size, output_size) * 0.006 - 0.003  # Используем диапазон [-0.003, 0.003]
        self.threshold = 0.0
    
    def activation_function(self, x):
        # Пороговая функция активации (ступенчатая функция)
        return np.where(x >= self.threshold, 1, 0)
    
    def forward_pass(self, inputs):
        # Вычисление взвешенной суммы входных сигналов
        weighted_sum = np.dot(inputs, self.weights)
        # Применение функции активации к взвешенной сумме
        return self.activation_function(weighted_sum)
    
    def train(self, inputs, target):
        # Проход вперед
        output = self.forward_pass(inputs)
        # Вычисление ошибки
        error = target - output
        # Коррекция весов через дельта-правило
        self.weights += self.learning_rate * np.outer(inputs, error)
        return error
    
    def test(self, inputs):
        # Проход вперед для тестирования
        return self.forward_pass(inputs)


def generate_training_data(symbol, num_samples, fonts):
    training_data = []
    for i in range(num_samples):
        # Выбор случайного шрифта для символа
        font = random.choice(fonts)
        
        # Создание изображения с символом
        image = Image.new('L', (28, 28), color=255)  # Создаем изображение размером 28x28 пикселей, белого цвета
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), symbol, font=font, fill=0)  # Рисуем символ на изображении
        
        # Преобразование изображения в массив numpy и нормализация значений пикселей к диапазону [0, 1]
        data = np.array(image) / 255.0
        flattened_data = data.flatten()
        # Добавляем данные в виде плоского вектора к обучающим данным, присоединяем метку класса в конце вектора
        training_data.append(np.append(flattened_data, symbols.index(symbol)))
        
        # Сохранение изображения
        image.save(f"{symbol}_{i}.png")  # Сохраняем изображение
    return training_data


def generate_test_data(symbol, font):
    # Создание изображения с символом
    image = Image.new('L', (28, 28), color=255)  # Создаем изображение размером 28x28 пикселей, белого цвета
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), symbol, font=font, fill=0)  # Рисуем символ на изображении
    
    # Преобразование изображения в массив numpy и нормализация значений пикселей к диапазону [0, 1]
    data = np.array(image) / 255.0
    
    # Сохранение изображения
    image.save(f"{symbol}_test.png")  # Сохраняем изображение
    
    return data.flatten()  # Возвращаем данные в виде плоского вектора


# Функция для генерации рандомного изображения с символом
def generate_random_image(symbol, font):
    image = Image.new('L', (28, 28), color=255)  # Создаем изображение размером 28x28 пикселей, белого цвета
    draw = ImageDraw.Draw(image)
    draw.text((5, 5), symbol, font=font, fill=0)  # Рисуем символ на изображении
    
    # Добавляем случайный шум к изображению
    for i in range(100):  # Добавляем 100 случайных пикселей
        x = random.randint(0, 0)  # Случайная координата x
        y = random.randint(0, 0)  # Случайная координата y
        draw.point((x, y), fill=0)  # Закрашиваем пиксель случайным образом
    
    # Преобразование изображения в массив numpy и нормализация значений пикселей к диапазону [0, 1]
    data = np.array(image) / 255.0
    return data.flatten()  # Возвращаем данные в виде плоского вектора


# Создаем список путей к файлам шрифтов для каждого класса символов
font_A_path = r"D:\PYTHON\SETI\fonts\font_A.ttf"
font_B_path = r"D:\PYTHON\SETI\fonts\font_B.ttf"
font_C_path = r"D:\PYTHON\SETI\fonts\font_C.ttf"
font_D_path = r"D:\PYTHON\SETI\fonts\font_D.ttf"

# Загрузка шрифтов для каждого символа
font_A = ImageFont.truetype(font_A_path, 20)
font_B = ImageFont.truetype(font_B_path, 20)
font_C = ImageFont.truetype(font_C_path, 20)
font_D = ImageFont.truetype(font_D_path, 20)

# Список шрифтов для каждого символа
fonts = [font_A, font_B, font_C, font_D]


num_samples_per_symbol = 4
training_data = []
symbols = ['A', 'B', 'C', 'D']
for symbol in symbols:
    training_data.extend(generate_training_data(symbol, num_samples_per_symbol, fonts))

# Создание тестовых данных для всех символов
test_data = {}
for symbol, font in zip(symbols, fonts):
    test_data[symbol] = generate_test_data(symbol, font)

# Генерация случайных тестовых данных
random_test_data = {}
for symbol, font in zip(symbols, fonts):
    random_test_data[symbol] = generate_random_image(symbol, font)

# Перемешиваем обучающие данные
np.random.shuffle(training_data)

# Создание персептрона
input_size = 28 * 28  # Размер изображения
output_size = 4  # Количество символов
learning_rate = 0.1
perceptron = Perceptron(input_size, output_size, learning_rate)

print("Количество входных сигналов =", input_size)
print("Количество выходных сигналов =", output_size)
print("Алгоритм обучения: Однослойный персептрон, метод коррекции ошибки через дельта-правило")
print("")

# Обучение персептрона
num_epochs = 100
for epoch in range(num_epochs):
    total_error = 0
    for data in training_data:
        inputs = data[:-1]  # Входные данные (пиксели изображения)
        target = data[-1]  # Ожидаемый выходной сигнал (индекс символа)
        error = perceptron.train(inputs, target)
        total_error += np.abs(error)
    accuracy = 1 - (total_error / (len(training_data) * output_size))
    print(f"Эпоха {epoch+1}/{num_epochs}, Точность: {accuracy}")

# Тестирование персептрона
correct_predictions = 0
for symbol, data in test_data.items():
    inputs = data
    prediction = perceptron.test(inputs)
    predicted_class = np.argmax(prediction)  # Получаем индекс класса с максимальной вероятностью
    actual_class = symbols.index(symbol)  # Получаем индекс актуального класса
    if predicted_class == actual_class:
        correct_predictions += 1
        print(f"Предсказано: {symbols[predicted_class]}, Фактическое: {symbol}")
    else:
        print(f"Предсказано: {symbols[predicted_class]}, Фактическое: {symbol}")
accuracy = correct_predictions / len(test_data)
print(f"Точность на тесте: {accuracy}")

# Тестирование на случайном изображении
test_input = np.random.randint(0, 2, input_size) # случайный вектор значений
print("Тестовый вектор:", test_input)
prediction = perceptron.test(test_input)
print("Предсказанный символ:", symbols[np.argmax(prediction)])

for symbol, data in random_test_data.items():
    inputs = data
    prediction = perceptron.test(inputs)
    predicted_class = np.argmax(prediction)  # Получаем индекс класса с максимальной вероятностью
    
    # Отображение случайного изображения перед тестированием
    plt.imshow(inputs.reshape(28, 28), cmap='gray')
    plt.title(f"Случайный тест - До: {symbol}")
    plt.show()
    
    # Отображение случайного изображения после предсказания
    plt.imshow(inputs.reshape(28, 28), cmap='gray')
    plt.title(f"Случайный тест - Предсказано: {symbols[predicted_class]}, Фактическое: {symbol}")
    plt.show()
