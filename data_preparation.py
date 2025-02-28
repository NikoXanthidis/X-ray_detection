import os
import cv2  # OpenCV para manipulação de imagens / OpenCV for image processing
import tensorflow as tf  # TensorFlow para criar e treinar redes neurais / TensorFlow to create and train neural networks
import numpy as np  # NumPy para operações matemáticas / NumPy for mathematical operations
from tensorflow.keras import layers, optimizers  # Camadas e otimizadores do Keras / Keras layers and optimizers
from tensorflow.keras.applications import ResNet50  # Importa a ResNet50 pré-treinada / Imports pre-trained ResNet50
from tensorflow.keras.layers import Input, Dense, AveragePooling2D, Dropout, Flatten  # Camadas usadas no modelo / Layers used in the model
from tensorflow.keras.models import Model  # Para definir o modelo personalizado / To define the custom model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Gera imagens aumentadas para o treinamento / Generates augmented images for training
from tensorflow.keras.callbacks import ModelCheckpoint  # Salva os melhores pesos durante o treinamento / Saves best weights during training
import matplotlib.pyplot as plt  # Biblioteca para plotar gráficos / Library to plot graphs
import seaborn as sns  # Biblioteca para visualização de dados / Library for data visualization
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  # Métricas de avaliação / Evaluation metrics

# Define o diretório do conjunto de dados de raios-X / Defines the directory of the X-ray dataset
xray_directory = "project_med/Dataset"

# Carrega o modelo ResNet50 pré-treinado, sem a camada de classificação final / Loads the pre-trained ResNet50 model without the final classification layer
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_tensor=Input(shape=(256, 256, 3)))

# Imprime a estrutura do modelo base (opcional) / Prints the base model structure (optional)
# print(base_model.summary())

# Congela todas as camadas exceto as últimas 10 para fine-tuning / Freezes all layers except the last 10 for fine-tuning
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Adiciona camadas personalizadas sobre a ResNet50 / Adds custom layers on top of ResNet50
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(2, 2))(head_model)  # Camada de pooling para redução de dimensionalidade / Pooling layer for dimensionality reduction
head_model = Flatten()(head_model)  # Achata a saída para tornar compatível com as camadas densas / Flattens the output to be compatible with dense layers
head_model = Dense(256, activation='relu')(head_model)  # Primeira camada densa / First dense layer
head_model = Dropout(0.2)(head_model)  # Dropout para evitar overfitting / Dropout to prevent overfitting
head_model = Dense(256, activation='relu')(head_model)  # Segunda camada densa / Second dense layer
head_model = Dropout(0.2)(head_model)  # Dropout adicional / Additional dropout
head_model = Dense(4, activation='softmax')(head_model)  # Camada de saída com 4 classes e ativação softmax / Output layer with 4 classes and softmax activation

# Cria o modelo final combinando ResNet50 e as novas camadas / Creates the final model combining ResNet50 and new layers
model = Model(inputs=base_model.input, outputs=head_model)

# Compila o modelo definindo a função de perda, otimizador e métrica de acurácia / Compiles the model defining loss function, optimizer, and accuracy metric
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['accuracy'])

# Callback para salvar os melhores pesos do modelo durante o treinamento / Callback to save the best model weights during training
checkpointer = ModelCheckpoint(filepath='weights.keras')

# Gera imagens aumentando os dados de treinamento / Generates images by augmenting training data
image_generator = ImageDataGenerator(rescale=1./255)

# Prepara os dados de treinamento / Prepares training data
train_generator = image_generator.flow_from_directory(batch_size=4, directory=xray_directory, shuffle=True, 
                                                      target_size=(256, 256), class_mode='categorical')

# Treina o modelo por 50 épocas / Trains the model for 50 epochs
history = model.fit(train_generator, epochs=50, callbacks=[checkpointer])

# Plota os gráficos de acurácia e erro durante o treinamento / Plots accuracy and loss graphs during training
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Erro e taxa de acerto durante o treinamento / Error and accuracy rate during training')
plt.xlabel('Época / Epoch')
plt.ylabel('Taxa de acerto e erro / Accuracy and Loss')
plt.legend(['Taxa de acerto / Accuracy', 'Erro / Loss'])
plt.show()

# Define o diretório do conjunto de teste / Defines the test dataset directory
test_directory = "project_med/Test"

test_gen = ImageDataGenerator(rescale=1./255)  # Normaliza os valores dos pixels / Normalizes pixel values
test_generator = test_gen.flow_from_directory(batch_size=40, directory=test_directory, shuffle=True, 
                                              target_size=(256, 256), class_mode='categorical')

# Avalia o modelo no conjunto de teste / Evaluates the model on the test dataset
evaluate = model.evaluate(test_generator)

# Inicializa listas para armazenar previsões e valores reais / Initializes lists to store predictions and true values
prediction = []
original = []
image = []

# Percorre todas as imagens do diretório de teste / Iterates over all images in the test directory
for i in range(len(os.listdir(test_directory))):
    for item in os.listdir(os.path.join(test_directory, str(i))):
        # Carrega e processa a imagem / Loads and processes the image
        img = cv2.imread(os.path.join(test_directory, str(i), item))
        img = cv2.resize(img, (256, 256))
        image.append(img)
        img = img / 255  # Normaliza a imagem / Normalizes the image
        img = img.reshape(-1, 256, 256, 3)  # Ajusta a dimensão da imagem para o modelo / Adjusts image dimension for the model
        predict = model.predict(img)  # Faz a previsão da classe da imagem / Predicts the image class
        predict = np.argmax(predict)  # Obtém a classe com maior probabilidade / Gets the class with the highest probability
        prediction.append(predict)  # Adiciona a previsão à lista / Adds prediction to the list
        original.append(i)  # Adiciona a classe real à lista / Adds true class to the list

# Calcula a matriz de confusão / Computes the confusion matrix
cm = confusion_matrix(original, prediction)
sns.heatmap(cm, annot=True)  # Plota a matriz de confusão / Plots the confusion matrix

# Exibe o relatório de classificação / Displays the classification report
print(classification_report(original, prediction))
