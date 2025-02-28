This project was created based on a course I took on the IA Expert platform.  

First, the AI is trained using the dataset located in the "Dataset" folder, and then testing is performed to verify accuracy. The test images are stored in the "Test" folder.  
The algorithm that handles both training and testing is a single script, "data_preparation.py", which generates the weights that will be used by the API.  
This API is located in the "backend" folder under the name "main.py". To complete the backend, there is a simple frontend. 
Together, this frontend and backend create a web page where users can upload an image, and the API will classify it into one of 
the AI's categories: "Covid-19", "Normal", "Viral Pneumonia", and "Bacterial Pneumonia".  

Steps to recreate this project:
1. Train the AI using the "data_preparation.py" script and copy the generated "weights.keras" file to the backend folder.  
2. Open a terminal in the backend folder and run the following command "uvicorn main:app --reload". 
3. Open the "index.html" file located in the "frontend" folder.

Esse projeto foi feito baseado em um curso que eu fiz na plataforma IA expert.

Primeiro é feito o treinamento da IA usando a base de dados que fica na pasta "Dataset"
e em seguida o teste para verificar a precisão as imagens de teste ficam na pasta "Teste"
o algoritimo que faz essas duas partes é um só o "data_preparation.py", ele vai gerar os pesos que vai ser usado pela API
essa API fica na pasta backend com o nome de "main.py" e para completar esse backend temos o frontend simples, o conjunto desse front e back criam uma página
onde é possivel caregar uma imagem e a API classificar ela dentro das categorias da IA que são Covid-19", "Normal", "Pneunomia viral", "Pneunomia bacterial.

Então o passo a passo para refazer esse projeto é :
1-treinar a IA no aquivo "data_preparation.py", copie o arquivo gerado "weights.keras" para a pasta do backed.
2-abrir o terminal na pasta do backend e digitar esse comando "uvicorn main:app --reload".
3- abra o "index.html" que fica na pasta "frontend".