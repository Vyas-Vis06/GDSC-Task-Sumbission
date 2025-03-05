# GDSC Submissions - Read me 
Task 1: CIFAR - 10 Image Recognation

CIFAR-10 Image Classification using CNN & VGG16
This project trains two deep learning models to classify images from the CIFAR-10 dataset:

A Custom CNN Model (built from scratch)
A VGG16 Transfer Learning Model (pre-trained on ImageNet)
Both models are trained and evaluated to compare their performance.

Dataset
The CIFAR-10 dataset contains 60,000 images (32x32 pixels) across 10 classes:
Car, Airplane, Frog, Dog, Cat, Other animals, etc.

50,000 images for training
10,000 images for testing
Each image has 3 color channels (RGB)
Project Structure
cifar10_classification.py → Main training script
README.md → This file (explains everything)
models/ → Folder to save trained models (optional)
results/ → Stores graphs of accuracy/loss curves (optional)
How It Works
Load & Preprocess Data

Download CIFAR-10 dataset
Normalize images (convert pixel values from 0-255 to 0-1)
Convert labels into one-hot encoding
Data Augmentation

Adds random shifts & flips to improve training
Train Two Models:

CNN Model (Built from Scratch)
3 Convolutional layers (Conv2D)
MaxPooling, Dropout, Flatten, Dense layers
VGG16 Transfer Learning Model
Uses a pre-trained VGG16 model
Adds extra Dense layers for classification
Freezes early layers to keep learned features
Evaluate Models

Compares accuracy & loss for both models
Plots accuracy/loss curves
Prints classification reports (Precision, Recall, F1-Score)
How to Run the Code
On Google Colab
Open Google Colab
Upload the script
Run it (GPU recommended)
On Your Local Machine
Install required libraries:
bash
Copy
Edit
pip install tensorflow numpy matplotlib scikit-learn
Run the script:
bash
Copy
Edit
python cifar10_classification.py
Results
The VGG16 model generally achieves higher accuracy than the custom CNN model.
The custom CNN model takes longer to train but is more flexible.
Why Use Transfer Learning?
VGG16 is already trained on millions of images, so it learns faster.
Freezing its layers helps keep important features.
We only train the last few layers to adapt to CIFAR-10.
Future Improvements
Try ResNet or MobileNet instead of VGG16
Experiment with batch sizes, optimizers, and dropout rates
Use more data augmentation for better generalization


Task 2: AI Agent Using Langchain

AI Agent with RAG – Simple ReadMe
This project is an AI-powered assistant that can:
Read and process files (PDF & Text)
Answer questions based on file content (RAG – Retrieval-Augmented Generation)
Perform basic math (addition, subtraction, etc.)
Summarize text automatically
Search the web for extra information

How It Works
Upload a file (PDF or TXT)
The AI reads & stores the content
Ask it questions about the file!
It retrieves the best matching parts of the file
Uses GPT-4 to generate an answer

How to Run This in Google Colab
Open Google Colab 🔗 https://colab.research.google.com/
Upload this script and run the code
Install the necessary libraries (they're included in the script)
Set your OpenAI API Key and SERP API Key
Upload a file using upload_and_process_file()
Ask questions using query_agent("Your question here")
Get web search results with agent.run("Your search query")

Requirements (Installed in Code)
langchain (AI tools)
openai (ChatGPT API)
faiss-cpu (Fast search for stored documents)
google-search-results (For web search)
pypdf (For reading PDFs)
