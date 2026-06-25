
# 📖 AI Story Generator

An AI-powered Story Generator that creates coherent, creative, and structured stories from user prompts. The project fine-tunes a transformer-based language model on a custom instruction-response dataset containing stories from multiple genres such as fantasy, science fiction, adventure, and moral tales.

🚀 Live Demo

🔗 Hugging Face Space: [https://huggingface.co/spaces/vermadhruv/ai-story-generator]


## 📌 Project Overview

This project was developed as part of the **Transformer Models and Applications** assignment.

The application generates engaging stories based on user-provided prompts. Users can describe characters, settings, or themes, and the model creates a complete story with a proper beginning, middle, and ending.

### Example Prompt

> Create a story about a boy named John who lives near a forest with talking animals.

### Example Output

The model generates a structured and creative story featuring John, the magical forest, and its talking animal inhabitants while maintaining coherence throughout the narrative.

---

## ✨ Features

* Story generation from custom prompts
* Fine-tuned transformer model
* Multiple story genres:

  * Fantasy
  * Adventure
  * Science Fiction
  * Moral Stories
  * Mystery
* Adjustable generation parameters

  * Temperature
  * Top-p Sampling
  * Max Length
* Interactive web interface using Gradio
* Deployed on Hugging Face Spaces

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Datasets
* Gradio
* Hugging Face Spaces

---

## 📂 Dataset Creation

A custom instruction-response dataset was created using Large Language Models (LLMs).

### Dataset Format

```text
### Instruction:
Write a story about a magical forest.

### Story:
Once upon a time...
```

### Dataset Characteristics

* 50–100+ stories
* Diverse themes
* Structured narratives
* Instruction-following format
* High-quality generated content

---

## 🤖 Model Fine-Tuning

### Base Model

* [GPT-2]


### Training Objectives

* Instruction following
* Story coherence
* Creativity enhancement
* Reduced repetition

### Training Process

1. Dataset preprocessing
2. Tokenization
3. Fine-tuning using Hugging Face Transformers
4. Model evaluation
5. Deployment

---

## 🎯 Text Generation

The model uses controlled decoding strategies to improve story quality.

### Parameters

| Parameter          | Purpose               |
| ------------------ | --------------------- |
| Temperature        | Controls creativity   |
| Top-p              | Nucleus sampling      |
| Max Length         | Story length          |
| Repetition Penalty | Reduces repeated text |

---

## 🖥️ Application Interface

The Gradio application allows users to:

* Enter custom prompts
* Generate stories instantly
* Experiment with generation settings
* View formatted story outputs

---

## 📁 Project Structure

```text
AI-Story-Generator/
│
├── app.py
├── train.py
├── dataset/
│   └── stories.json
├── model/
│   └── fine_tuned_model
├── requirements.txt
├── README.md
└── screenshots/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/vermadhruv2004/Transformer-Based-Story-Generator.git
cd AI-Story-Generator
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

## 📊 Sample Results

### Prompt

```text
Create a story about a robot exploring Mars.
```

### Generated Story

```text
A young robot named Nova landed on Mars with a mission to discover signs of life...
```

---

## Challenges Faced

* Creating a diverse and high-quality dataset
* Preventing repetitive story generation
* Maintaining story coherence
* Optimizing model performance within hardware limitations

---

## Future Improvements

* Larger dataset
* Better fine-tuned models
* Story genre selection
* Character memory
* Story continuation feature
* Image generation for stories

---

## 👨‍💻 Author

**Dhruv**

Transformer Models and Applications Assignment

---

## 📜 License

This project is developed for educational and learning purposes.
