# 📘 UniMate AI – Intelligent Academic Assistant

## 🧠 Overview
UniMate AI is an advanced AI-powered assistant designed for students and learners.  
It supports conversational AI chat, PDF-based question answering, web-search fallback,  
quiz generation, and exporting chat conversations as PDF.

---

## 🚀 Features
- **AI Chat Assistant** – Ask anything and get structured, helpful responses.  
- **PDF Question-Answering (RAG)** – Upload PDFs and query their content.  
- **Knowledge Retrieval** – Text is processed, chunked, embedded, and stored in a vector database.  
- **Web Search Fallback** – When the model lacks information, online search is used.  
- **Quiz Generator** – Creates MCQs from recent chat history.  
- **Chat Export to PDF** – Download your conversation in a formatted PDF layout.  
- **Contextual Memory** – Maintains chat history for coherent responses.

---

# 💻 Tech Stack:
![C](https://img.shields.io/badge/c-%2300599C.svg?style=for-the-badge&logo=c&logoColor=white) ![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) ![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white) ![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E) ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Scipy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![MySQL](https://img.shields.io/badge/mysql-4479A1.svg?style=for-the-badge&logo=mysql&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Windows Terminal](https://img.shields.io/badge/Windows%20Terminal-%234D4D4D.svg?style=for-the-badge&logo=windows-terminal&logoColor=white)

## 🔐 Environment Variables
The following keys are required (use an `.env` file or environment variables):

```
HUGGINGFACEHUB_API_TOKEN=
GOOGLE_API_KEY=
GOOGLE_CSE_ID=
```

---

## 📂 Project Structure

```
/
├── Home.py
├── pages/
├── vectordb/
├── requirements.txt
├── runtime.txt
└── .env.example
```

---

## 📦 Installation

```bash
git clone <repo-url>
cd <project-folder>
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
python Home.py
```

(Add instructions specific to your runtime environment)

---

## 📝 Usage Guide
1. Upload a PDF (optional)
2. Ask any question
3. View AI responses
4. Generate quizzes from chat history
5. Download your conversation as a PDF

---

## 🧩 Troubleshooting
- Ensure environment variables are set correctly  
- Check that all dependencies in `requirements.txt` are installed  
- Verify that vector database files (if any) are accessible  

---

## 🔮 Future Improvements
- Add multi-document support  
- Add UI themes  
- Improve quiz generation model  
- Add advanced analytics  

---

## 📜 License
Add your license information here.


## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## API Reference

#### Get all items

```http
  GET /api/items
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `api_key` | `string` | **Required**. Your API key |

#### Get item

```http
  GET /api/items/${id}
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `id`      | `string` | **Required**. Id of item to fetch |

#### add(num1, num2)

Takes two numbers and returns the sum.


## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

