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

## 🛠 Tech Stack
- **Python**
- **LangChain**
- **HuggingFace Models**
- **FAISS Vector Store**
- **python-dotenv**
- **pypdf**
- **ReportLab**

---

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

