## 🛠️ Tech Stack

### Frontend

- React.js
- JavaScript
- HTML5
- CSS3
- Axios

### Backend

- Python
- FastAPI
- Pydantic
- Uvicorn
- JWT Authentication
- Passlib

### Database

- MongoDB
- Motor

### Machine Learning

- Python
- Pandas
- Scikit-learn

### Tools

- Git
- GitHub
- VS Code
- Postman


## 🏗️ Project Architecture

```text
┌──────────────────────┐
│      React.js        │
│      Frontend        │
└──────────┬───────────┘
           │
           │ REST API
           ▼
┌──────────────────────┐
│       FastAPI        │
│       Backend        │
│                      │
│ Authentication       │
│ Loan Prediction      │
│ Prediction History   │
│ Dashboard            │
└───────┬───────┬──────┘
        │       │
        │       │
        ▼       ▼
   ┌────────┐ ┌──────────────┐
   │MongoDB │ │ ML Model     │
   │        │ │ Prediction   │
   └────────┘ └──────────────┘




### Project description bhi update karo

```markdown
# 💳 Loan Decision Intelligence

Loan Decision Intelligence is a full-stack web application that helps analyze and predict loan approval decisions using Machine Learning.

The application provides a React-based frontend and FastAPI backend. Users can securely register and log in, submit loan application details, receive an ML-based loan prediction with probability scores, and view their previous prediction history.

The application stores user and prediction history data in MongoDB.



## ✨ Key Features

- 🔐 User Registration & Login
- 🔑 JWT-based Authentication
- 🔒 Secure Password Hashing
- 🤖 ML-based Loan Prediction
- 📊 Loan Approval Probability
- 📜 Prediction History
- 👤 User-specific History
- 📈 Dashboard Statistics
- ⚡ React.js Frontend
- 🚀 FastAPI REST API
- 🗄️ MongoDB Database
- 📚 Swagger API Documentation
