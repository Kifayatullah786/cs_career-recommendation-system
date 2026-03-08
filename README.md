#  CS Career Recommendation System

A **Machine Learning–based web application** that recommends suitable career paths for Computer Science students based on their academic performance, programming skills, and project experience.

The system analyzes student profiles and predicts the most appropriate **career group** using a trained Random Forest machine learning model. The application is built with Streamlit to provide an interactive and user-friendly web dashboard.

---

#  Project Overview

Choosing the right career path in computer science can be difficult because the field contains many specializations. This project helps students identify the most suitable career domain by analyzing their academic and technical skill profiles using machine learning.

The system predicts the best career path and also shows a **confidence score**, which indicates how confident the model is about its prediction.

---

#  Features

- Interactive web-based dashboard
- Student profile input form
- Machine learning–based career prediction
- Confidence score for prediction reliability
- Data preprocessing and feature engineering
- Clean and simple user interface
- Cloud deployment using Streamlit

---

#  Input Features

The system uses the following student attributes:

- Gender  
- Age  
- GPA  
- Number of Projects  
- Python Skill Level (Weak / Average / Strong)  
- SQL Skill Level (Weak / Average / Strong)  
- Java Skill Level (Weak / Average / Strong)  

### Derived Feature
- **Total Skill Score** (combined score of Python, SQL, and Java skills)

---

#  Predicted Career Groups

The model predicts one of the following career categories:

- AI & Data  
- Software Development  
- Security  
- Cloud & DevOps  
- Emerging Tech  
- Research & Advanced  

---

# ⚙️ Technologies Used

**Programming & Frameworks**

- Python  
- Streamlit  

**Machine Learning & Data Processing**

- Scikit-learn  
- Pandas  
- NumPy  
- Joblib  

**Visualization**

- Plotly  

---

#  Machine Learning Model

The project uses a **Random Forest Classifier** for career prediction.

Reasons for choosing this model:

- Handles categorical and numerical data effectively  
- Reduces overfitting using ensemble learning  
- Provides reliable and accurate predictions  

---

#  How the System Works

1. The user enters their profile details in the dashboard.
2. Input data is preprocessed and converted into numerical format.
3. The trained machine learning model analyzes the input features.
4. The model predicts the most suitable career group.
5. The system displays the predicted career along with a confidence score.

---

---

#  Deployment

The application is deployed using **Streamlit Community Cloud**.

Steps for deployment:

1. Upload the project to GitHub
2. Connect the repository with Streamlit Cloud
3. Deploy the app using `app.py` as the main file

---

#  Author

Kifayatullah  
Information Teachnology Student

---

#  License

This project is developed for **educational and academic purposes**.
