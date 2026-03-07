# ==========================================
# CS CAREER RECOMMENDATION SYSTEM
# FINAL STREAMLIT VERSION WITH VISUALIZATION
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="CS Career Recommender", layout="wide")
# Dashboard Style

page_bg = """
<style>

/* MAIN DASHBOARD BACKGROUND */

[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#f4f7ff,#eaf1ff,#dde8ff);
}

/* SIDEBAR BACKGROUND */

[data-testid="stSidebar"]{
background: linear-gradient(180deg,#5f9cff,#3b82f6,#2563eb);
}

/* SIDEBAR LABEL TEXT */

[data-testid="stSidebar"] label{
color:white !important;
font-weight:600;
}

/* DROPDOWN BOX */

[data-baseweb="select"] > div{
background-color:white !important;
color:black !important;
border-radius:8px;
}

/* DROPDOWN TEXT */

[data-baseweb="select"] span{
color:black !important;
font-weight:500;
}

/* INPUT TEXT */

input{
color:black !important;
}

/* SLIDER VALUE */

[data-testid="stSidebar"] span{
color:white !important;
font-weight:500;
}

/* METRIC CARDS */

[data-testid="metric-container"]{
background-color:white;
border-radius:12px;
padding:15px;
box-shadow:0px 4px 10px rgba(0,0,0,0.1);
}

/* TITLE */

h1{
color:#1e3a8a;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# page_bg = """
# <style>

# [data-testid="stAppViewContainer"]{
# background: linear-gradient(120deg,#f6f9fc,#e9f1ff);
# }

# [data-testid="stSidebar"]{
# background-color:#eef2ff;
# }

# </style>
# """

st.markdown(page_bg, unsafe_allow_html=True)
# LOAD MODELS
# ==========================================

model = joblib.load("career_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")
kmeans = joblib.load("kmeans.pkl")
knn = joblib.load("knn.pkl")
processed_df = joblib.load("processed_df.pkl")
cluster_summary = joblib.load("cluster_summary.pkl")

# ==========================================
# SIDEBAR INPUT
# ==========================================

st.sidebar.header("🎓 Student Profile")

gender = st.sidebar.selectbox("Gender", ["Male","Female"])
age = st.sidebar.slider("Age",18,35,22)
gpa = st.sidebar.slider("GPA",2.0,4.0,3.0)
projects = st.sidebar.slider("Projects",0,15,3)

python_skill = st.sidebar.selectbox("Python",["Weak","Average","Strong"])
sql_skill = st.sidebar.selectbox("SQL",["Weak","Average","Strong"])
java_skill = st.sidebar.selectbox("Java",["Weak","Average","Strong"])

skill_map = {"Weak":1,"Average":2,"Strong":3}

# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

input_dict = {
    "Gender":gender,
    "Age":age,
    "GPA":gpa,
    "Projects":projects,
    "Python":skill_map[python_skill],
    "SQL":skill_map[sql_skill],
    "Java":skill_map[java_skill]
}

input_df = pd.DataFrame([input_dict])

input_df["Total_Skill"] = (
    input_df["Python"] +
    input_df["SQL"] +
    input_df["Java"]
)

input_df["Coding_Strength"] = (
    input_df["Python"]*0.4 +
    input_df["SQL"]*0.3 +
    input_df["Java"]*0.3
)

input_df["Project_Intensity"] = input_df["Projects"] * input_df["GPA"]

input_df = pd.get_dummies(input_df)

input_df = input_df.reindex(columns=feature_columns, fill_value=0)

input_scaled = scaler.transform(input_df)

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(input_scaled)[0]
prediction_label = label_encoder.inverse_transform([prediction])[0]

probabilities = model.predict_proba(input_scaled)[0]
confidence = np.max(probabilities)*100

# ==========================================
# TITLE
# ==========================================
# col1, col2, col3 = st.columns([1,3,1])

# with col1:
#     st.image("logo_left.png", width=120)

# with col3:
#     st.image("logo_right.png", width=120)

# st.markdown(
#     "<h1 style='text-align:center;'>🎓 CS Career Recommendation System</h1>",
#     unsafe_allow_html=True
# )


st.title("🎓 CS Career Recommendation System")

# ==========================================
# PREDICTION DISPLAY
# ==========================================

col1,col2 = st.columns(2)

with col1:
    st.subheader("🎯 Recommended Career Group")
    st.metric("Predicted Career",prediction_label)

with col2:
    st.subheader("📊 Confidence")
    st.metric("Model Confidence",f"{confidence:.2f}%")

st.markdown("---")

# ==========================================
# CAREER PROBABILITY DISTRIBUTION
# ==========================================
st.subheader("📈 Career Probability Distribution")

prob_df = pd.DataFrame({
    "Career Group": label_encoder.classes_,
    "Probability (%)": probabilities * 100
}).sort_values(by="Probability (%)", ascending=False)

fig_prob = px.bar(
    prob_df,
    x="Career Group",
    y="Probability (%)",
    text=prob_df["Probability (%)"].apply(lambda x: f"{x:.1f}%"),
    color="Career Group",
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig_prob.update_traces(textposition="outside")

fig_prob.update_layout(
    xaxis_tickangle=-40,
    yaxis_title="Probability (%)",
    showlegend=False
)

st.plotly_chart(fig_prob, use_container_width=True)
# st.subheader("📈 Career Probability Distribution")

# prob_df = pd.DataFrame({
#     "Career Group": label_encoder.classes_,
#     "Probability (%)": probabilities * 100
# }).sort_values(by="Probability (%)", ascending=False)

# fig_prob = px.bar(
#     prob_df,
#     x="Career Group",
#     y="Probability (%)",
#     text=prob_df["Probability (%)"].apply(lambda x: f"{x:.1f}%"),
#     color="Probability (%)"
# )

# fig_prob.update_traces(textposition="outside")

# fig_prob.update_layout(
#     xaxis_tickangle=-40,
#     yaxis_title="Probability (%)"
# )

# st.plotly_chart(fig_prob, use_container_width=True)

st.subheader("📡 Skill Radar")

radar_fig = go.Figure()

radar_fig.add_trace(go.Scatterpolar(
    r=[
        skill_map[python_skill],
        skill_map[sql_skill],
        skill_map[java_skill]
    ],
    theta=["Python","SQL","Java"],
    fill='toself'
))

radar_fig.update_layout(
    polar=dict(radialaxis=dict(visible=True,range=[0,3])),
    showlegend=False
)

st.plotly_chart(radar_fig,use_container_width=True)

# ==========================================
# DATASET OVERVIEW
# ==========================================

with st.expander("📂 Dataset Overview (Basic Statistics)"):
    
    st.write("Dataset Shape:",processed_df.shape)

    st.write("Columns")
    st.write(processed_df.columns)

    st.write("Sample Data")
    st.dataframe(processed_df.head())

    career_counts = processed_df["Career_Group"].value_counts()

    fig = px.pie(
        values=career_counts.values,
        names=career_counts.index,
        title="Career Distribution in Dataset"
    )

    st.plotly_chart(fig,use_container_width=True)

# # ==========================================
# # DATA QUALITY ANALYSIS
# # ==========================================

# with st.expander("🧹 Data Cleaning & Quality Analysis"):

#     null_values = processed_df.isnull().sum()

#     fig_null = px.bar(
#         x=null_values.index,
#         y=null_values.values,
#         labels={"x":"Features","y":"Missing Values"},
#         title="Missing Values per Feature"
#     )

#     st.plotly_chart(fig_null,use_container_width=True)

#     duplicate_count = processed_df.duplicated().sum()

#     st.metric("Duplicate Rows Found",duplicate_count)

# ==========================================
# EXPLORATORY DATA ANALYSIS
# ==========================================

with st.expander("📊 Data Analysis"):

    fig1 = px.scatter(
        processed_df,
        x="Projects",
        y="GPA",
        color="Career_Group",
        title="Projects vs GPA by Career"
    )

    st.plotly_chart(fig1,use_container_width=True)

    fig2 = px.box(
        processed_df,
        x="Career_Group",
        y="Python",
        title="Python Skill Distribution Across Careers"
    )

    st.plotly_chart(fig2,use_container_width=True)

    fig3 = px.histogram(
        processed_df,
        x="GPA",
        color="Career_Group",
        title="GPA Distribution by Career"
    )

    st.plotly_chart(fig3,use_container_width=True)

# ==========================================
# CAREER DISTRIBUTION (REPLACED FEATURE CONTRIBUTION)
# ==========================================

with st.expander("🎯 Career Distribution"):

    career_counts = processed_df["Career_Group"].value_counts()

    fig = px.pie(
        values=career_counts.values,
        names=career_counts.index
    )

    st.plotly_chart(fig, use_container_width=True)

# ==============================
# CLUSTER INSIGHT (DROPDOWN)
# ==============================

# with st.expander("🔍 Cluster Insight"):

#     cluster = kmeans.predict(input_scaled)[0]

#     cluster_info = cluster_summary.loc[cluster]

#     st.write("This cluster mainly contains:")

#     st.dataframe(cluster_info)

# ==============================
# SIMILAR STUDENTS (DROPDOWN)
# ==============================

with st.expander("👨‍🎓 Similar Students"):

    distances, indices = knn.kneighbors(input_scaled)

    similar_students = processed_df.iloc[indices[0]]

    st.dataframe(
        similar_students[
            ["GPA","Python","SQL","Java","Projects","Career_Group"]
        ]
    )



