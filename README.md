# Two Months ML Journey: Building Confidence for a Career in Machine Learning
# [Day: 55]

## Welcome to My Journey
This repository chronicles my **2-month project-based learning journey** to build the skills and confidence needed to become a Junior Machine Learning (ML) Engineer.  

As a third-year student with foundational ML knowledge (e.g., familiar with the Iris dataset), I’m not aiming to master ML in just two months, that’s a lifelong pursuit! Instead, this journey is about proving 
to myself and future employers, especially at **Nepali AI/ML companies** like *Fusemachines, NAAMII, Cedar Gate, Paaila Technology, and more*, that I have the drive, skills, and potential to contribute to 
the industry.

This README is a living document, updated regularly as I progress through prerequisite revision, complete projects, and achieve milestones. Each project emphasizes **Exploratory Data Analysis (EDA)** to 
uncover data insights, followed by model building, evaluation, and deployment, mirroring real-world ML workflows.  

My journey begins with refreshing core skills to ensure a strong foundation for the exciting projects ahead.

---

## Why This Journey?

- **Purpose**: To develop a portfolio of 4-5 industry-relevant ML projects, mastering skills like EDA, preprocessing, modeling, and deployment, while building confidence as an ML practitioner.  
- **Nepal Focus**: Projects address local challenges, such as agriculture (crop disease detection) and customer support (chatbot), aligning with Nepal’s AI/ML industry needs.  
- **Mindset**: This is a marathon, not a sprint. Every dataset explored, plot created, and model trained fuels my belief that I can become an ML Engineer.

---

  ## Current Focus: Prerequisite Revision [**Status**: 🟢 Completed]
I’m starting by revisiting foundational skills to ensure I’m ready for project-based learning. This phase (2-3 days) focuses on strengthening my Python, data handling, and ML basics, preparing me for EDA-heavy projects like **Titanic Survival Prediction** and **Crop Disease Detection**.
### Skills Being Revised
- **Python**: Variables, loops, functions, list comprehensions, error handling.  
- **Data Handling**: Pandas (filtering, grouping, handling missing values), NumPy (array operations).  
- **Visualization**: Matplotlib and seaborn for scatter plots, histograms, and heatmaps.  
- **ML Basics**: Loading datasets (e.g., Iris), basic algorithms (logistic regression, decision trees), and evaluation metrics (accuracy, MSE).  
- **Tools Setup**: Jupyter Notebook, Google Colab, scikit-learn, TensorFlow.  

### Practice Tasks
- Load and explore the Iris dataset (scikit-learn).  
- Create visualizations (e.g., scatter plot of sepal vs. petal length, histogram of features).  
- Build a simple classifier (e.g., logistic regression) on Iris and evaluate accuracy.  
- Document findings in a notebook to kickstart my portfolio.  

### Resources
- Kaggle, YouTube, Google, Scikit-Learn.  

---

<!-- Planned Projects -->
<h2>Planned Projects</h2>

<p>
  Below are the projects I am working on or have completed. Each one builds core ML skills while also moving toward an
  <strong>AI engineering</strong> mindset with clean pipelines, modular code, and deployment readiness.
</p>

<h3>1. Titanic Survival Prediction (Classification)</h3>
<p><strong>Status:</strong> 🟢 Completed<br>
<strong>Folder:</strong> <code>P1-Titanic_Project/</code><br>
<strong>Dataset:</strong> Titanic (Kaggle, ~891 samples)<br>
<strong>Objective:</strong> Predict passenger survival based on features like age, gender, and class.</p>

<h4>What I worked on</h4>
<ul>
  <li>Performed structured EDA on survival rates and feature distributions.</li>
  <li>Handled missing values and encoded categorical features.</li>
  <li>Built preprocessing and modeling pipelines using scikit-learn.</li>
  <li>Trained models such as Logistic Regression, Decision Trees, and Random Forests.</li>
</ul>

<h4>AI/ML angle</h4>
<ul>
  <li>Introduced reproducible ML pipelines instead of one-off scripts.</li>
  <li>Prepared the project so it can later be wrapped into an API or simple web app.</li>
</ul>

<hr>

<h3>2. Boston Housing Price Prediction (Regression)</h3>
<p><strong>Status:</strong> 🟢 Completed<br>
<strong>Folder:</strong> <code>P2-Boston_House_Pred/</code><br>
<strong>Dataset:</strong> Boston Housing (scikit-learn, 506 samples)<br>
<strong>Objective:</strong> Predict house prices based on features like number of rooms, crime rate, and other neighborhood characteristics.</p>

<h4>What I worked on</h4>
<ul>
  <li>Conducted EDA using scatter plots, histograms, and correlation heatmaps.</li>
  <li>Implemented custom logic (e.g., Mahalanobis distance in <code>mahalanobis.py</code>) to explore outliers and feature space.</li>
  <li>Built modular pipelines (e.g., exported as <code>bhp_pipeline.joblib</code>) for preprocessing and regression models.</li>
  <li>Evaluated models such as Linear Regression and Random Forest Regressor using metrics like MSE and R².</li>
</ul>

<h4>AI/ML angle</h4>
<ul>
  <li>Focused on clean, reusable code via custom transformers and saved pipelines.</li>
  <li>Took a step toward production readiness by exporting trained pipelines using <code>joblib</code>.</li>
</ul>

<hr>

<h3>3. Wine Quality Prediction (Multiclass Classification)</h3>
<p><strong>Status:</strong> 🟢 Completed<br>
<strong>Folder:</strong> <code>P3-Wine_Quality_Pred/</code><br>
<strong>Dataset:</strong> Wine Quality (UCI/Kaggle, ~4,800 samples)<br>
<strong>Objective:</strong> Predict wine quality scores based on chemical properties (e.g., pH, alcohol content, acidity).</p>

<h4>What I worked on</h4>
<ul>
  <li>Performed EDA with box plots, distribution analysis, and class balance checks.</li>
  <li>Built multiple models: k-Nearest Neighbors, SVM, and a simple Neural Network in Keras.</li>
  <li>Compared model performance using accuracy and F1-score.</li>
  <li>Structured the notebooks (<code>wqp_eda.ipynb</code>, <code>wqp_model.ipynb</code>, <code>wqp_model_nn.ipynb</code>) to separate EDA and modeling logic.</li>
</ul>

<h4>AI/ML angle</h4>
<ul>
  <li>Practiced experiment-style workflow by training several models on the same problem.</li>
  <li>Continued the habit of saving and organizing pipelines for future reuse.</li>
</ul>

<hr>

<h3>4. Crop Disease Detection (Computer Vision, End-to-End AI Project)</h3>
<p><strong>Status:</strong> 🟢 Completed<br>
<strong>Folder:</strong> <code>P4-Crop_Disease_Pred/</code><br>
<strong>Dataset:</strong> PlantVillage (Kaggle, ~50,000 images)<br>
<strong>Objective:</strong> Classify crop images (e.g., rice, maize) as healthy or diseased to support farmers and agriculture.</p>

<h4>What I worked on</h4>
<ul>
  <li>Performed dataset exploration: visualized sample images, inspected class distribution, and identified imbalance.</li>
  <li>Used Convolutional Neural Networks and Transfer Learning (e.g., MobileNet-based architecture).</li>
  <li>Applied data augmentation to improve generalization.</li>
  <li>Exported trained models and weights for inference.</li>
  <li>In a separate repository, built a <strong>FastAPI backend</strong> to serve predictions and a <strong>React frontend</strong> for users to upload images.</li>
  <li>Containerized the system using Docker and deployed it on the cloud.</li>
</ul>

<h4>AI engineering angle</h4>
<ul>
  <li>Goes beyond a notebook and becomes a real application stack: model + API + UI + Docker.</li>
  <li>Designed with real users in mind (farmers or agriculture workers), not just offline experiments.</li>
</ul>

<p><strong>Know more about this project:</strong><br>
<a href="https://github.com/UjjwalKarkeyy/Crop_Disease_Prediction_Deploy">Add external repository link here</a>
</p>

<hr>

<h3>5. BainiAI – Customer Support Chatbot (RAG + LLM Tools)</h3>
<p><strong>Status:</strong> 🟢 Completed<br>
<strong>Folder:</strong> <code>P5-ChatBot/</code><br>
<strong>Dataset / Knowledge Source:</strong> Internal documents and text data loaded into a Retrieval-Augmented Generation (RAG) pipeline.<br>
<strong>Objective:</strong> Build an AI assistant that can answer user queries from documents and also book appointments through a conversational form.</p>

<h4>What I worked on</h4>
<ul>
  <li>Implemented a RAG pipeline using <strong>LangChain</strong> and <strong>Gemini</strong>.</li>
  <li>Loaded and chunked documents, created embeddings, and stored them in <strong>Chroma</strong> as a vector database.</li>
  <li>Used <code>MultiQueryRetriever</code> to improve recall and handle diverse user questions.</li>
  <li>Added conversational memory using LangChain’s message history for multi-turn chats.</li>
  <li>Designed a conversational appointment flow that:
    <ul>
      <li>Asks for and validates name, email, and phone number.</li>
      <li>Understands natural language dates like “next Monday” or “coming Thursday” and converts them to proper date formats.</li>
      <li>Stores booked appointments into an Excel sheet for non-technical staff to review easily.</li>
    </ul>
  </li>
  <li>Built a <strong>FastAPI backend</strong> and <strong>React (Vite) frontend</strong> for the chatbot interface.</li>
  <li>Dockerized the backend and frontend for consistent deployment.</li>
</ul>

<h4>AI engineering angle</h4>
<ul>
  <li>Combines LLMs, tools, memory, and external storage (Excel, vector DB) into one system.</li>
  <li>Moves from “just a model” to a real AI product with an API, UI, and integrations.</li>
</ul>

<p><strong>Know more about this project:</strong><br>
<a href="https://github.com/UjjwalKarkeyy/bainiAI_Chatbot_Full">Add external repository link here</a>
</p>

<hr>
---

## Skills Progress

### Current Skills
- Python: Basics (variables, loops, functions, class and objects).  
- Data Handling: Pandas, NumPy.  
- Visualization: Matplotlib/seaborn.  
- ML Basics: Familiar with Iris dataset, logistic regression.  

### Skills After this Journey
- Advanced EDA (correlation analysis, handling imbalanced data).  
- Neural Networks (CNNs).  
- Deployment (Streamlit, Flask).  

---

<!-- Milestones -->
<h2>Milestones</h2>

<p>
  Instead of a strict week-by-week plan, these milestones reflect how the journey has actually progressed in phases.
</p>

<h3>Phase 0: Prerequisite Revision</h3>
<ul>
  <li>Revised Python fundamentals, NumPy, pandas, visualization, and ML basics.</li>
  <li>Practiced with the Iris dataset and other small exercises.</li>
</ul>

<h3>Phase 1: Classic ML Projects</h3>
<ul>
  <li>Completed:
    <ul>
      <li>Titanic Survival Prediction (classification).</li>
      <li>Boston Housing Price Prediction (regression).</li>
      <li>Wine Quality Prediction (multiclass classification).</li>
    </ul>
  </li>
  <li>Focused on EDA, preprocessing, modeling, and building reproducible pipelines.</li>
</ul>

<h3>Phase 2: Deep Learning and End-to-End System</h3>
<ul>
  <li>Completed the Crop Disease Detection project using CNNs and transfer learning.</li>
  <li>Extended the work into a FastAPI + React + Docker stack hosted on the cloud.</li>
  <li>First full taste of an “AI system,” not just a notebook.</li>
</ul>

<h3>Phase 3: LLMs, RAG, and AI Applications (Ongoing)</h3>
<ul>
  <li>Currently developing BainiAI, a LangChain + Gemini based chatbot with:
    <ul>
      <li>A RAG pipeline over documents.</li>
      <li>Conversational memory.</li>
      <li>Appointment booking via conversational forms.</li>
      <li>Excel-based storage for appointments.</li>
      <li>FastAPI backend and React frontend, both containerized.</li>
    </ul>
  </li>
</ul>

<h3>Phase 4: Portfolio, Refinement, and Applications (Planned)</h3>
<ul>
  <li>Polish documentation and READMEs across all projects.</li>
  <li>Improve deployments and explore basic CI/CD.</li>
  <li>Create a clean portfolio website to showcase these projects.</li>
  <li>Apply to internships and roles at companies like Fusemachines, NAAMII, Cedar Gate, Paaila Technology, and others.</li>
</ul>

---

## Future Plans
- Add more projects (e.g., hydropower forecasting for Nepal’s energy sector).  
- Explore Nepal-specific datasets (e.g., from Nepal Agricultural Research Council).  
- Network with Nepal’s AI community.  
- Contribute to open-source ML projects.  

---

## About Me
I’m a third-year student passionate about AI/ML, driven to solve real-world problems in Nepal through data-driven solutions. This journey is my stepping stone to a career in ML, fueled by 
curiosity and determination.  

👉 Connect with me on LinkedIn at **https://www.linkedin.com/in/ujjwal-karki-871b592a9/** or check my progress on GitHub.  

---

## How to Use This Repository
- Explore project notebooks for detailed EDA, code, and results.  
- Check READMEs in each project folder for insights and Nepal-specific applications.  
- Feel free to fork or suggest improvements, I’m always learning!  

---

**Last Updated**: November 18, 2025
