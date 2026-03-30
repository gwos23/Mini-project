import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from docx import Document
import json
import os


# -------------------- NLTK --------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")


# -------------------- USER STORAGE --------------------
def load_users():
    if not os.path.exists("users.json"):
        return {}
    with open("users.json", "r") as f:
        return json.load(f)

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)


# -------------------- SESSION --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Login"


# -------------------- LOGIN PAGE --------------------
def login_page():
    st.title("Sign in")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign in"):
        users = load_users()

        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.write("Don't have an account?")
    if st.button("Go to Sign Up"):
        st.session_state.page = "Sign Up"
        st.rerun()


# -------------------- SIGNUP PAGE --------------------
def signup_page():
    st.title("Create Account")

    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")

    if st.button("Create Account"):
        users = load_users()

        if new_user in users:
            st.error("User already exists")
        else:
            users[new_user] = new_pass
            save_users(users)
            st.success("Account created successfully!")
            st.session_state.page = "Login"
            st.rerun()

    if st.button("Back to Login"):
        st.session_state.page = "Login"
        st.rerun()


# -------------------- PROTECT APP --------------------
if not st.session_state.logged_in:
    if st.session_state.page == "Login":
        login_page()
    else:
        signup_page()
    st.stop()


# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Smart Resume Analyzer",page_icon="📄",layout="wide")
st.title("Smart Resume Analyzer")

st.markdown("""
Upload your resume (PDF) and paste a job description to see how well they match!  
This tool uses **TF-IDF + Cosine Similarity** to analyze your resume against job requirements.
""")

with st.sidebar:
    st.header("About")
    st.info("""
    This tool helps you:
    - Measures how your resume matches a job description
    - Identify important job keywords
    - Improve your reseume based on missing terms
    """)
    st.header("How It works")
    st.write("""
    1. Upload your resume (PDF)
    2. Paste the job description
    3. Click **Analyze Match**
    4. Review score & suggetion
    """)


# -------------------- SIDEBAR --------------------
with st.sidebar:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "Login"
        st.rerun()


# -------------------- FUNCTIONS --------------------
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return " ".join([w for w in words if w not in stop_words])


def calculate_similarity(resume_text, job_text):
    r = remove_stopwords(clean_text(resume_text))
    j = remove_stopwords(clean_text(job_text))

    vec = TfidfVectorizer()
    tfidf = vec.fit_transform([r, j])

    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
    return round(score, 2), r, j


def extract_keywords(text, n=15):
    words = word_tokenize(text)
    tagged = pos_tag(words)

    keywords = [w for w, p in tagged if p.startswith('NN') or p.startswith('JJ')]
    freq = Counter(keywords)

    return [w for w, _ in freq.most_common(n)]


def find_missing_keywords(resume, job):
    return list(set(extract_keywords(job)) - set(extract_keywords(resume)))[:10]


def generate_suggestions(missing):
    suggestions = []

    for w in missing:
        suggestions.append(f"Add or emphasize '{w}' in your resume.")

    suggestions.extend([
        "Use strong action verbs (developed, built, optimized).",
        "Add measurable achievements (e.g., increased performance by 30%).",
        "Include relevant technical and soft skills.",
        "Tailor your resume title to match the job role.",
        "Use bullet points for clarity."
    ])

    return suggestions


def ats_score(similarity, missing_keywords):
    penalty = len(missing_keywords) * 2
    return max(0, round(similarity - penalty, 2))


def highlight_keywords(text, keywords):
    for word in keywords:
        text = re.sub(f"\\b{word}\\b", f"**{word.upper()}**", text, flags=re.IGNORECASE)
    return text


def improve_resume_free(resume_text, missing_keywords):
    improved = []

    improved.append("IMPROVED RESUME SUGGESTIONS:\n")

    if missing_keywords:
        improved.append("Add these keywords:\n")
        improved.append(", ".join(missing_keywords) + "\n")

    improved.append("Improve your summary:\n")
    improved.append("Write a strong professional summary highlighting your skills and experience.\n")

    improved.append("Improve bullet points:\n")
    improved.append("- Use action verbs\n")
    improved.append("- Add measurable results\n")
    improved.append("- Focus on achievements\n")

    improved.append("Add sections if missing:\n")
    improved.append("- Skills\n- Projects\n- Certifications\n")

    return "\n".join(improved)


def export_doc(text):
    doc = Document()
    doc.add_paragraph(text)
    file_path = "improved_resume.docx"
    doc.save(file_path)
    return file_path


# -------------------- MAIN APP --------------------
uploaded = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job = st.text_area("Paste Job Description", height=200)

if st.button("Analyze"):

    if not uploaded or not job:
        st.warning("Upload resume and job description")

    else:
        resume_text = extract_text_from_pdf(uploaded)

        score, r, j = calculate_similarity(resume_text, job)
        missing = find_missing_keywords(r, j)
        suggestions = generate_suggestions(missing)
        ats = ats_score(score, missing)

        st.subheader("Results")
        st.metric("Similarity", f"{score}%")
        st.metric("ATS Score", f"{ats}%")

        # Chart
        fig, ax = plt.subplots()
        ax.barh(["Score"], [score])
        ax.set_xlim(0, 100)
        st.pyplot(fig)

        if score < 70:
            st.subheader("Missing Keywords")
            st.write(", ".join(missing))

            st.subheader("Suggestions")
            for s in suggestions:
                st.write("- " + s)

        st.subheader("Resume Highlighted")
        st.write(highlight_keywords(resume_text, missing))

        if score < 60:
            st.subheader("Resume Improvement")

            improved = improve_resume_free(resume_text, missing)
            st.write(improved)

            file = export_doc(improved)
            with open(file, "rb") as f:
                st.download_button(
                    "Download Improved Resume",
                    f,
                    file_name="improved_resume.docx"
                )