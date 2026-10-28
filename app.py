import io
import re
import pandas as pd
import streamlit as st
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ---------------------------------------------------------
# 1. MODEL TRAINING & INITIALIZATION
# ---------------------------------------------------------
@st.cache_resource
def train_model():
    """
    Trains a lightweight TF-IDF + PassiveAggressiveClassifier model 
    on a sample baseline dataset.
    """
    training_data = {
        'text': [
            # Real News Examples
            "Government announces new tax incentives for clean energy businesses starting next fiscal year.",
            "Central bank leaves benchmark interest rates unchanged following the monthly policy meeting.",
            "Health authority issues updated dietary guidelines recommending higher fiber intake.",
            "Local city council approves funding for public transportation infrastructure improvements.",
            "Astronomers observe a distant supernova using space-based telescopes.",
            "Scientists report progress in renewable battery efficiency during laboratory trials.",
            "The national ministry of education releases the annual report on high school graduation rates.",
            "Official statistics indicate a modest decline in consumer price inflation over the last quarter.",
            
            # Fake News Examples
            "SHOCKING: Secret potion guarantees 200 years of youth, doctors don't want you to know!",
            "BREAKING: Alien spacecraft landed in downtown city center last night, military hiding evidence!",
            "Miracle household item cures all known diseases overnight without medication!",
            "You won't believe what this celebrity did, click here to see the secret truth now!",
            "Government secretly putting microchips in tap water to control minds, whistleblowers reveal!",
            "Scientists confirm the Earth is hollow and inhabited by giant glowing creatures!",
            "Instant cash system makes everyone $10,000 a day from home with zero effort required!",
            "Unbelievable truth revealed: Ancient pyramids were actually prehistoric power plants!"
        ],
        'label': [
            'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL',
            'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE'
        ]
    }
    
    df = pd.DataFrame(training_data)
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
    X_train = vectorizer.fit_transform(df['clean_text'])
    
    model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    model.fit(X_train, df['label'])
    
    return vectorizer, model


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------
# 2. DOCUMENT PARSERS (PDF, DOCX, EXCEL)
# ---------------------------------------------------------
def extract_text_from_pdf(file_bytes):
    text = ""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text


def extract_text_from_excel(file_bytes, filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))
    
    text_list = []
    for row in df.astype(str).values:
        row_text = " ".join([cell for cell in row if cell.lower() != 'nan' and cell.strip()])
        if row_text:
            text_list.append(row_text)
            
    return text_list, df


# ---------------------------------------------------------
# 3. STREAMLIT UI LAYOUT
# ---------------------------------------------------------
def main():
    st.title("📰 Fake News Detection System")
    st.write("Upload a document (**PDF**, **Word**, **Excel**, or **CSV**) or enter custom text to analyze authenticity.")
    
    vectorizer, model = train_model()
    st.divider()
    
    tab1, tab2 = st.tabs(["📁 File Upload (PDF / Word / Excel)", "✏️ Direct Text Input"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=["pdf", "docx", "xlsx", "xls", "csv"],
            help="Supported formats: PDF, DOCX, XLSX, XLS, CSV"
        )
        
        if uploaded_file is not None:
            file_name = uploaded_file.name
            file_bytes = uploaded_file.read()
            st.info(f"File uploaded successfully: **{file_name}**")
            
            extracted_text = ""
            excel_rows = None
            raw_df = None
            
            try:
                if file_name.endswith('.pdf'):
                    extracted_text = extract_text_from_pdf(file_bytes)
                elif file_name.endswith('.docx'):
                    extracted_text = extract_text_from_docx(file_bytes)
                elif file_name.endswith(('.xlsx', '.xls', '.csv')):
                    excel_rows, raw_df = extract_text_from_excel(file_bytes, file_name)
            except Exception as e:
                st.error(f"Error parsing file: {e}")
                return
            
            if excel_rows is not None:
                st.subheader("Data Analysis Preview")
                st.dataframe(raw_df.head(5))
                
                if st.button("Analyze Excel / CSV Content", type="primary"):
                    cleaned_rows = [preprocess_text(r) for r in excel_rows]
                    vectorized_rows = vectorizer.transform(cleaned_rows)
                    predictions = model.predict(vectorized_rows)
                    
                    results_df = pd.DataFrame({
                        'Extracted Content': excel_rows,
                        'Prediction': predictions
                    })
                    
                    st.subheader("Classification Results")
                    
                    def highlight_prediction(val):
                        color = '#ff4b4b' if val == 'FAKE' else '#21c354'
                        return f'background-color: {color}; color: white; font-weight: bold;'
                    
                    st.dataframe(results_df.style.map(highlight_prediction, subset=['Prediction']))
                    
                    fake_count = (predictions == 'FAKE').sum()
                    real_count = (predictions == 'REAL').sum()
                    st.metric("Total Records Analyzed", len(predictions))
                    st.metric("Detected Fake News Items", int(fake_count))
                    st.metric("Detected Real News Items", int(real_count))
            else:
                if not extracted_text.strip():
                    st.warning("No readable text could be extracted from this document.")
                else:
                    st.subheader("Extracted Document Text")
                    st.text_area("Content Preview", extracted_text, height=200)
                    
                    if st.button("Analyze Document Authenticity", type="primary"):
                        cleaned = preprocess_text(extracted_text)
                        vectorized = vectorizer.transform([cleaned])
                        prediction = model.predict(vectorized)[0]
                        
                        st.subheader("Analysis Result")
                        if prediction == 'FAKE':
                            st.error("🚨 **PREDICTION: FAKE NEWS**")
                        else:
                            st.success("✅ **PREDICTION: REAL NEWS**")

    with tab2:
        user_input = st.text_area("Paste news text or headline here:", height=150)
        
        if st.button("Analyze Text", type="primary"):
            if not user_input.strip():
                st.warning("Please enter some text to analyze.")
            else:
                cleaned = preprocess_text(user_input)
                vectorized = vectorizer.transform([cleaned])
                prediction = model.predict(vectorized)[0]
                
                if prediction == 'FAKE':
                    st.error("🚨 **PREDICTION: FAKE NEWS**")
                else:
                    st.success("✅ **PREDICTION: REAL NEWS**")


if __name__ == "__main__":
    main()