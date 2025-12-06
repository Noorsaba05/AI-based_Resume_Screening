import streamlit as st
import pandas as pd
import utils

def main():
    st.set_page_config(page_title="AI Resume Screen", layout="wide")
    
    st.title("📄 AI-Based Resume Screening System")
    st.markdown("### Rank resumes based on job description relevance using TF-IDF & Cosine Similarity")

    # 1. Sidebar for Inputs
    with st.sidebar:
        st.header("1. Job Description")
        job_description = st.text_area("Paste the Job Description (JD) here:", height=300)
        
        st.header("2. Upload Resumes")
        uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)
        
        process_btn = st.button("Analyze & Rank")

    # 2. Main Processing Area
    if process_btn:
        if not job_description:
            st.error("⚠️ Please provide a Job Description.")
        elif not uploaded_files:
            st.error("⚠️ Please upload at least one resume.")
        else:
            with st.spinner("Processing resumes..."):
                # A. Parse and Clean Resumes
                resume_data = []  # <--- FIXED: Added brackets here
                for file in uploaded_files:
                    # Extract raw text
                    raw_text = utils.extract_text_from_pdf(file)
                    
                    # Clean text (remove emails, phones, special chars)
                    cleaned_text = utils.clean_text(raw_text)
                    
                    # Extract basic entities (Optional display)
                    skills_preview = utils.extract_skills(cleaned_text)
                    
                    resume_data.append({
                        "filename": file.name,
                        "raw_text": raw_text,
                        "cleaned_text": cleaned_text,
                        "skills_preview": skills_preview
                    })
                
                df = pd.DataFrame(resume_data)

                # B. Calculate Ranking
                # We pass the JD and the list of cleaned resume texts
                scores = utils.calculate_similarity(job_description, df['cleaned_text'].tolist())
                
                # --- FIX IS HERE ---
                # Add scores to dataframe as a NEW COLUMN (do not overwrite df)
                df['Match Score'] = scores
                
                # Convert to percentage for display
                df['Match %'] = (df['Match Score'] * 100).round(2)
                # -------------------

                # C. Display Results
                # Sort by score descending
                df_ranked = df.sort_values(by='Match %', ascending=False).reset_index(drop=True)
                
                # Add a rank column
                df_ranked.index += 1
                df_ranked.index.name = 'Rank'

                st.success(f"✅ Analysis Complete! Processed {len(df_ranked)} resumes.")
                
                # Display Summary Table
                st.subheader("🏆 Ranked Candidates")
                
                # columns to display
                display_cols = ['filename', 'Match %', 'skills_preview']
                
                # Style the dataframe (highlight high scores)
                st.dataframe(df_ranked[display_cols].style.background_gradient(subset=['Match %'], cmap="Greens"))

                # D. Visualization
                if not df_ranked.empty:
                    st.subheader("📊 Relevance Distribution")
                    st.bar_chart(df_ranked.set_index('filename')['Match %'])

if __name__ == "__main__":
    main()