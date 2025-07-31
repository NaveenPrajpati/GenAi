import os, sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

# Determine absolute path to the PDF
script_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(script_dir, "resume.pdf")
# Verify existence


try:
    # Load all pages from the PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from resume.")
    # Combine all page contents into one string
    resume_text = "\n".join(doc.page_content for doc in docs)
except Exception as e:
    print("error in loading")
    sys.exit(1)

# text_splitter = SemanticChunker(embedder=OpenAIEmbeddings())
# chunks = text_splitter.split_documents(docs)

prompt = PromptTemplate.from_template(" Extract detail from resume. {data}")


infoSchema = {
    "title": "BasicInfoOutput",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "contact": {"type": "string"},
        "location": {"type": "string"},
        "education_history": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string"},
                    "field_of_study": {"type": "string"},
                    "institution": {"type": "string"},
                    "start_year": {"type": "integer"},
                    "end_year": {"type": "integer"},
                },
                "required": ["degree", "institution"],
            },
        },
        "work_experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "job_title": {"type": "string"},
                    "company": {"type": "string"},
                    "location": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "description": {"type": "string"},
                    "achievements": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["job_title", "company"],
            },
        },
        "skills": {"type": "array", "items": {"type": "string"}},
        "certifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "issuer": {"type": "string"},
                    "date_obtained": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    "required": ["name", "contact", "education_history", "work_experience", "skills"],
}


llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm.with_structured_output(infoSchema)
basic_info = chain.invoke({"data": resume_text})
formatPrompt = PromptTemplate.from_template(
    """
You are a resume formatting expert.
Review the following resume and evaluate its layout and visual structure:
{resume_text}

Return a JSON with:
- overall_format_score (0 to 10)
- readability_issues: list of detected problems (e.g. font inconsistency, poor spacing)
- layout_quality: comment on visual flow (e.g. reverse chronological, proper section headings)
- suggestions: actionable formatting improvements
"""
)

qualityPrompt = PromptTemplate.from_template(
    """
You are a professional resume reviewer.
Analyze the achievement descriptions and writing style of the following resume:
{resume_text}

Return a JSON with:
- achievement_quality_score (0 to 10)
- weak_phrases: list of vague or passive descriptions
- strong_action_verbs: examples used in resume
- quantified_results: number of measurable outcomes mentioned
- recommendations: how to improve content strength
"""
)

atsPrompt = PromptTemplate.from_template(
    """
You are a professional resume reviewer.
Analyze the achievement descriptions and writing style of the following resume:
{resume_text}

Return a JSON with:
- achievement_quality_score (0 to 10)
- weak_phrases: list of vague or passive descriptions
- strong_action_verbs: examples used in resume
- quantified_results: number of measurable outcomes mentioned
- recommendations: how to improve content strength
"""
)

industryPrompt = PromptTemplate.from_template(
    """
You are an expert in resume evaluation for industry alignment.
Compare the resume to the following job description and assess fit:
Resume:
{resume_text}

Job Description:
{job_description}

Return a JSON with:
- match_score (0 to 100)
- matched_skills: list of skills that align
- missing_qualifications: what’s lacking for the role
- industry_specific_terms: detected terminology relevant to the field
- alignment_feedback: what changes would better tailor the resume
"""
)

chain1 = atsPrompt | llm
ats_feedback = chain1.invoke({"resume_text": resume_text})
print(ats_feedback.content)
