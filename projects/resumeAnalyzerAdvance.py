from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import os
import json
from typing import Dict, Any, List


class ResumeAnalyzer:
    def __init__(self, openai_api_key: str = None):
        """Initialize the Resume Analyzer with necessary components."""
        load_dotenv()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0.1,  # Lower temperature for more consistent analysis
        )
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = SemanticChunker(self.embeddings)

    def load_document(self, file_path: str) -> str:
        """Load and extract text from PDF or text files."""
        try:
            if file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                # Combine all pages
                text = "\n".join([page.page_content for page in pages])
            else:
                loader = TextLoader(file_path)
                docs = loader.load()
                text = docs[0].page_content
            return text
        except Exception as e:
            raise Exception(f"Error loading document: {str(e)}")

    def extract_basic_info(self, resume_text: str) -> Dict[str, Any]:
        """Extract structured information from resume."""
        info_schema = {
            "title": "BasicInfoOutput",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "contact": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                        "phone": {"type": "string"},
                        "linkedin": {"type": "string"},
                    },
                },
                "location": {"type": "string"},
                "professional_summary": {"type": "string"},
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
                            "gpa": {"type": "string"},
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
                            "achievements": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "technologies_used": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["job_title", "company"],
                    },
                },
                "skills": {
                    "type": "object",
                    "properties": {
                        "technical_skills": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "soft_skills": {"type": "array", "items": {"type": "string"}},
                        "programming_languages": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "frameworks_tools": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "certifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "issuer": {"type": "string"},
                            "date_obtained": {"type": "string"},
                            "credential_id": {"type": "string"},
                        },
                        "required": ["name"],
                    },
                },
            },
            "required": [
                "name",
                "contact",
                "education_history",
                "work_experience",
                "skills",
            ],
        }

        extraction_prompt = PromptTemplate.from_template(
            """
            Extract detailed information from the following resume text.
            Be thorough and accurate in extracting all relevant details.
            If information is not available, use null or empty arrays as appropriate.
            
            Resume Text:
            {resume_text}
            """
        )

        chain = extraction_prompt | self.llm.with_structured_output(info_schema)
        return chain.invoke({"resume_text": resume_text})

    def analyze_formatting(self, resume_text: str) -> Dict[str, Any]:
        """Analyze resume formatting and layout."""
        format_schema = {
            "type": "object",
            "properties": {
                "overall_format_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                },
                "readability_issues": {"type": "array", "items": {"type": "string"}},
                "layout_quality": {"type": "string"},
                "section_organization": {"type": "array", "items": {"type": "string"}},
                "suggestions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["overall_format_score", "readability_issues", "suggestions"],
        }

        format_prompt = PromptTemplate.from_template(
            """
            You are a professional resume formatting expert.
            Analyze the structure, organization, and readability of this resume:
            
            {resume_text}
            
            Evaluate:
            1. Section organization and flow
            2. Consistency in formatting
            3. Use of white space and readability
            4. Professional appearance
            5. ATS-friendly structure
            
            Provide specific, actionable formatting recommendations.
            """
        )

        chain = format_prompt | self.llm.with_structured_output(format_schema)
        return chain.invoke({"resume_text": resume_text})

    def analyze_content_quality(self, resume_text: str) -> Dict[str, Any]:
        """Analyze content quality and achievement descriptions."""
        quality_schema = {
            "type": "object",
            "properties": {
                "achievement_quality_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                },
                "weak_phrases": {"type": "array", "items": {"type": "string"}},
                "strong_action_verbs": {"type": "array", "items": {"type": "string"}},
                "quantified_results": {"type": "integer"},
                "improvement_suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "content_strengths": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "achievement_quality_score",
                "weak_phrases",
                "improvement_suggestions",
            ],
        }

        quality_prompt = PromptTemplate.from_template(
            """
            You are a professional resume content reviewer.
            Analyze the quality of achievements, descriptions, and overall content:
            
            {resume_text}
            
            Focus on:
            1. Use of strong action verbs vs. weak/passive language
            2. Quantified achievements and measurable results
            3. Relevance and impact of described experiences
            4. Professional language and clarity
            5. Demonstration of value and accomplishments
            
            Provide specific examples and actionable improvements.
            """
        )

        chain = quality_prompt | self.llm.with_structured_output(quality_schema)
        return chain.invoke({"resume_text": resume_text})

    def analyze_ats_compatibility(self, resume_text: str) -> Dict[str, Any]:
        """Analyze ATS (Applicant Tracking System) compatibility."""
        ats_schema = {
            "type": "object",
            "properties": {
                "ats_score": {"type": "integer", "minimum": 0, "maximum": 10},
                "keyword_density": {"type": "string"},
                "formatting_issues": {"type": "array", "items": {"type": "string"}},
                "recommended_keywords": {"type": "array", "items": {"type": "string"}},
                "ats_improvements": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["ats_score", "formatting_issues", "ats_improvements"],
        }

        ats_prompt = PromptTemplate.from_template(
            """
            You are an ATS (Applicant Tracking System) compatibility expert.
            Analyze how well this resume would perform with automated screening systems:
            
            {resume_text}
            
            Evaluate:
            1. Keyword optimization and density
            2. Standard section headings
            3. Simple formatting without complex elements
            4. File format compatibility
            5. Scannable structure
            
            Provide specific recommendations for ATS optimization.
            """
        )

        chain = ats_prompt | self.llm.with_structured_output(ats_schema)
        return chain.invoke({"resume_text": resume_text})

    def analyze_job_match(
        self, resume_text: str, job_description: str
    ) -> Dict[str, Any]:
        """Analyze how well the resume matches a specific job description."""
        if not job_description:
            return {"error": "Job description is required for matching analysis"}

        match_schema = {
            "type": "object",
            "properties": {
                "match_score": {"type": "integer", "minimum": 0, "maximum": 100},
                "matched_skills": {"type": "array", "items": {"type": "string"}},
                "missing_qualifications": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "experience_alignment": {"type": "string"},
                "suggested_improvements": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "keyword_gaps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["match_score", "matched_skills", "missing_qualifications"],
        }

        match_prompt = PromptTemplate.from_template(
            """
            You are an expert in resume-job matching analysis.
            Compare this resume against the job description and provide detailed matching analysis:
            
            Resume:
            {resume_text}
            
            Job Description:
            {job_description}
            
            Analyze:
            1. Skill alignment and gaps
            2. Experience relevance
            3. Required vs. preferred qualifications match
            4. Industry-specific terminology usage
            5. Potential areas for resume customization
            
            Provide actionable recommendations for improving the match.
            """
        )

        chain = match_prompt | self.llm.with_structured_output(match_schema)
        return chain.invoke(
            {"resume_text": resume_text, "job_description": job_description}
        )

    def generate_comprehensive_analysis(
        self, file_path: str, job_description: str = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive analysis of the resume."""
        try:
            # Load the document
            resume_text = self.load_document(file_path)

            # Run all analyses
            basic_info = self.extract_basic_info(resume_text)
            formatting_analysis = self.analyze_formatting(resume_text)
            content_analysis = self.analyze_content_quality(resume_text)
            ats_analysis = self.analyze_ats_compatibility(resume_text)

            # Job matching analysis (optional)
            job_match = None
            if job_description:
                job_match = self.analyze_job_match(resume_text, job_description)

            # Compile comprehensive report
            analysis_report = {
                "basic_information": basic_info,
                "formatting_analysis": formatting_analysis,
                "content_quality": content_analysis,
                "ats_compatibility": ats_analysis,
                "job_matching": job_match,
                "overall_recommendations": self._generate_overall_recommendations(
                    formatting_analysis, content_analysis, ats_analysis, job_match
                ),
            }

            return analysis_report

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _generate_overall_recommendations(
        self, formatting, content, ats, job_match
    ) -> List[str]:
        """Generate overall recommendations based on all analyses."""
        recommendations = []

        # Priority recommendations based on scores
        if formatting.get("overall_format_score", 0) < 7:
            recommendations.append("Focus on improving resume formatting and structure")

        if content.get("achievement_quality_score", 0) < 7:
            recommendations.append(
                "Strengthen achievement descriptions with quantifiable results"
            )

        if ats.get("ats_score", 0) < 7:
            recommendations.append(
                "Optimize for ATS compatibility with better keyword usage"
            )

        if job_match and job_match.get("match_score", 0) < 70:
            recommendations.append(
                "Customize resume to better align with the target job requirements"
            )

        return recommendations


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ResumeAnalyzer()

    # Get script directory and file path
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, "resume.pdf")

    # Optional job description for matching
    job_description = """
    We are looking for a Senior Software Engineer with experience in Python, 
    machine learning, and web development. The ideal candidate should have 
    5+ years of experience and expertise in Django, React, and cloud platforms.
    """

    # Generate comprehensive analysis
    try:
        analysis = analyzer.generate_comprehensive_analysis(file_path, job_description)

        # Print results in a formatted way
        print("=== RESUME ANALYSIS REPORT ===\n")

        if "error" in analysis:
            print(f"Error: {analysis['error']}")
        else:
            # Basic Information
            print("EXTRACTED INFORMATION:")
            basic_info = analysis["basic_information"]
            print(f"Name: {basic_info.get('name', 'N/A')}")
            print(f"Location: {basic_info.get('location', 'N/A')}")
            print(
                f"Skills: {len(basic_info.get('skills', {}).get('technical_skills', []))} technical skills found"
            )
            print()

            # Scores Summary
            print("ANALYSIS SCORES:")
            print(
                f"Formatting Score: {analysis['formatting_analysis'].get('overall_format_score', 'N/A')}/10"
            )
            print(
                f"Content Quality Score: {analysis['content_quality'].get('achievement_quality_score', 'N/A')}/10"
            )
            print(
                f"ATS Compatibility Score: {analysis['ats_compatibility'].get('ats_score', 'N/A')}/10"
            )
            if analysis["job_matching"]:
                print(
                    f"Job Match Score: {analysis['job_matching'].get('match_score', 'N/A')}/100"
                )
            print()

            # Key Recommendations
            print("KEY RECOMMENDATIONS:")
            for i, rec in enumerate(analysis.get("overall_recommendations", []), 1):
                print(f"{i}. {rec}")

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
