import os                 
from pathlib import Path
import time
from dotenv import load_dotenv  # library to access the env file or varaibles
from groq import Groq # This is an LLM like Open AI etc
from pydantic import BaseModel
import json
from file_reader import extract_text


load_dotenv() # loading env

my_key = os.getenv("GROQ_API_KEY") # accessing the env variable value 
#print(my_key)
if not my_key:
    raise ValueError("API not found")

my_Client = Groq(api_key=my_key) #connect to client using api key 
#print(document_text)

Requirement = """About the Role
We're hiring an AI Engineer who has real hands-on experience developing applications using Python, LLM APIs and AI agents.
This is a production engineering role, not a research internship, prompt engineering role or basic chatbot development position.
You will work on implementing AI agents for consumer applications and enterprise workflows, including integrations with APIs, automated processes and human approval systems.
Must-Have Technical Experience
We are specifically looking for engineers with:
2–3 years of professional software development experience, preferably involving Python and AI applications.
Strong hands-on Python programming and backend development skills.
Experience with REST APIs, JSON and Git.
Practical experience developing applications using OpenAI, Claude, Gemini or comparable LLM APIs.
Experience implementing AI agents with tool calling, routing, workflows or agent orchestration.
Ability to debug API failures, handle exceptions and implement retries.
Understanding of prompt evaluation, testing and LLM failure handling.
Familiarity with Claude Code and AI-assisted development.
At least one working AI application or agent-based project they can demonstrate.
What You Will Build
AI agents configured using YAML and Python.
LLM-powered workflows with tool integrations.
Multi-agent routing and human-in-the-loop processes.
Backend integrations connecting AI agents with business systems.
Automated tests and evaluation datasets.
Production improvements involving reliability, latency and cost optimization.
Good to Have LangGraph, CrewAI or AutoGen, FastAPI, AWS, Azure or GCP
n8n or workflow automation tools
Android or IoT integration
Production application monitoring
Before You Apply
Please ensure that you can explain at least one AI application you have personally developed, including:
What problem the application solves.
Which LLM APIs, tools and frameworks you used.
What you personally implemented.
What technical problems you encountered.
How you tested, debugged or improved the application.
Candidates should be prepared for a practical AI agent development assessment and technical discussion.
We are looking for demonstrated engineering ability, not certifications or copied tutorial projects.
"""
class Candidate(BaseModel):
    name : str
    Skill: str
    EXperience: str
    Projects: str #= Field(alias="email")
    Percentage: str #= Field(alias="contact_number")


def ParseResume(Resum):
    role = "user" #providing role ie., user/assist
    
    req = f""""based on the extracted information by system role read the where it should meet all
               the criteria of may of them {Requirement} and what percentage of match
               IMPORTANT:
               Dont invent Data
               Provide the exact data 
               check for the data
                -- Percentage value given based on constraints like weak, average, Good"""


    sys_prompt =f"""You are a Resume parser extract the required data from  {document_text} and provide output as json format as {Candidate.model_json_schema()} 
                -- Dont invent data 
                -- provide in specific main constraints only
                -- Make the list of the skills in order and"""
               
    message_sys={
            "role":"system",
            "content":sys_prompt
        }
        #for n in prompts:
    
    message={
                "role":role,
                "content":req
        }

    messas=[message_sys,message]
    
    model_name = "openai/gpt-oss-120b"
    
            #response from LLM
    response = my_Client.chat.completions.create(
                model=model_name,
                messages=messas,
                temperature=1,
                response_format={"type": "json_object"}
        )
    
    answer=response.choices[0].message.content
        #print(answer) #first message of response
        #usages = response.usage
        # print(f"{n} --> tokens prompt used:{usages.prompt_tokens} compeletion used tokens:{usages.completion_tokens} ! text:{response.choices[0].finish_reason}")
    
    datar = json.loads(answer)
    
    json_data1 = json.dumps(datar, indent=5)
    #print(json_data1)
    return json_data1

# Extract document content
def Resume_Text(document_text):
    
    role = "user" #providing role ie., user/assist

    req = f""""Analyze the following candidate resume against the provided job description.

CANDIDATE RESUME:
{document_text}

JOB DESCRIPTION:
{Requirement}

Perform the complete analysis according to your system instructions.

Return:
1. Overall match percentage.
2. Matching skills.
3. Partially matching skills.
4. Missing skills.
5. Technical skill comparison.
6. Experience and education compatibility.
7. ATS keyword analysis.
8. Resume improvement suggestions.
9. Recommended projects.
10. Final recruiter assessment.

Return the result in valid JSON format.
                 -- Percentage given value  based on constraints like weak, average, Good"""

    sys_prompt =f"""You are an expert ATS Resume Analyst and Technical Recruitment Specialist.
Your task is to analyze the candidate's resume against the provided job description and generate a detailed, structured resume evaluation report.
INPUT
Candidate Resume: {document_text}
Job Description: {Requirement}
ANALYSIS REQUIREMENTS
1. Resume and Job Description Analysis
Extract the candidate's technical skills, programming languages, tools, frameworks, education, certifications, and work experience.
Identify the key responsibilities and required skills from the job description.
Understand mandatory requirements versus preferred or optional skills.
2. Skill Matching
Compare the candidate's skills with the job requirements.
Categorize each skill into:
Exact Match: Skill explicitly present in the resume.
Partial Match: Related knowledge or limited exposure.
Missing Skill: Not mentioned or demonstrated in the resume.
Do not assume that a candidate knows a skill just because a related skill is mentioned.
3. Missing Skills Identification
Identify technical skills, tools, technologies, and qualifications missing from the resume.
Distinguish critical missing skills from optional skills.
Explain why each missing skill matters for the role.
4. Resume Improvement Suggestions
Suggest improvements to the Professional Summary.
Recommend relevant technical skills to highlight.
Identify work experience points that should be rewritten or emphasized.
Suggest relevant projects that could strengthen the candidate's profile.
Recommend keywords from the job description that can be truthfully incorporated into the resume.
Never fabricate work experience, project contributions, certifications, or technical expertise.
5. ATS Compatibility Analysis
Evaluate keyword alignment with the job description.
Identify missing important keywords.
Check whether the resume clearly communicates relevant experience.
Suggest improvements to section organization and formatting.
6. Overall Evaluation
Calculate an estimated Resume–Job Description Match Percentage based on skill alignment, experience, education, and responsibilities.
Provide a separate technical skill match percentage.
Highlight the candidate's strongest matching areas.
Identify the top five areas requiring improvement.
Assess whether the candidate is a Strong Match, Moderate Match, or Low Match.
OUTPUT FORMAT
Generate the report using the following structure:
Overall Match Score (%)
Candidate Profile Summary
Job Requirement Breakdown
Matching Skills
Partially Matching Skills
Missing Skills
Technical Skill Comparison Table
For the comparison table, use these columns:
Required Skill
Resume Evidence
Match Status
Improvement Needed
Experience and Education Compatibility
ATS Keyword Analysis
Resume Improvement Recommendations
Suggested Professional Summary
Recommended Projects
Top 5 Skills to Learn
Final Recruiter Assessment
IMPORTANT RULES
Base your analysis strictly on the supplied resume and job description.
Do not invent candidate qualifications or experience.
Do not treat missing information as proof that the candidate lacks the skill; classify it as "Not Mentioned".
Clearly distinguish professional experience, personal projects, certifications, and theoretical knowledge.
Do not give an inflated match percentage.
Explain the reasoning behind the match score.
Provide actionable and realistic recommendations.
Use professional, clear, concise language.
Return the complete analysis as a well-structured report with headings, tables, and bullet points. """
    
    message_sys={
            "role":"system",
            "content":sys_prompt
        }
    
    message={
                "role":role,
                "content":req
        }
    
    messas=[message_sys,message]
    
    model_name = "openai/gpt-oss-120b"
    
            #response from LLM
    response = my_Client.chat.completions.create(
                model=model_name,
                messages=messas,
                temperature=1,
                response_format={"type": "json_object"}
        )
    
    answer=response.choices[0].message.content
        #print(answer) #first message of response
        #usages = response.usage
        # print(f"{n} --> tokens prompt used:{usages.prompt_tokens} compeletion used tokens:{usages.completion_tokens} ! text:{response.choices[0].finish_reason}")
    
    datar = json.loads(answer)
    
    json_data1 = json.dumps(datar, indent=5)
    #print(json_data1)
    return json_data1

all_results=[]
folder_path = Path(__file__).parent / "Resumess"

if not folder_path.exists():
    print("Resume folder not found!")
else:
    for file_path in folder_path.iterdir():
        print(file_path)
    #C:\Users\Pratyush\padho_with_pratyush\week1\day5\resumes\abhay resume new - Abhay Singh.pdf
        if file_path.suffix.lower() not in [".pdf", ".docx"]:
            continue
        print("\nProcessing:", file_path.name)
        document_text = extract_text(file_path)
        Resum = Resume_Text(document_text)
        print(Resum)
        time.sleep(5)
        Resumee = ParseResume(Resum)
        print(Resumee)
        all_results=[Resumee]
        time.sleep(5)

