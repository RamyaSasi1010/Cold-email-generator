import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Chain:
    def __init__(self):
        # Load API key securely from the environment variable
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key is missing! Please add it to your .env file.")
        
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=api_key, 
            model_name="llama-3.1-8b-instant"
        )

    def extract_jobs(self, cleaned_text):
        # Create a prompt for extracting job details from the scraped text
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        
        # Set up the chain for extraction
        chain_extract = prompt_extract | self.llm
        
        # Execute the chain with the input cleaned text
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        
        try:
            # Parse the response to valid JSON format
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        
        # Return result
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        # Create a prompt for writing a cold email based on the job description
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Mohan, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase AtliQ's portfolio: {link_list}
            Remember you are Mohan, BDE at AtliQ. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )
        
        # Set up the chain for email generation
        chain_email = prompt_email | self.llm
        
        # Execute the chain to generate the email
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        
        return res.content

if __name__ == "__main__":
    # Debugging: Ensure the API key is correctly loaded (in development only)
    print("Groq API Key:", os.getenv("GROQ_API_KEY"))
    
    # Create the Chain object and test
    try:
        chain = Chain()  # Will raise error if the API key is missing
        print("Chain object created successfully.")
    except Exception as e:
        print(f"Error: {e}")
