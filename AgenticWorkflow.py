import os
import json
import asyncio
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv
import nest_asyncio
import regex as re
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
    InputRequiredEvent,
    HumanResponseEvent
)


# Apply nest_asyncio to enable nested asyncio event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

class ParseFormEvent(Event):
    application_form: str

class QueryEvent(Event):
    query: str
    field: str

class ResponseEvent(Event):
    field: str
    response: str

class NextFieldEvent(Event):
    pass

class FormCompletedEvent(Event):
    pass

class FeedbackEvent(Event):
    feedback: str

class InsuranceFormWorkflow(Workflow):
    def __init__(self, timeout=120, verbose=False):
        super().__init__(timeout=timeout, verbose=verbose)
        self.storage_dir = "./workflow_data"
        os.makedirs(self.storage_dir, exist_ok=True)
        self.vector_dir = "./vector_storage"
        os.makedirs(self.vector_dir, exist_ok=True)
        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llama_parse_api_key = os.getenv("LLAMA_PARSE_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")
        if not self.llama_parse_api_key:
            raise ValueError("LLAMA_PARSE_API_KEY must be set in environment variables")
        # Initialize the LLM
        self.llm = OpenAI(model="gpt-4o-mini", api_key=self.openai_api_key)
        self.query_engine = None
        self.company_info = None

    @step
    async def set_up(self, ctx: Context, ev: StartEvent) -> ParseFormEvent:
        if not ev.company_id:
            raise ValueError("No company ID provided")
        if not ev.application_form:
            raise ValueError("No application form provided")
        print(f"Starting workflow for company {ev.company_id}")
        # Get company data (including both structured and unstructured data)
        company_data = self.get_company_data(ev.company_id)
        # Store company data in context
        await ctx.set("company_data", company_data)
        # Store company metadata for prompt enrichment
        self.company_info = company_data.get("company_meta", {})
        company_info_str = json.dumps(self.company_info, indent=2)
        await ctx.set("company_info_str", company_info_str)
        # Create vector index from markdown content
        self.create_vector_index(ev.company_id, company_data["md_content"])
        # Initialize feedback
        await ctx.set("has_feedback", False)
        await ctx.set("feedback", "")
        # Move to parsing the form
        return ParseFormEvent(application_form=ev.application_form)

    def get_company_data(self, company_id):
        """Get company data from file or API."""
        # Check if we already have the data
        company_dir = os.path.join(self.storage_dir, f"company_{company_id}")
        md_path = os.path.join(company_dir, "content.md")
        meta_path = os.path.join(company_dir, "metadata.json")
        if os.path.exists(md_path) and os.path.exists(meta_path):
            # Data exists, load it
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            with open(meta_path, "r", encoding="utf-8") as f:
                company_meta = json.load(f)
            print(f"Loaded existing company memory and structured data for company {company_id}")
            return {
                "md_content": md_content,
                "company_meta": company_meta
            }
        # Data doesn't exist, extract it from API
        print(f"Fetching company memory and structured data for company {company_id} from API")
        return self.extract_company_data(company_id)

    def extract_company_data(self, company_id):
        """Extract company data from API including unstructured call transcripts and structured data."""
        memory_endpoint = "https://tatch.retool.com/url/memory-interview"
        memory_api_key = os.getenv("MEMORY_API_KEY", "retool_wk_dc0b4514fc4545d99d78175c985010bb")
        memory_headers = {
            "X-Workflow-Api-Key": memory_api_key,
            "Content-Type": "application/json"
        }
        try:
            # Get company memory using POST request
            response = requests.post(
                memory_endpoint,
                headers=memory_headers,
                json={"company_id": company_id}
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            # Parse the data
            memory_data = response.json()
            # Extract markdown content (contains call transcripts and other unstructured data)
            if "company" in memory_data and "md" in memory_data["company"]:
                md_content = memory_data["company"]["md"]
            # Extract company metadata (structured data from lead generation sources)
            company_meta = {}
            if "company" in memory_data and "json" in memory_data["company"] and "company" in memory_data["company"]["json"]:
                company_meta = memory_data["company"]["json"]["company"]
            # Save to disk
            company_dir = os.path.join(self.storage_dir, f"company_{company_id}")
            os.makedirs(company_dir, exist_ok=True)
            # Save markdown content
            md_path = os.path.join(company_dir, "content.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            # Save metadata
            meta_path = os.path.join(company_dir, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "company_id": company_id,
                    "company_name": company_meta.get("company_name", f"Company_{company_id}"),
                    "metadata": company_meta
                }, f, indent=2)
            print(f"Successfully retrieved and saved company memory for {company_id}")
            return {
                "md_content": md_content,
                "company_meta": company_meta
            }
        except requests.exceptions.RequestException as e:
            print(f"Error extracting company memory: {e}")
            return {
                "md_content": f"Error retrieving data for company {company_id}: {str(e)}",
                "company_meta": {"company_name": f"Company_{company_id}"}
            }

    def create_vector_index(self, company_id, md_content):
        """Create a vector index from markdown content and save it for querying."""
        vector_dir = os.path.join(self.vector_dir, f"company_{company_id}")
        # Check if vector index already exists
        if os.path.exists(vector_dir):
            print(f"Loading existing vector index for company {company_id}")
            storage_context = StorageContext.from_defaults(persist_dir=vector_dir)
            index = load_index_from_storage(storage_context)
        else:
            print(f"Creating vector index for company {company_id}")
            # Process markdown content into chunks
            chunks = self.chunk_markdown(md_content)
            documents = [Document(text=chunk) for chunk in chunks]
            # Create vector index
            embed_model = OpenAIEmbedding(model_name="text-embedding-3-small", api_key=self.openai_api_key)
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=embed_model
            )
            # Save vector index
            index.storage_context.persist(persist_dir=vector_dir)
            print(f"Vector index saved for company {company_id}")
        # Create query engine
        self.query_engine = index.as_query_engine(
            llm=self.llm,
            similarity_top_k=3,
            response_mode="compact"
        )

    def chunk_markdown(self, md_content, chunk_size=1000, overlap=200):
        """Split markdown content into overlapping chunks for better vector search."""
        # Simple chunking by characters with overlap
        chunks = []
        for i in range(0, len(md_content), chunk_size - overlap):
            chunk = md_content[i:i + chunk_size]
            if len(chunk) > 200:  # Only include chunks that have enough content
                chunks.append(chunk)
        return chunks

    @step
    async def parse_form(self, ctx: Context, ev: ParseFormEvent) -> QueryEvent | StopEvent:
        print("Parsing insurance application form...")
        try:
            # Parse the form asynchronously using LlamaParse
            parser = LlamaParse(
                api_key=self.llama_parse_api_key,
                result_type="markdown",
                content_guideline_instruction="Extract ALL fillable fields from this insurance application form.  A field is any element where a user would enter information.",
                formatting_instruction="Return a comprehensive list containing ONLY the names of each fillable field identified in the form.  Ensure no commentary is included, and that they are seperated with commas."
            )
            # Use asynchronous parsing method
            result = await parser.aload_data(ev.application_form)
            parsed_text = result[0].text

            # Directly extract fields from LlamaParse output
            fields = [s.strip() for s in parsed_text.split(",")]

            if not fields:
                print("No fields found in the insurance form.")
                return StopEvent(result="No fields found in the insurance form.")

            print(f"Found {len(fields)} fields: {', '.join(fields)}")
            await ctx.set("fields_to_fill", fields)
            await ctx.set("responses", {})
            await ctx.set("current_field_index", 0)

            # Begin with the first field
            first_field = fields[0]

            # Check if we have feedback from a previous iteration
            has_feedback = await ctx.get("has_feedback")
            feedback = await ctx.get("feedback")

            query = f"What is the appropriate value for this insurance form field: {first_field}"
            if has_feedback:
                query += f"\n\nConsider this feedback when answering: {feedback}"

            return QueryEvent(field=first_field, query=query)

        except Exception as e:
            print(f"Error parsing insurance form: {e}")
            return StopEvent(result=f"Error parsing insurance form: {str(e)}")

    @step
    async def ask_question(self, ctx: Context, ev: QueryEvent) -> ResponseEvent:
        print(f"Processing insurance form field: {ev.field}")
        try:
            # Get company info string for context
            company_info_str = await ctx.get("company_info_str")
            # Use the query engine to extract relevant information
            response = self.query_engine.query(
                f"""You are filling out an insurance application form. The field to fill in is: {ev.field} Use only the information from the retrieved context to determine the appropriate value for this field. Here is additional structured information about the company: {company_info_str} {ev.query} Provide a concise, accurate answer using only the information available about the company. If the information cannot be determined from the context and company data, respond with "Information not available". """
            )
            print(f"Generated response for field '{ev.field}'")
            return ResponseEvent(field=ev.field, response=str(response))
        except Exception as e:
            print(f"Error extracting information for field {ev.field}: {e}")
            return ResponseEvent(
                field=ev.field, response=f"Unable to generate response for this field. Error: {str(e)}"
            )

    @step
    async def process_response(self, ctx: Context, ev: ResponseEvent) -> NextFieldEvent | FormCompletedEvent:
        # Store the response
        responses = await ctx.get("responses")
        responses[ev.field] = ev.response
        await ctx.set("responses", responses)
        # Get field information
        fields = await ctx.get("fields_to_fill")
        current_index = await ctx.get("current_field_index")
        # Move to the next field
        current_index += 1
        await ctx.set("current_field_index", current_index)
        # Check if we have more fields to process
        if current_index < len(fields):
            print(f"Moving to field {current_index+1} of {len(fields)}")
            return NextFieldEvent()
        else:
            print("All insurance form fields processed, preparing form for review")
            return FormCompletedEvent()

    @step
    async def process_next_field(self, ctx: Context, ev: NextFieldEvent) -> QueryEvent:
        # Get information for the next field
        fields = await ctx.get("fields_to_fill")
        current_index = await ctx.get("current_field_index")
        next_field = fields[current_index]
        # Check if we have feedback from a previous iteration
        has_feedback = await ctx.get("has_feedback")
        feedback = await ctx.get("feedback")
        query = f"What is the appropriate value for this insurance form field: {next_field}"
        if has_feedback:
            query += f"\n\nConsider this feedback when answering: {feedback}"
        return QueryEvent(field=next_field, query=query)

    @step
    async def form_completed(self, ctx: Context, ev: FormCompletedEvent) -> InputRequiredEvent:
        # Get all responses
        responses = await ctx.get("responses")
        company_data = await ctx.get("company_data")
        company_name = company_data.get("company_meta", {}).get("company_name", "Unknown Company")
        # Format responses for display in a well-structured format
        formatted_responses = f"Insurance Application Form for {company_name}\n\n"
        formatted_responses += "\n\n".join([
            f"Field: {field}\nValue: {response}" for field, response in responses.items()
        ])
        # Store the form for later
        await ctx.set("filled_form", formatted_responses)
        # Ask for human feedback
        return InputRequiredEvent(
            prefix="How does this form look? Give me any feedback or say 'looks good' to finalize.",
            result=formatted_responses
        )

    @step
    async def get_feedback(self, ctx: Context, ev: HumanResponseEvent) -> QueryEvent | StopEvent:
        # Process the human feedback
        print("Received feedback:", ev.response)
        
        # Use LLM to interpret feedback
        verdict_response = self.llm.complete(
            f"""You have received some human feedback on the insurance form-filling task you've done. Does everything look good, or is there more work to be done?
            The feedback is: {ev.response}.
            Answer "GOOD" if the feedback indicates the form is complete and correct. Answer "BAD" if the feedback indicates that further work is needed.
            """
        )
        verdict = verdict_response.text.strip().upper()
        
        if "GOOD" in verdict:
            print("Form filling task complete based on feedback.")
            return StopEvent(result="Form filling task complete.")
        else:
            # Store feedback for future reference
            await ctx.set("has_feedback", True)
            await ctx.set("feedback", ev.response)
            
            # Get fields and responses
            fields = await ctx.get("fields_to_fill")
            responses = await ctx.get("responses")
            
            # Check if fields is not empty before proceeding
            if not fields:
                print("No fields available for feedback.")
                return StopEvent(result="No fields available for feedback.")
            
            # Use LLM to analyze feedback and identify specific fields that need correction
            field_analysis = self.llm.complete(
                f"""The user has provided feedback on an insurance form with the following fields: {", ".join(fields)}
                
                Current form responses:
                {json.dumps(responses, indent=2)}
                
                The feedback is: "{ev.response}"
                
                Which specific fields need to be corrected based on this feedback? Return ONLY a comma-separated list of field names that need correction.
                If no specific fields are mentioned, return "ALL" to indicate a general revision is needed.
                """
            ).text.strip()
            
            # Process the field analysis
            if field_analysis.upper() == "ALL":
                # General revision needed, start from the beginning
                await ctx.set("current_field_index", 0)
                current_field = fields[0]
                print(f"General revision needed, restarting with field '{current_field}'")
            else:
                # Extract field names from the analysis
                fields_to_correct = [f.strip() for f in field_analysis.split(",")]
                
                # Find the first field that needs correction and exists in our field list
                identified_field = None
                for field in fields_to_correct:
                    if field in fields:
                        identified_field = field
                        break
                
                if identified_field:
                    # Set the current index to the identified field
                    field_index = fields.index(identified_field)
                    await ctx.set("current_field_index", field_index)
                    current_field = identified_field
                    print(f"Identified field for correction: '{current_field}'")
                else:
                    # If we can't identify any specific field, start from the beginning
                    await ctx.set("current_field_index", 0)
                    current_field = fields[0]
                    print(f"Could not identify specific field, restarting with '{current_field}'")
            
            # Prepare query with feedback
            query = f"What is the appropriate value for this insurance form field: {current_field}"
            query += f"\n\nConsider this feedback when answering: {ev.response}"
            
            print(f"Re-querying field '{current_field}' with feedback.")
            return QueryEvent(field=current_field, query=query)
        

# Helper function to parse the form output
def parse_form_output(form_text):
    """Parse the form output text into a structured dictionary."""
    form_data = {}
    
    # Skip the header line
    content = form_text.split("\n\n", 1)[1] if "\n\n" in form_text else form_text
    
    # Split by field entries
    entries = content.split("\n\n")
    for entry in entries:
        lines = entry.strip().split("\n")
        if len(lines) >= 2:
            field_name = lines[0].replace("Field: ", "")
            field_value = lines[1].replace("Value: ", "")
            form_data[field_name] = field_value
    
    return form_data


async def run_workflow(company_id, application_form_path, output_path=None):
    """Run the insurance form completion workflow with the given company ID and application form."""
    print(f"Running insurance form workflow for company {company_id} with form {application_form_path}")
    
    # Create the workflow
    workflow = InsuranceFormWorkflow(timeout=300, verbose=True)
    
    # Determine output paths
    if output_path is None:
        output_path = f"insurance_form_{company_id}_filled"
    
    json_output_path = f"{output_path}.json"
    md_output_path = f"{output_path}.md"
    
    try:
        handler = workflow.run(
            company_id=company_id,
            application_form=application_form_path
        )
        
        # Store the formatted form for saving later
        formatted_form = None
        
        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                # This is the event where we get the formatted form
                formatted_form = event.result
                
                # Display form completion results
                print("\n======= Form Completion Results =======")
                print(formatted_form)
                print("======================================\n")
                
                # Get text feedback from the user
                user_feedback = input(f"{event.prefix}\n> ")
                
                # Send feedback back to the workflow
                handler.ctx.send_event(
                    HumanResponseEvent(response=user_feedback)
                )
        
        # Get final result
        final_result = await handler
        print("\n======= Insurance Form Completed! =======")
        print(final_result)
        print("========================================\n")
        
        # If we didn't capture the formatted form earlier, use the final result
        if not formatted_form:
            formatted_form = str(final_result)
        
        # Save the formatted form to markdown file
        with open(md_output_path, "w", encoding="utf-8") as f:
            f.write(formatted_form)
        
        print(f"Filled insurance form saved to {md_output_path}")
        
        # Parse form data for JSON export
        form_data = parse_form_output(formatted_form)
        
        # Save form data as JSON
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(form_data, f, indent=2)
        
        print(f"Form data saved as JSON to {json_output_path}")
        
        return final_result
    
    except Exception as e:
        print(f"Error in insurance form workflow: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 3:
        company_id = sys.argv[1]
        application_form_path = sys.argv[2]
        output_path = sys.argv[3]
    elif len(sys.argv) > 2:
        company_id = sys.argv[1]
        application_form_path = sys.argv[2]
        output_path = None
    else:
        company_id = input("Enter company ID: ")
        application_form_path = input("Enter path to insurance application form: ")
        output_path = input("Enter base path for output files (leave empty for default): ")
        if not output_path.strip():  # Check if the input is empty or just whitespace
            output_path = None
    
    # Run the workflow
    asyncio.run(run_workflow(company_id, application_form_path, output_path))
