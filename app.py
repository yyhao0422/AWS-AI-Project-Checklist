import streamlit as st
import pandas as pd
import boto3
import json
import io
from io import BytesIO
import os
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Proposal & WBS Processor",
    page_icon="📊",
    layout="wide"
)

# Initialize Bedrock client
def get_bedrock_client():
    try:
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv("AWS_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        return bedrock_client
    except Exception as e:
        st.error(f"Error initializing Bedrock client: {str(e)}")
        return None

# Function to read and encode files for API
def read_file_for_api(file):
    """Read file and prepare it for sending to Bedrock API"""
    if file is None:
        return None
    
    try:
        # Get file content as bytes
        file_content = file.getvalue()
        
        # Get file type and name
        file_type = file.type
        file_name = file.name
        
        # Return file info and content
        return {
            "file_name": file_name,
            "file_type": file_type,
            "content": file_content  # Keep as bytes for Converse API
        }
    except Exception as e:
        st.error(f"Error reading file {file.name}: {str(e)}")
        return None

# Function to process files with Bedrock
def process_with_bedrock(proposal_file, wbs_file, model_id):
    client = get_bedrock_client()
    if not client:
        return None
    
    try:
        # Read files
        proposal_data = read_file_for_api(proposal_file)
        wbs_data = read_file_for_api(wbs_file)
        
        if not proposal_data or not wbs_data:
            st.error("Failed to read one or both files")
            return None
        
        # Prepare the message with file attachments for Converse API
        user_message = {
            "role": "user",
            "content": [
                {
                    "text": """
                    I have a project proposal and a Work Breakdown Structure (WBS).
                    
                    Please analyze these documents and provide a checklist of the AWS service configuration items that are important to deliver to the customer as stated in the WBS.
                    Please also analyze the number of accounts needed to deliver the WBS and provide a checklist for each account.
                    
                    Provide the output in CSV format with the following columns: Service Type,Configuration Item,Status,Importance,Reference WBS Item
                    
                    Only provide the CSV data with no markdown formatting, explanations, or other text. The first line should be the header row.
                    Example:
                    Account,Service Type,Configuration Item,Reference WBS Item
                    Management Account,Account Baseline,AWS CloudTrail Setup,2.01.2
                    Log Archive Account,Account Baseline,AWS Config Setup,2.01.3
                    """
                },
                {
                    "document": {
                        "name": "projectproposal",
                        "format": "pdf",
                        "source": {
                            "bytes": proposal_data["content"]
                        }
                    }
                },
                {
                    "document": {
                        "name": "wbs",
                        "format": "xlsx",
                        "source": {
                            "bytes": wbs_data["content"]
                        }
                    }
                }
            ]
        }
        
        # Inference configuration
        inference_config = {
            "temperature": 0.7,
            "maxTokens": 8192
        }
            
        # Call Bedrock Converse API
        st.info(f"Calling model: {model_id}")
        st.info(f"Sending proposal file: {proposal_data['file_name']} and WBS file: {wbs_data['file_name']}")
        
        response = client.converse(
            modelId=model_id,
            messages=[user_message],
            inferenceConfig=inference_config
        )
        
        # Parse response from Converse API
        output_message = response.get('output', {}).get('message', {})
        content = ""
        
        # Extract text content from response
        for content_item in output_message.get('content', []):
            if 'text' in content_item:
                content += content_item['text']
        
        # Show the raw response content
        st.subheader("Analysis Results")
        
        csv_file = create_csv_from_response(content)
        if csv_file:
            st.success("Successfully created CSV file from response")
            
            # Display download button
            st.download_button(
                label="Download AWS Service Checklist",
                data=csv_file,
                file_name="aws_service_checklist.csv",
                mime="text/csv"
            )
        
        # Display token usage if available
        if 'usage' in response:
            st.info(f"Input tokens: {response['usage'].get('inputTokens', 'N/A')}")
            st.info(f"Output tokens: {response['usage'].get('outputTokens', 'N/A')}")
            st.info(f"Total tokens: {response['usage'].get('totalTokens', 'N/A')}")
        
        return content
            
    except Exception as e:
        st.error(f"Error calling Bedrock: {str(e)}")
        
        # Add more detailed error information
        error_str = str(e)
        if "ValidationException" in error_str:
            st.error("Request format validation failed. Details:")
            st.code(error_str)
        
        return None

# Function to create CSV file from the model response
def create_csv_from_response(response_text):
    try:
        # Create a StringIO object to store the CSV content
        output = io.StringIO()
        
        # Clean up the response text to extract just the CSV data
        lines = response_text.strip().split('\n')
        csv_lines = []
        
        # Filter out any non-CSV content
        in_csv = False
        for line in lines:
            line = line.strip()
            # Check if this looks like a CSV line (contains commas and no markdown table formatting)
            if ',' in line and not line.startswith('|') and not line.startswith('```'):
                in_csv = True
                csv_lines.append(line)
            elif line.startswith('```csv'):
                in_csv = True
                continue
            elif line.startswith('```') and in_csv:
                in_csv = False
                continue
            elif in_csv and line:
                csv_lines.append(line)
        
        # Write the CSV content to the StringIO object
        if csv_lines:
            for line in csv_lines:
                output.write(line + '\n')
        else:
            # If no CSV format detected, try to convert the entire response
            output.write(response_text)
        
        # Get the CSV content as a string
        output.seek(0)
        return output.getvalue()
    
    except Exception as e:
        st.error(f"Error creating CSV file: {str(e)}")
        return None

# Main app
def main():
    st.title("Proposal & WBS Processor")
    
    st.markdown("""
    This app processes your project proposal and Work Breakdown Structure (WBS) files,
    analyzes them using AI, and provides a checklist of AWS services and configurations.
    """)
    
    # File upload section
    st.header("Upload Files")
    col1, col2 = st.columns(2)
    
    with col1:
        proposal_file = st.file_uploader("Upload Proposal Document", type=["docx", "pdf", "txt"])
    
    with col2:
        wbs_file = st.file_uploader("Upload WBS Excel File", type=["xlsx", "xls", "csv"])

    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"  
    
    # Process button
    if st.button("Generate Analysis", type="primary", disabled=not (proposal_file and wbs_file)):
        with st.spinner("Processing files with AI..."):
            # Process with Bedrock
            bedrock_response = process_with_bedrock(proposal_file, wbs_file, model_id)
            
            if not bedrock_response:
                st.error("Failed to get a valid response from Bedrock.")
                

if __name__ == "__main__":
    main()
