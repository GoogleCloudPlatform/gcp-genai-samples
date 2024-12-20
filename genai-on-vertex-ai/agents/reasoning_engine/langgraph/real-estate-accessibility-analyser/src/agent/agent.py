import json
from typing import Dict, List, Optional

import requests
import vertexai
from pydantic import BaseModel, Field

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from vertexai.preview import reasoning_engines

from src.config import PROJECT_ID, LOCATION, API_URL, STAGING_BUCKET

vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket="gs://"+STAGING_BUCKET)

model = ChatVertexAI(model="gemini-1.5-pro-002")

class Feature(BaseModel):
    feature: str = Field(description="Name of the accessibility feature")
    feature_score: float = Field(description="Score from 0-100")
    confidence: float = Field(description="Confidence level from 0-100")

class Issue(BaseModel):
    issue: str = Field(description="Description of accessibility issue")
    severity_score: float = Field(description="Severity score from 0-100")
    confidence: float = Field(description="Confidence level from 0-100")

class ImageAnalysis(BaseModel):
    features: List[Feature]
    issues: List[Issue]

class DescriptionAnalysis(BaseModel):
    features: List[Feature]
    issues: List[Issue]

class AccessibilityReport(BaseModel):
    global_accessibility_score: float
    description_analysis_summary: str
    image_analysis_summary: str

class EmailDraft(BaseModel):
    subject: str
    body: str


def get_property_data(property_id: str) -> Dict:
    """
    Retrieve comprehensive property data from a remote API, given a unique property identifier. 

    This tool makes a GET request to a backend API endpoint using the provided `property_id`. 
    Upon successful retrieval, it returns a dictionary containing detailed property information. 
    This includes the property's title, a description of it, a list of image urls, and 
    whether it is accessible or not. If the remote API request fails (for example, due to the property ID 
    not existing or network issues), an HTTP error is raised.

    Args:
        property_id (str): The unique ID of the property to retrieve from the API.

    Returns:
        Dict: A dictionary containing the full set of property data
    """
    url = f"{API_URL}/houses/{property_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def analyze_description(description: str) -> DescriptionAnalysis:
    """
    Analyze the accessibility aspects of a property description text, focusing on wheelchair-related features.

    This tool uses a language model to process a given textual description and identify key features 
    that support wheelchair accessibility, as well as potential issues or barriers. The analysis 
    returns structured data following the `DescriptionAnalysis` schema, which includes:
    - A list of recognized wheelchair-accessibility features, each with a name, a confidence score, and 
      a feature score.
    - A list of identified issues or accessibility barriers, each with a descriptive name, a severity score, 
      and a confidence level.

    Scores range from 0-100, and confidence indicates how certain the model is about the identified feature or issue.

    Args:
        description (str): The property’s textual description, including details about its layout, 
                           amenities, and environment.

    Returns:
        DescriptionAnalysis: A pydantic model instance containing the analysis results with 
                             `features` and `issues` lists.
    """
    text_message = {
        "type": "text",
        "text": f"""Analyze this property description for wheelchair accessibility:
        {description}

        Return the results in JSON strictly following the `DescriptionAnalysis` schema:
        {{
          "features": [
            {{
              "feature": "string",
              "feature_score": float,
              "confidence": float
            }}
          ],
          "issues": [
            {{
              "issue": "string",
              "severity_score": float,
              "confidence": float
            }}
          ]
        }}
        """
    }
    response = model.with_structured_output(DescriptionAnalysis).invoke([HumanMessage(content=[text_message])])
    return response


def analyze_images(images: List[str]) -> List[ImageAnalysis]:
    """
    Assess a collection of images to determine the presence of wheelchair-accessible features and identify potential issues.

    For each image URL provided, this tool uses a vision-capable language model to analyze visual cues 
    related to wheelchair accessibility. It examines elements such as ramps, wide doorways, and other 
    accessible design characteristics. It also looks for potential barriers, like steps, narrow passages, 
    or obstructions.

    The results adhere to the `ImageAnalysis` schema and include:
    - A list of detected accessibility features in the image, with feature names, scores, and confidence.
    - A list of identified issues in the image, each described with a severity score and confidence level.

    Args:
        images (List[str]): A list of image URLs representing the property’s visuals.

    Returns:
        List[ImageAnalysis]: A list of `ImageAnalysis` model instances, one per input image, each containing 
                             `features` and `issues` fields.
    """
    analyses = []
    for image_url in images:
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"{image_url}"}
        }
        text_message = {
            "type": "text",
            "text": """Analyze this image for wheelchair accessibility features and issues.
            
            Return results in JSON strictly following the `ImageAnalysis` schema:
            {
              "features": [
                {
                  "feature": "string",
                  "feature_score": float,
                  "confidence": float
                }
              ],
              "issues": [
                {
                  "issue": "string",
                  "severity_score": float,
                  "confidence": float
                }
              ]
            }
            """
        }
        response = model.with_structured_output(ImageAnalysis).invoke([HumanMessage(content=[text_message, image_message])])
        analyses.append(response)
    return analyses


def generate_accessibility_report(description_analysis: DescriptionAnalysis, image_analyses: List[ImageAnalysis]) -> AccessibilityReport:
    """
    Compile a comprehensive accessibility report for a property by integrating both textual and visual analyses.

    This tool takes the results from the description analysis and image analyses, merging them into a single 
    summary that provides a global accessibility score, a summary of the textual description findings, and 
    a summary of the visual (image-based) findings.

    The `global_accessibility_score` offers an aggregated metric representing the property’s wheelchair 
    accessibility level. The `description_analysis_summary` and `image_analysis_summary` explain the rationale 
    behind the score by detailing key features and issues discovered in the text and visuals.

    Args:
        description_analysis (DescriptionAnalysis): The result of analyzing the property's textual description.
        image_analyses (List[ImageAnalysis]): The results of analyzing multiple images of the property.

    Returns:
        AccessibilityReport: A pydantic model instance containing the integrated report, including a global score 
                             and summaries of the description and image analyses.
    """
    text_message = {
        "type": "text",
        "text": f"""Generate a comprehensive accessibility report based on these analyses:
        Description Analysis: {json.dumps(description_analysis.model_dump(), indent=2)}
        Image Analyses: {json.dumps([a.model_dump() for a in image_analyses], indent=2)}

        Return the results in JSON strictly following the `AccessibilityReport` schema:
        {{
          "global_accessibility_score": float,
          "description_analysis_summary": "string",
          "image_analysis_summary": "string"
        }}
        """
    }
    response = model.with_structured_output(AccessibilityReport).invoke([HumanMessage(content=[text_message])])
    return response


def draft_host_email(report: AccessibilityReport) -> Optional[EmailDraft]:
    """
    Draft a professional and courteous email to the property host requesting additional accessibility details, 
    if the determined accessibility score meets or exceeds a certain threshold.

    This tool examines the `global_accessibility_score` from the `AccessibilityReport`. If the score is 
    high enough (>= 70 in this case), it constructs a polite email that asks the host for more information 
    to further clarify and possibly improve the wheelchair accessibility of the property. The email is 
    returned adhering to the `EmailDraft` schema. If the score is below the threshold, indicating that 
    additional details are not currently necessary, this function returns None.

    Args:
        report (AccessibilityReport): The comprehensive accessibility report of the property.

    Returns:
        Optional[EmailDraft]: If the global accessibility score is >= 70, returns an `EmailDraft` model 
                              containing the email subject and body. Otherwise, returns None.
    """
    if report.global_accessibility_score >= 70:
        text_message = {
            "type": "text",
            "text": f"""Draft an email to the host asking for more accessibility details.
            Given this report: {json.dumps(report.model_dump(), indent=2)}

            Return the results in JSON strictly following the `EmailDraft` schema:
            {{
              "subject": "string",
              "body": "string"
            }}
            """
        }
        response = model.with_structured_output(EmailDraft).invoke([HumanMessage(content=[text_message])])
        return response
    else:
        return None

@tool
def get_property(property_id: str) -> Dict:
    """
    Retrieve comprehensive property data from a remote API, given a unique property identifier. 

    This tool makes a GET request to a backend API endpoint using the provided `property_id`. 
    Upon successful retrieval, it returns a dictionary containing detailed property information. 
    This includes the property's title, a description of it, a list of image urls, and 
    whether it is accessible or not. If the remote API request fails (for example, due to the property ID 
    not existing or network issues), an HTTP error is raised.

    Args:
        property_id (str): The unique ID of the property to retrieve from the API.

    Returns:
        Dict: A dictionary containing the full set of property data
    """
    return get_property_data(property_id)

@tool
def analyze_description_tool(description: str):
    """
    Analyze the accessibility aspects of a property description text, focusing on wheelchair-related features.

    This tool uses a language model to process a given textual description and identify key features 
    that support wheelchair accessibility, as well as potential issues or barriers. The analysis 
    returns structured data following the `DescriptionAnalysis` schema, which includes:
    - A list of recognized wheelchair-accessibility features, each with a name, a confidence score, and 
      a feature score.
    - A list of identified issues or accessibility barriers, each with a descriptive name, a severity score, 
      and a confidence level.

    Scores range from 0-100, and confidence indicates how certain the model is about the identified feature or issue.

    Args:
        description (str): The property’s textual description, including details about its layout, 
                           amenities, and environment.

    Returns:
        DescriptionAnalysis: A pydantic model instance containing the analysis results with 
                             `features` and `issues` lists.
    """
    analysis = analyze_description(description)
    return analysis.model_dump()

@tool
def analyze_images_tool(images: List[str]):
    """
    Assess a collection of images to determine the presence of wheelchair-accessible features and identify potential issues.

    For each image URL provided, this tool uses a vision-capable language model to analyze visual cues 
    related to wheelchair accessibility. It examines elements such as ramps, wide doorways, and other 
    accessible design characteristics. It also looks for potential barriers, like steps, narrow passages, 
    or obstructions.

    The results adhere to the `ImageAnalysis` schema and include:
    - A list of detected accessibility features in the image, with feature names, scores, and confidence.
    - A list of identified issues in the image, each described with a severity score and confidence level.

    Args:
        images (List[str]): A list of image URLs representing the property’s visuals.

    Returns:
        List[ImageAnalysis]: A list of `ImageAnalysis` model instances, one per input image, each containing 
                             `features` and `issues` fields.
    """
    analyses = analyze_images(images)
    return [a.model_dump() for a in analyses]

@tool
def generate_report_tool(description_analysis: dict, image_analyses: List[dict]):
    """
    Compile a comprehensive accessibility report for a property by integrating both textual and visual analyses.

    This tool takes the results from the description analysis and image analyses, merging them into a single 
    summary that provides a global accessibility score, a summary of the textual description findings, and 
    a summary of the visual (image-based) findings.

    The `global_accessibility_score` offers an aggregated metric representing the property’s wheelchair 
    accessibility level. The `description_analysis_summary` and `image_analysis_summary` explain the rationale 
    behind the score by detailing key features and issues discovered in the text and visuals.

    Args:
        description_analysis (DescriptionAnalysis): The result of analyzing the property's textual description.
        image_analyses (List[ImageAnalysis]): The results of analyzing multiple images of the property.

    Returns:
        AccessibilityReport: A pydantic model instance containing the integrated report, including a global score 
                             and summaries of the description and image analyses.
    """
    desc_obj = DescriptionAnalysis(**description_analysis)
    img_objs = [ImageAnalysis(**ia) for ia in image_analyses]
    report = generate_accessibility_report(desc_obj, img_objs)
    return report.model_dump()

@tool
def draft_email_tool(report: dict):
    """
    Draft a professional and courteous email to the property host requesting additional accessibility details, 
    if the determined accessibility score meets or exceeds a certain threshold.

    This tool examines the `global_accessibility_score` from the `AccessibilityReport`. If the score is 
    high enough (>= 70 in this case), it constructs a polite email that asks the host for more information 
    to further clarify and possibly improve the wheelchair accessibility of the property. The email is 
    returned adhering to the `EmailDraft` schema. If the score is below the threshold, indicating that 
    additional details are not currently necessary, this function returns None.

    Args:
        report (AccessibilityReport): The comprehensive accessibility report of the property.

    Returns:
        Optional[EmailDraft]: If the global accessibility score is >= 70, returns an `EmailDraft` model 
                              containing the email subject and body. Otherwise, returns None.
    """
    report_obj = AccessibilityReport(**report)
    email = draft_host_email(report_obj)
    if email:
        return email.model_dump()
    else:
        return {"message": "Global accessibility score below 70, not drafting email."}

tools = [
    get_property,
    analyze_description_tool,
    analyze_images_tool,
    generate_report_tool,
    draft_email_tool,
]

class PropertyAccessibilityApp:
    def __init__(self, project: str, location: str, staging_bucket: str):
        self.project_id = project
        self.location = location
        self.staging_bucket = staging_bucket
        vertexai.init(project=self.project_id, location=self.location, staging_bucket="gs://"+self.staging_bucket)

    def set_up(self) -> None:
        prompt = """
        You are an AI agent specializing in assessing wheelchair accessibility for properties. Your goal is to provide 
        accurate and concise answers based on a user's query by invoking the appropriate tool. You have access to the 
        following tools:

        1. **get_property(property_id: str)**: Fetches detailed property information, including title, description, 
           images, and accessibility status, using the provided property ID.

        2. **analyze_description_tool(description: str)**: Analyzes the property's textual description to extract accessibility 
           features and issues specifically related to wheelchair users.

        3. **analyze_images_tool(images: List[str])**: Evaluates images to detect accessibility features (e.g., ramps, wide 
           doorways) or barriers (e.g., steps, obstructions).

        4. **generate_report_tool(description_analysis: dict, image_analyses: List[dict])**:
           Generates a comprehensive report summarizing the accessibility aspects of the property, combining text and image 
           analyses. (Note: This tool requires the outputs from `analyze_description` and `analyze_images` as inputs.)

        5. **draft_email_tool(report: dict)**: Drafts a professional email to the property host requesting 
           additional accessibility details if the global accessibility score is high enough. (Note: This tool requires the 
           output from `generate_accessibility_report`.)

        Key Points:
        - For every query, determine if a tool needs to be invoked. If so, use **only one** tool per query. 
        - The selected tool should directly address the user's request.
        - Some tools depend on the outputs of other tools. For example:
          - `generate_accessibility_report` needs results from `analyze_description` and `analyze_images`.
          - `draft_host_email` needs the `AccessibilityReport` from `generate_accessibility_report`.
        - Plan your actions accordingly and guide the user through the process if multiple steps are needed.
        - If the query does not require a tool, respond directly without invoking any tools.
        - Ensure clear and structured communication of results to the user.
        - Make sure to output the result of the tool to the user in a pretty JSON format.

        Examples:
        - Query: "Can you analyze the description for property ID 12345?"
          Action: First, use `get_property_data` to fetch the description. Then, with that description in hand, use 
          `analyze_description` on the description. Since only one tool call is allowed per query, you might need to 
          guide the user step by step:
          1) On user's first request, call `get_property_data`. Make sure to show the data to the user in a pretty JSON format.
          2) On user's next request, take the returned description and call `analyze_description`.

        - Query: "What does the global accessibility report say for this property?"
          Action: You would need the outputs from both `analyze_description` and `analyze_images`. If you do not have them 
          yet, try to find them in conversation history. Even if they are not in the right format, format them accordingly.
          Otherwise instruct the user to get them first. Once both are available, call `generate_accessibility_report`.

        Stay focused and efficient in tool usage while ensuring clarity and helpfulness in responses.
        """
        self.agent = create_react_agent(model, tools=tools, state_modifier=prompt)

    def query(self, conversation: List[Dict[str, str]]) -> str:
        result = self.agent.invoke({"messages": conversation})
        last_msg = result["messages"][-1]
        return last_msg.content
    
remote_agent = reasoning_engines.ReasoningEngine.create(
    PropertyAccessibilityApp(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET),
    requirements=[
        "langchain-google-vertexai",
        "google-cloud-aiplatform[langchain,reasoningengine]",
        "cloudpickle==3.0.0",
        "pydantic==2.7.4",
        "langgraph",
        "requests"
    ],
    display_name="Property Accessibility Reasoning Engine",
    description="A reasoning engine that analyzes properties for accessibility features and drafts emails to hosts.",
    extra_packages=[],
)

print(f"Created reasoning engine: {remote_agent.resource_name}")