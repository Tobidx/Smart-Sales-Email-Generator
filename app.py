import os
import torch
import gradio as gr
from langchain import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline

# Initialize sentiment analyzer
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="finiteautomata/bertweet-base-sentiment-analysis",
    device=0 if torch.cuda.is_available() else -1
)

# Initialize LLM
llm = HuggingFaceHub(
    repo_id="deepseek-ai/deepseek-coder-33b-instruct",
    model_kwargs={"temperature": 0.7},
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

email_template = PromptTemplate(
    input_variables=["previous_interaction", "situation_type", "tone", "urgency"],
    template="""Based on these details, generate a professional follow-up email:

Previous Interaction: {previous_interaction}
Situation Type: {situation_type}
Tone: {tone}
Urgency Level: {urgency}

Generate a personalized email that:
1. Maintains {tone} tone
2. Addresses the specific situation
3. Provides clear next steps
4. Is appropriate for {urgency} urgency level
"""
)
 #Create LangChain
email_chain = LLMChain(llm=llm, prompt=email_template)

def analyze_sentiment(text):
    try:
        result = sentiment_analyzer(text)[0]
        sentiment_to_tone = {
            'POS': 'Friendly',
            'NEU': 'Professional',
            'NEG': 'Apologetic'
        }
        return sentiment_to_tone.get(result['label'], 'Professional')
    except Exception as e:
        return 'Professional'

def generate_followup_email(previous_interaction, situation_type, tone, urgency):
    try:
        if not tone:
            tone = analyze_sentiment(previous_interaction)
        return email_chain.run({
            "previous_interaction": previous_interaction,
            "situation_type": situation_type,
            "tone": tone,
            "urgency": urgency
        })
    except Exception as e:
        return f"Error generating email: {str(e)}"

demo = gr.Interface(
    fn=generate_followup_email,
    inputs=[
        gr.Textbox(label="Previous Interaction", lines=5,
                  placeholder="Describe the previous interaction with the customer..."),
        gr.Dropdown(label="Situation Type",
                   choices=["Complaint Resolution", "Service Issue",
                           "Payment Dispute", "Product Query", "General Follow-up"]),
        gr.Dropdown(label="Tone (Optional - will be automatically detected if not specified)",
                   choices=["", "Professional", "Apologetic", "Friendly", "Formal", "Empathetic"]),
        gr.Dropdown(label="Urgency", choices=["High", "Medium", "Low"])
    ],
    outputs=gr.Textbox(label="Generated Email"),
    title="Smart Sales Email Generator",
    description="Generate personalized follow-up emails based on previous interactions",
    examples=[
        ["Customer complained about slow website loading times and threatened to cancel subscription",
         "Complaint Resolution", "Apologetic", "High"],
        ["Client requested information about premium features and pricing",
         "Product Query", "Professional", "Medium"]
    ]
)

if __name__ == "__main__":
    demo.launch()
