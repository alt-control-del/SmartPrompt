import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Streamlit app title
st.title("Effective Prompt Engineering with Llama 3.1 8B - Groq ")

# Sidebar for API key input
st.sidebar.header("API Key Input")
user_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
api_key = user_api_key

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Input prompt from the user
user_prompt = st.text_area("Enter your prompt here:")

prompt_template = """
Act as a professional prompt engineer.
Rephrase the user's input prompt according to the following guidelines. Identify missing information that might be critical for providing an accurate and comprehensive response and ask the user to specify them and additional details if needed, such as the goal they want to achieve, or if there are implicit assumptions in the input that need clarification. Ensure that the rephrased prompt is clearer, more actionable, and achieves the desired task effectively based on the following criteria:

Clarity & Specificity: Ensure the task is clear, unambiguous, and uses precise language.

Context Enhancement: Add relevant background or context if it's missing or unclear in the original input.

Expected Output: Specify the desired output format (e.g., paragraph, list, table), tone, style, or length as necessary and make sure the prompt indicates if the response should be formal, casual, persuasive, technical, or another tone.

Task Breakdown: For complex tasks, break the instructions into manageable steps for ease of understanding and execution, and provide logical sequencing if multiple parts need to be completed.

Examples for Guidance: Include relevant examples, if necessary, to guide the LLM in interpreting the prompt.

Tone & Style Alignment: If the tone or style isn’t specified, set a suitable one (e.g., formal, casual, persuasive) based on the content and intent of the task.

Constraints & Limitations: Apply any relevant constraints such as word count, time limits, vocabulary choices, or format requirements.

Use delimiters to clearly indicate distinct parts of the input.

Option to remove hallucination or reduce by mentioning, if you do not know, say “I don’t Know”.

Add where to focus – key areas to focus on should be specified by the user.

Mention the role for the LLM: Consider yourself as a professional [specific role] (e.g., lawyer, engineer, consultant) while generating the response.
User Prompt: ```{user_prompt}```

"""

# Function to call the LLAMA 3.1 model using langchain_chatgroq
def call_groq_model(api_key, user_prompt):
    try:
        # Initialize the ChatGroq LLAMA 3.1 model
        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)
        prompt = PromptTemplate(
            input_variables=["user_prompt"], template=prompt_template
        )
        # Invoke the model with the user's prompt
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke(user_prompt)

        return response if isinstance(response, str) else str(response)

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Display conversation history
if st.session_state.conversation_history:
    st.write("**Conversation History:**")
    for i, (user_input, bot_response) in enumerate(st.session_state.conversation_history):
        st.write(f"**User {i+1}:** {user_input}")
        st.write(f"**Bot {i+1}:** {bot_response}")

# Button to trigger the prompt engineering
if st.button("Generate Refined Prompt"):
    if not user_api_key:
        st.error("Please provide a valid API key.")
    elif not user_prompt:
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Refining your prompt..."):
            refined_prompt = call_groq_model(user_api_key, user_prompt)
            st.session_state.conversation_history.append((user_prompt, refined_prompt))
            st.write(f"**Refined Prompt:** {refined_prompt}")

            # Option to download the refined prompt as a text file
            st.download_button(
                label="Download Refined Prompt",
                data=refined_prompt,
                file_name="refined_prompt.txt",
                mime="text/plain"
            )

# Sidebar instructions
st.sidebar.markdown(
    """
    **Instructions**:
    1. Get your Groq API key from the [Groq Cloud](https://console.groq.com/keys).
    2. Ensure that your key has access to the llama-3.1-8b-instant model.
    3. Enter your API key in the sidebar.
    """
)

