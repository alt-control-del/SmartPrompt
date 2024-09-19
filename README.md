# 🌟 SmartPrompt
This is a Streamlit application designed to assist users in refining their input prompts using Google's Gemini 1.5 pro model. The app takes a user-provided prompt, enhances its clarity and effectiveness based on prompt engineering best practices, and outputs a refined version. The app can also provide the refined prompt for download.

## 🛠️ Installation and Working
1. Download or clone the Repository
2. To install the necessary dependencies, run:
```
pip install streamlit langchain_google_genai langchain_core
```
3. Run the Streamlit app:
```
streamlit run app.py
```


## ✨ Features
1. API Key Input: Securely input your Google API key through the sidebar.
2. Prompt Refinement: Enhance user prompts based on criteria such as clarity, context, tone, and specificity.
3. Conversation History: Track and display conversation history between user prompts and the model's refined output.
4. Downloadable Output: Download the refined prompt as a .txt file.
5. Error Handling: Friendly error messages when inputs are missing or invalid API keys are provided.

<br> 
Enjoy crafting effective prompts! 💡
