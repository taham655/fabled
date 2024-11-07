import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import json
from typing import List, Optional


load_dotenv()


os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']



os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY']=st.secrets['LANGCHAIN_API_KEY']
os.environ['LANGCHAIN_PROJECT']=st.secrets['LANGCHAIN_PROJECT']




# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_character' not in st.session_state:
    st.session_state.current_character = None
if 'character_prompt' not in st.session_state:
    st.session_state.character_prompt = None
if 'model' not in st.session_state:
    st.session_state.model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.7
    )
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False


class BasicCharacter(BaseModel):
    name: str = Field(description="The character's full name")
    role: str = Field(description="The character's primary role (protagonist, antagonist, supporting)")
    plot_importance: str = Field(
        description="Detailed explanation of why this character is important to the plot"
    )
    importance_level: int = Field(
        description="Importance level 1-10, based on their impact on the main plot"
    )
    key_relationships: List[str] = Field(description="List of important relationships")

class BasicCharacterList(BaseModel):
    characters: List[BasicCharacter] = Field(description="List of identified characters")

class CharacterDescription(BaseModel):
    name: str
    detailed_description: str = Field(
        description="Comprehensive physical and personality description"
    )
    llm_persona_prompt: str = Field(
        description="A prompt section for emulating this character's personality"
    )

class CharacterDepth(BaseModel):
    name: str
    character_arc: str = Field(
        description="Detailed progression of character development"
    )
    personality_traits: List[str] = Field(
        description="List of personality traits with examples"
    )
    memorable_quotes: List[str] = Field(
        description="Key quotes that demonstrate character development",
        default=[]
    )

class CharacterDescriptionList(BaseModel):
    characters: List[CharacterDescription] = Field(description="List of character descriptions")

class CharacterDepthList(BaseModel):
    characters: List[CharacterDepth] = Field(description="List of character development analyses")

# Initialize prompts
first_pass_template = """
You are a literary character identifier and plot analyst. Identify all significant characters and their importance to the plot structure.

For each character, analyze:
1. Their role in the story
2. Their specific impact on major plot points
3. How central they are to the main narrative
4. Their key relationships that drive the story forward

{book_text}

{format_instructions}
"""

second_pass_template = """
For each character, provide a detailed description of their physical appearance, mannerisms, and personality. Be very specific in your analysis. Find the exact personality like kindness, anger, witty and much more. If you are aware of the character use that knowledge to provide a detailed description.
Also create a specific prompt section that could guide an LLM in portraying this character's personality.

Characters to analyze:
{character_list}

{format_instructions}
"""

third_pass_template = """
For each character, analyze their complete journey through the story, focusing on:
1. Character development arc
2. Personality traits with evidence
3. Memorable quotes that show their growth (if none available, provide an empty list)

Characters to analyze:
{character_list}

Note: For characters without memorable quotes, please use an empty list ([]) rather than indicating "No quotes available".

{format_instructions}
"""

# Initialize parsers
basic_parser = PydanticOutputParser(pydantic_object=BasicCharacterList)
description_parser = PydanticOutputParser(pydantic_object=CharacterDescriptionList)
depth_parser = PydanticOutputParser(pydantic_object=CharacterDepthList)

# Create prompts
first_pass_prompt = PromptTemplate(
    template=first_pass_template,
    input_variables=["book_text"],
    partial_variables={"format_instructions": basic_parser.get_format_instructions()}
)

second_pass_prompt = PromptTemplate(
    template=second_pass_template,
    input_variables=["character_list"],
    partial_variables={"format_instructions": description_parser.get_format_instructions()}
)

third_pass_prompt = PromptTemplate(
    template=third_pass_template,
    input_variables=["character_list"],
    partial_variables={"format_instructions": depth_parser.get_format_instructions()}
)

def analyze_characters(book_text: str):
    """Analyze characters in a given text using a three-pass system."""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

    try:
        with st.spinner("Finding all the characters in the story... This may take a few minutes... 📚✨"):
            first_pass = model.invoke(first_pass_prompt.format(book_text=book_text))
            basic_characters = basic_parser.parse(first_pass.content)
            st.success("Found some interesting characters! 🎭")

        character_list = [{"name": char.name, "role": char.role}
                         for char in basic_characters.characters]

        with st.spinner("Establishing a portal... 🤝💭"):
            second_pass = model.invoke(second_pass_prompt.format(
                character_list=str(character_list)))
            character_descriptions = description_parser.parse(second_pass.content)
            st.success("The characters are taking shape! ✨")

        with st.spinner("bring the characters over... 📖💫"):
            third_pass = model.invoke(third_pass_prompt.format(
                character_list=str(character_list)))
            try:
                character_depth = depth_parser.parse(third_pass.content)
            except Exception as e:
                st.error("Some character memories are a bit fuzzy... 🌫️")
                character_depth = CharacterDepthList(characters=[])
            st.success("Everyone's ready for a chat! 🎉")

        result = {
            "basic_info": [
                {
                    "name": char.name,
                    "role": char.role,
                    "plot_importance": char.plot_importance,
                    "importance_level": char.importance_level,
                    "key_relationships": char.key_relationships
                }
                for char in basic_characters.characters
            ],
            "descriptions": [
                {
                    "name": char.name,
                    "detailed_description": char.detailed_description,
                    "llm_persona_prompt": char.llm_persona_prompt
                }
                for char in character_descriptions.characters
            ],
            "character_depth": [
                {
                    "name": char.name,
                    "character_arc": char.character_arc,
                    "personality_traits": char.personality_traits,
                    "memorable_quotes": char.memorable_quotes
                }
                for char in character_depth.characters
            ]
        }

        # Save to JSON file
        with open('character_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        return result

    except Exception as e:
        st.error(f"Error during character analysis: {str(e)}")
        raise

def load_character_data():
    """Load character data from the JSON file"""
    try:
        with open('character_analysis.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_character_list(character_data):
    """Extract list of available characters"""
    return [f"{char['name']} - {char['role']}" for char in character_data['basic_info']]

def get_character_details(character_data, selected_name):
    """Get all relevant details for the selected character"""
    basic_info = next((char for char in character_data['basic_info']
                      if char['name'] == selected_name), None)
    description = next((char for char in character_data['descriptions']
                       if char['name'] == selected_name), None)
    depth = next((char for char in character_data['character_depth']
                 if char['name'] == selected_name), None)

    return {
        'basic_info': basic_info,
        'description': description,
        'depth': depth
    }

def create_character_prompt(character_details):
    """Create a comprehensive prompt for the character"""
    basic_info = character_details['basic_info']
    description = character_details['description']
    depth = character_details['depth']

    return f"""You are roleplaying as {basic_info['name']}. You should respond to messages as this character would respond, not as an AI assistant or chatbot. Never break character or acknowledge that you are an AI.

Background Information:
{description['detailed_description']}

Your Character Arc:
{depth['character_arc']}

Your Personality Traits:
{', '.join(depth['personality_traits'])}

Your Key Relationships:
{', '.join(basic_info['key_relationships'])}

Specific Guidance for Portraying this Character:
{description['llm_persona_prompt']}

Important Roleplaying Rules:
1. Always respond in first person as {basic_info['name']}
2. Never acknowledge that you are an AI or that this is a simulation
3. Never mention the book or story directly - you are living these events
4. Keep your responses natural and conversational
5. You can only reference events and knowledge that {basic_info['name']} would know
6. Stay true to your character's personality, knowledge level, and way of speaking
7. Respond with the emotional depth appropriate to your character
8. You can ask questions or make statements based on your character's perspective
9. You can use your own knowledge if you know the character well
10. Adapt their speech patterns, vocabulary, and tone to match the character
11. You need to have the personality and knowledge of the character. Adopt their persona fully. Adopt their demeanor, speech patterns, and knowledge level.

Remember: You ARE {basic_info['name']} in this conversation. Respond exactly as this character would, with their personality, knowledge, and manner of speaking."""

def initialize_chat(selected_character_full):
    """Initialize or reset chat with a new character"""
    selected_name = selected_character_full.split(" - ")[0]
    character_data = load_character_data()
    character_details = get_character_details(character_data, selected_name)
    st.session_state.character_prompt = create_character_prompt(character_details)
    st.session_state.current_character = selected_name
    st.session_state.messages = []

def get_chatbot_response(prompt):
    """Get response from the chatbot"""

    messages = [{"role": "system", "content": st.session_state.character_prompt}]


    for msg in st.session_state.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})


    messages.append({"role": "user", "content": prompt})

    try:
        response = st.session_state.model.invoke(messages)
        return response.content
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        return None

def main():
    st.title("Talk to Literary Characters 📚🗣️")


    if not st.session_state.analysis_complete:
        st.subheader("Upload PDF for fun to start :)")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:

            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())

            if st.button("Let's Go!"):

                loader = PyPDFLoader("temp.pdf")
                document = loader.load()
                book_text = "\n\n".join(doc.page_content for doc in document)


                analyze_characters(book_text)
                st.session_state.analysis_complete = True
                st.success("Character analysis complete! You can now start chatting with the characters.")

                if os.path.exists("temp.pdf"):
                  os.remove("temp.pdf")
                st.rerun()

    else:

        character_data = load_character_data()
        if not character_data:
            st.error("Character analysis file not found. Please restart the analysis.")
            st.session_state.analysis_complete = False
            return


        st.sidebar.title("Character Selection")
        characters = get_character_list(character_data)
        selected_character = st.sidebar.selectbox(
            "Choose a character to chat with:",
            characters
        )


        if selected_character:
            character_name = selected_character.split(" - ")[0]
            character_details = get_character_details(character_data, character_name)

            with st.sidebar.expander("Character Information"):
                st.write(f"**Role:** {character_details['basic_info']['role']}")
                st.write(f"**Importance:** {character_details['basic_info']['plot_importance']}")
                st.write("**Key Relationships:**")
                for relation in character_details['basic_info']['key_relationships']:
                    st.write(f"- {relation}")


        if st.sidebar.button("Start New Chat"):
            initialize_chat(selected_character)
            st.rerun()


        if st.session_state.current_character:
            st.write(f"Chatting with: **{st.session_state.current_character}**")


            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])


            if prompt := st.chat_input("Type your message here..."):

                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)


                response = get_chatbot_response(prompt)
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    with st.chat_message("assistant"):
                        st.write(response)


                st.rerun()
        else:
            st.info("👈 Please select a character and click 'Start New Chat' to begin the conversation.")


        if st.sidebar.button("Restart Analysis"):
            st.session_state.analysis_complete = False
            st.session_state.messages = []
            st.session_state.current_character = None
            st.session_state.character_prompt = None
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")
            if os.path.exists("character_analysis.json"):
                os.remove("character_analysis.json")
            st.rerun()

if __name__ == "__main__":
    main()
