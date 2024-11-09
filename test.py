# from typing import List, Dict, Any, Optional
# from pydantic import BaseModel, Field
# from datetime import datetime
# import json

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_anthropic import ChatAnthropic




# class Appearance(BaseModel):
#     physical_description: str = Field(description="Character's physical appearance")
#     typical_attire: Optional[str] = Field(description="Common clothing or distinctive features", default=None)
#     unique_features: List[str] = Field(description="Distinctive physical characteristics", default_factory=list)

# class PersonalityTraits(BaseModel):
#     core_traits: List[str] = Field(description="Primary personality characteristics")
#     strengths: List[str] = Field(description="Character's main strengths")
#     flaws: List[str] = Field(description="Character's main flaws or weaknesses")
#     motivations: List[str] = Field(description="Primary driving forces and goals")

# class Relationship(BaseModel):
#     related_character: str = Field(description="Name of the related character")
#     relationship_type: str = Field(description="Nature of the relationship (friend, enemy, family, etc.)")
#     relationship_arc: str = Field(description="How the relationship develops through the story")
#     key_interactions: List[str] = Field(description="Important moments between characters")

# class CharacterArc(BaseModel):
#     initial_state: str = Field(description="Character's state at the beginning")
#     key_events: List[str] = Field(description="Major events that shape the character")
#     transformations: List[str] = Field(description="How the character changes")
#     final_state: str = Field(description="Character's state at the end")

# class Quote(BaseModel):
#     text: str = Field(description="The actual quote")
#     context: str = Field(description="When and where this was said")
#     significance: str = Field(description="Why this quote is important")
#     chapter_or_location: Optional[str] = Field(description="Where in the story this occurs", default=None)

# class Character(BaseModel):
#     name: str = Field(description="Character's full name")
#     aliases: List[str] = Field(description="Alternative names or titles", default_factory=list)
#     role: str = Field(description="Role in the story (protagonist, antagonist, supporting)")
#     importance_level: int = Field(
#         description="Importance scale 1-10",
#         ge=1,
#         le=10
#     )
#     first_appearance: Optional[str] = Field(description="When the character first appears", default=None)
#     appearance: Optional[Appearance] = Field(description="Physical appearance details", default=None)
#     personality: PersonalityTraits
#     relationships: List[Relationship] = Field(default_factory=list)
#     character_arc: Optional[CharacterArc] = Field(description="Character's development", default=None)
#     memorable_quotes: List[Quote] = Field(default_factory=list)
#     plot_significance: str = Field(description="Overall importance to the story")

# class ChunkAnalysis(BaseModel):
#     chunk_summary: str = Field(description="Summary of the current chunk")
#     characters: List[Character] = Field(description="Characters identified in this chunk")

# class CharacterList(BaseModel):
#     book_title: str = Field(description="Title of the analyzed book")
#     analysis_date: str = Field(default_factory=lambda: datetime.now().isoformat())
#     characters: List[Character] = Field(description="List of analyzed characters")
#     analysis_summary: str = Field(description="Overall analysis of character dynamics")

# # ============= Prompts =============

# CHUNK_ANALYSIS_TEMPLATE = """
# Previous Context Summary:
# {context_summary}

# Analyze the following text chunk and:
# 1. Identify all characters and their roles
# 2. Note character development and relationships
# 3. Capture memorable quotes and key scenes
# 4. Track character arcs and transformations

# Focus especially on:
# - New character introductions
# - Character development and changes
# - Relationship dynamics
# - Significant dialogue and actions
# - Important plot contributions

# Current known characters: {known_characters}

# Text chunk:
# {chunk_text}

# {format_instructions}
# """

# FINAL_SUMMARY_TEMPLATE = """
# Based on all the character information provided, create a comprehensive analysis of the character dynamics
# and their collective impact on the story. Consider:

# 1. Major character relationships and conflicts
# 2. How characters influence each other's development
# 3. The overall character web and its impact on the plot
# 4. Thematic elements revealed through character interactions

# Character information:
# {character_info}

# Provide a detailed analysis in a clear, narrative format.
# """


# class BookCharacterAnalyzer:
#     def __init__(
#         self,
#         model: str = "claude-3-5-sonnet-latest",
#         chunk_size: int = 50000,
#         chunk_overlap: int = 500,
#         temperature: float = 0.7
#     ):
#         self.llm = ChatAnthropic(
#             model=model,
#             temperature=temperature

#         )

#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             length_function=len,
#             separators=["\n\n", "\n", ". ", " ", ""]
#         )

#         self.chunk_parser = PydanticOutputParser(pydantic_object=ChunkAnalysis)
#         self.summary_parser = PydanticOutputParser(pydantic_object=str)

#         self.context = {
#             "summary": "",
#             "characters": {},
#             "current_chunk": 0,
#             "total_chunks": 0
#         }


#         self.chunk_prompt = PromptTemplate(
#             template=CHUNK_ANALYSIS_TEMPLATE,
#             input_variables=["context_summary", "known_characters", "chunk_text"],
#             partial_variables={"format_instructions": self.chunk_parser.get_format_instructions()}
#         )

#         self.summary_prompt = PromptTemplate(
#             template=FINAL_SUMMARY_TEMPLATE,
#             input_variables=["character_info"]
#         )

#     def merge_character_data(self, existing: Dict, new: Dict) -> Dict:
#         """Smart merge of character information from different chunks."""
#         if not existing:
#             return new

#         merged = existing.copy()


#         for field in ['role', 'plot_significance']:
#             if len(new.get(field, '')) > len(existing[field]):
#                 merged[field] = new[field]


#         for field in ['aliases', 'memorable_quotes']:
#             existing_items = set(str(item) for item in existing.get(field, []))
#             new_items = set(str(item) for item in new.get(field, []))
#             merged[field] = list(existing_items.union(new_items))


#         merged['importance_level'] = max(
#             existing['importance_level'],
#             new.get('importance_level', 1)
#         )


#         existing_relationships = {r['related_character']: r for r in existing.get('relationships', [])}
#         for new_rel in new.get('relationships', []):
#             if new_rel['related_character'] not in existing_relationships:
#                 existing_relationships[new_rel['related_character']] = new_rel
#             else:

#                 old_rel = existing_relationships[new_rel['related_character']]
#                 if len(new_rel['relationship_arc']) > len(old_rel['relationship_arc']):
#                     old_rel['relationship_arc'] = new_rel['relationship_arc']
#                 old_rel['key_interactions'].extend(new_rel['key_interactions'])

#         merged['relationships'] = list(existing_relationships.values())


#         if new.get('character_arc'):
#             if not existing.get('character_arc'):
#                 merged['character_arc'] = new['character_arc']
#             else:
#                 existing_arc = existing['character_arc']
#                 new_arc = new['character_arc']
#                 existing_arc['transformations'].extend(new_arc['transformations'])
#                 existing_arc['key_events'].extend(new_arc['key_events'])
#                 if new_arc['final_state']:
#                     existing_arc['final_state'] = new_arc['final_state']

#         return merged

#     def create_analysis_chain(self):
#         """Create the LCEL chain for analyzing chunks."""

#         chunk_analysis = (
#             RunnablePassthrough() |
#             self.chunk_prompt |
#             self.llm |
#             self.chunk_parser
#         )


#         summary_chain = (
#             self.summary_prompt |
#             self.llm |
#             self.summary_parser
#         )

#         return chunk_analysis, summary_chain

#     def process_book(self, file_path: str, book_title: str) -> CharacterList:
#         """Process the entire book using LangChain components."""

#         loader = PyPDFLoader(file_path)
#         document = loader.load()
#         document = "\n\n".join(doc.page_content for doc in document)


#         words = document.split()
#         chunks = [' '.join(words[i:i + 50000]) for i in range(0, len(words), 50000)]
#         self.context['total_chunks'] = len(chunks)


#         chunk_chain, summary_chain = self.create_analysis_chain()


#         for i, chunk in enumerate(chunks, 1):
#             self.context['current_chunk'] = i
#             print(f"Processing chunk {i}/{len(chunks)}")


#             chunk_input = {
#                 "context_summary": self.context['summary'],
#                 "known_characters": ", ".join(self.context['characters'].keys()),
#                 "chunk_text": chunk
#             }


#             try:
#                 result = chunk_chain.invoke(chunk_input)


#                 self.context['summary'] += f"\nChunk {i}: {result.chunk_summary}"


#                 for char_data in result.characters:
#                     char_name = char_data.name
#                     if char_name in self.context['characters']:
#                         self.context['characters'][char_name] = self.merge_character_data(
#                             self.context['characters'][char_name],
#                             char_data.dict()
#                         )
#                     else:
#                         self.context['characters'][char_name] = char_data.dict()

#             except Exception as e:
#                 print(f"Error processing chunk {i}: {e}")
#                 continue


#         final_summary = summary_chain.invoke({
#             "character_info": json.dumps(list(self.context['characters'].values()), indent=2)
#         })


#         return CharacterList(
#             book_title=book_title,
#             characters=list(self.context['characters'].values()),
#             analysis_summary=final_summary
#         )



# def analyze_book(file_path: str, book_title: str, model: str = "claude-3-5-sonnet-latest") -> CharacterList:
#     """Main function to analyze a book's characters using LangChain."""
#     analyzer = BookCharacterAnalyzer(model=model)
#     return analyzer.process_book(file_path, book_title)


# from dotenv import load_dotenv
# load_dotenv()

# file_path = "percy_jackson.pdf"
# book_title = "percy jackson and the olympians"

# character_analysis = analyze_book(file_path, book_title)

# # Save the analysis
# with open(f"{book_title}_character_analysis.json", 'w') as f:
#     f.write(character_analysis.model_dump_json(indent=2))



# from typing import List
# from pydantic import BaseModel, Field
# import json
# from langchain_anthropic import ChatAnthropic
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_community.document_loaders import PyPDFLoader

# class BasicCharacter(BaseModel):
#     name: str = Field(description="The character's full name")
#     role: str = Field(description="The character's primary role (protagonist, antagonist, supporting)")
#     plot_importance: str = Field(
#         description="Detailed explanation of why this character is important to the plot"
#     )
#     importance_level: int = Field(
#         description="Importance level 1-10, based on their impact on the main plot",
#         ge=1,
#         le=10
#     )
#     key_relationships: List[str] = Field(description="List of important relationships")

# class CharacterDescription(BaseModel):
#     name: str
#     detailed_description: str = Field(
#         description="Comprehensive physical and personality description"
#     )
#     llm_persona_prompt: str = Field(
#         description="A prompt section for emulating this character's personality"
#     )

# class CharacterDepth(BaseModel):
#     name: str
#     character_arc: str = Field(
#         description="Detailed progression of character development"
#     )
#     personality_traits: List[str] = Field(
#         description="List of personality traits with examples"
#     )
#     memorable_quotes: List[str] = Field(
#         description="Key quotes that demonstrate character development",
#         default=[]
#     )

# class CharacterAnalysis(BaseModel):
#     basic_info: BasicCharacter
#     description: CharacterDescription
#     depth: CharacterDepth

# class CharacterAnalysisList(BaseModel):
#     characters: List[CharacterAnalysis] = Field(description="Complete character analyses")

# class BookAnalyzer:
#     def __init__(self, model: str = "claude-3-sonnet-20240229"):
#         self.llm = ChatAnthropic(
#             model=model,
#             temperature=0.7
#         )
#         self.parser = PydanticOutputParser(pydantic_object=CharacterAnalysisList)

#         self.analysis_template = """
#         Analyze all significant characters in the following text. For each character provide:

#         1. Basic Information:
#            - Full name
#            - Role in the story
#            - Plot importance
#            - Importance level (1-10)
#            - Key relationships

#         2. Detailed Description:
#            - Physical appearance
#            - Personality traits
#            - How they speak and behave

#         3. Character Depth:
#            - Character arc and development
#            - Personality traits with specific examples
#            - Memorable quotes that show their character

#         Text to analyze:
#         {text}

#         {format_instructions}
#         """

#         self.prompt = PromptTemplate(
#             template=self.analysis_template,
#             input_variables=["text"],
#             partial_variables={"format_instructions": self.parser.get_format_instructions()}
#         )

#     def analyze_text(self, text: str) -> CharacterAnalysisList:
#         """Analyze text and extract character information."""
#         formatted_prompt = self.prompt.format(text=text)
#         response = self.llm.invoke(formatted_prompt)
#         return self.parser.parse(response.content)

#     def process_book(self, file_path: str) -> None:
#         """Process a PDF book and save character analysis."""
#         # Load PDF
#         loader = PyPDFLoader(file_path)
#         pages = loader.load()
#         text = "\n\n".join(page.page_content for page in pages)

#         # Analyze text
#         analysis = self.analyze_text(text)

#         # Save results
#         output_file = f"{file_path.split('.')[0]}_character_analysis.json"
#         with open(output_file, 'w') as f:
#             json.dump(analysis.model_dump(), f, indent=2)

#         print(f"Analysis saved to {output_file}")

# def analyze_book(file_path: str, model: str = "claude-3-5-sonnet-latest"):
#     """Main function to analyze a book's characters."""
#     analyzer = BookAnalyzer(model=model)
#     analyzer.process_book(file_path)

# # Usage example:
# if __name__ == "__main__":
#     from dotenv import load_dotenv
#     load_dotenv()

#     file_path = "percy_jackson.pdf"
#     analyze_book(file_path)




from typing import List, Dict
from pydantic import BaseModel, Field
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

class BasicCharacter(BaseModel):
    name: str = Field(description="The character's full name")
    role: str = Field(description="The character's primary role (protagonist, antagonist, supporting)")
    plot_importance: str = Field(
        description="Detailed explanation of why this character is important to the plot"
    )
    importance_level: int = Field(
        description="Importance level 1-10, based on their impact on the main plot",
        ge=1,
        le=10
    )
    key_relationships: List[str] = Field(description="List of important relationships")

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

class CharacterAnalysis(BaseModel):
    basic_info: BasicCharacter
    description: CharacterDescription
    depth: CharacterDepth

class ChunkAnalysis(BaseModel):
    characters: List[CharacterAnalysis] = Field(description="Characters identified in this chunk")

class BookAnalyzer:
    def __init__(self, model: str = "claude-3-sonnet-20240229"):
        self.llm = ChatAnthropic(
            model=model,
            temperature=0.7
        )
        self.parser = PydanticOutputParser(pydantic_object=ChunkAnalysis)

        self.chunk_template = """
        Analyze the following chunk of text and identify all characters present. For each character provide:

        1. Basic Information:
           - Full name
           - Role in the story
           - Plot importance
           - Importance level (1-10)
           - Key relationships

        2. Detailed Description:
           - Physical appearance
           - Personality traits
           - How they speak and behave

        3. Character Depth:
           - Character arc and development
           - Personality traits with specific examples
           - Memorable quotes that show their character

        Previously identified characters: {known_characters}

        Text chunk to analyze:
        {chunk_text}

        {format_instructions}
        """

        self.prompt = PromptTemplate(
            template=self.chunk_template,
            input_variables=["known_characters", "chunk_text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        self.characters = {}

    def merge_character_data(self, existing: Dict, new: CharacterAnalysis) -> Dict:
        """Smart merge of character information from different chunks."""
        if not existing:
            return new.model_dump()

        merged = existing.copy()
        new_dict = new.model_dump()


        basic_info = merged['basic_info']
        new_basic = new_dict['basic_info']


        if len(new_basic['plot_importance']) > len(basic_info['plot_importance']):
            basic_info['plot_importance'] = new_basic['plot_importance']


        basic_info['importance_level'] = max(
            basic_info['importance_level'],
            new_basic['importance_level']
        )


        basic_info['key_relationships'] = list(set(
            basic_info['key_relationships'] + new_basic['key_relationships']
        ))


        if len(new_dict['description']['detailed_description']) > len(merged['description']['detailed_description']):
            merged['description'] = new_dict['description']


        depth = merged['depth']
        new_depth = new_dict['depth']


        if len(new_depth['character_arc']) > len(depth['character_arc']):
            depth['character_arc'] = new_depth['character_arc']


        depth['personality_traits'] = list(set(depth['personality_traits'] + new_depth['personality_traits']))
        depth['memorable_quotes'] = list(set(depth['memorable_quotes'] + new_depth['memorable_quotes']))

        return merged

    def process_chunk(self, chunk: str) -> None:
        """Process a single chunk of text."""
        known_chars = ", ".join(self.characters.keys()) if self.characters else "None yet"


        formatted_prompt = self.prompt.format(
            known_characters=known_chars,
            chunk_text=chunk
        )
        response = self.llm.invoke(formatted_prompt)


        chunk_analysis = self.parser.parse(response.content)


        for char in chunk_analysis.characters:
            char_name = char.basic_info.name
            if char_name in self.characters:
                self.characters[char_name] = self.merge_character_data(
                    self.characters[char_name],
                    char
                )
            else:
                self.characters[char_name] = char.model_dump()

    def process_book(self, file_path: str) -> None:
        """Process a PDF book in chunks and save character analysis."""

        loader = PyPDFLoader(file_path)
        document = loader.load()
        text = "\n\n".join(doc.page_content for doc in document)


        words = text.split()
        chunks = [' '.join(words[i:i + 50000]) for i in range(0, len(words), 50000)]


        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}")
            try:
                self.process_chunk(chunk)
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue


        output_file = f"{file_path.split('.')[0]}_character_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(
                {"characters": list(self.characters.values())},
                f,
                indent=2
            )

        print(f"Analysis saved to {output_file}")

def analyze_book(file_path: str, model: str = "claude-3-5-sonnet-latest"):
    """Main function to analyze a book's characters."""
    analyzer = BookAnalyzer(model=model)
    analyzer.process_book(file_path)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    file_path = "percy_jackson.pdf"
    analyze_book(file_path)