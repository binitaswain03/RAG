from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser

def broken_function():
    print("This file contains deprecated imports.")
    template = PromptTemplate(template="Hello {name}!", input_variables=["name"])
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    parser = PydanticOutputParser(pydantic_object=None)
    return template, splitter, parser

if __name__ == "__main__":
    broken_function()
