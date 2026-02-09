from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import config

class AnswerGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=config.OPENAI_API_KEY, temperature=0)
        self.prompt = PromptTemplate(
            template="""You are a helpful assistant. Use the following pieces of retrieved context to answer the question.
If the answer is not in the context, just say that you don't know. Keep the answer concise.

Question: {question}

Context:
{context}

Answer:""",
            input_variables=["question", "context"]
        )

    def generate_answer(self, query, context_chunks):
        """Generates an answer based on context."""
        context_text = "\n\n".join([doc.page_content for doc in context_chunks])
        chain = self.prompt | self.llm
        response = chain.invoke({"question": query, "context": context_text})
        return response.content, context_chunks
