from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import config

class VerificationResult(BaseModel):
    score: float = Field(description="A score between 0.0 and 1.0 indicating factual grounding.")
    reasoning: str = Field(description="Explanation of the score and any hallucinations found.")

from utils.llm_utils import retry_with_backoff

class Verifier:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=config.OPENAI_API_KEY, temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=VerificationResult)
        self.prompt = PromptTemplate(
            template="""You are a strict fact-checker. Verify if the generated answer is fully supported by the provided context.
Output a score between 0.0 (not supported) and 1.0 (fully supported) and provide reasoning.

Question: {question}
Answer: {answer}
Context:
{context}

{format_instructions}
""",
            input_variables=["question", "answer", "context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def verify_answer(self, query, answer, context_chunks):
        """Verifies the generated answer against the context."""
        context_text = "\n\n".join([doc.page_content for doc in context_chunks])
        chain = self.prompt | self.llm | self.parser
        try:
            result = chain.invoke({"question": query, "answer": answer, "context": context_text})
            return result
        except Exception as e:
            # Fallback in case of parsing error
            return VerificationResult(score=0.0, reasoning=f"Verification failed: {str(e)}")
