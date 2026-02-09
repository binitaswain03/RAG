from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import config

class CorrectionModule:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=config.OPENAI_API_KEY, temperature=0.7)
        self.prompt = PromptTemplate(
            template="""The user asked: "{original_query}"
The retrieval system provided context, but the generated answer was verified as low confidence or hallucinated (Reason: {reasoning}).
Please reformulate the query to be more specific or better targeted to retrieve relevant information.
Return ONLY the reformulated query.
""",
            input_variables=["original_query", "reasoning"]
        )

    def reformulate_query(self, original_query, reasoning):
        """Reformulates a query based on previous failure reasoning."""
        chain = self.prompt | self.llm
        response = chain.invoke({"original_query": original_query, "reasoning": reasoning})
        return response.content.strip()

class PipelineOrchestrator:
    def __init__(self):
        from modules.retrieval import Retriever
        from modules.generation import AnswerGenerator
        from modules.verification import Verifier
        from utils.logger import setup_logger
        
        self.retriever = Retriever()
        self.generator = AnswerGenerator()
        self.verifier = Verifier()
        self.corrector = CorrectionModule()
        self.logger = setup_logger("pipeline_orchestrator")

    def run_pipeline(self, query):
        """Runs the self-correcting RAG pipeline."""
        original_query = query
        current_query = query
        attempts = 0
        logs = []

        while attempts <= config.MAX_RETRIES:
            attempts += 1
            self.logger.info(f"Attempt {attempts}: Query='{current_query}'")
            logs.append(f"Attempt {attempts}: Query='{current_query}'")

            # 1. Retrieval
            chunks = self.retriever.get_relevant_chunks(current_query)
            if not chunks:
                msg = "No relevant context found."
                self.logger.warning(msg)
                logs.append(msg)
                if attempts < config.MAX_RETRIES:
                    current_query = self.corrector.reformulate_query(current_query, "No context found")
                    continue
                return {"answer": "I could not find any relevant information to answer your question.", "confidence": 0.0, "logs": logs, "chunks": []}

            # 2. Generation
            answer, used_chunks = self.generator.generate_answer(current_query, chunks)
            self.logger.info("Answer generated.")

            # 3. Verification
            verification = self.verifier.verify_answer(current_query, answer, used_chunks)
            self.logger.info(f"Verification Score: {verification.score}. Reasoning: {verification.reasoning}")
            logs.append(f"Verification: Score={verification.score}. Reason={verification.reasoning}")

            if verification.score >= config.CONFIDENCE_THRESHOLD:
                return {
                    "answer": answer,
                    "confidence": verification.score,
                    "reasoning": verification.reasoning,
                    "logs": logs,
                    "chunks": used_chunks
                }
            
            # 4. Correction Loop
            if attempts <= config.MAX_RETRIES:
                self.logger.info("Score below threshold. Reformulating...")
                logs.append("Score below threshold. Reformulating query...")
                current_query = self.corrector.reformulate_query(current_query, verification.reasoning)
            else:
                self.logger.warning("Max retries reached.")
                logs.append("Max retries reached. Returning best effort.")
                # Return the last answer but mark it as low confidence
                return {
                    "answer": f"**Warning: Low Confidence ({verification.score})**\n{answer}",
                    "confidence": verification.score,
                    "reasoning": verification.reasoning,
                    "logs": logs,
                    "chunks": used_chunks
                }

