INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="gpt-4o-mini",
        context_fields=None,
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model
        self.context_fields = context_fields

    def search(self, query, num_results=5, boost_dict=None, filter_dict=None):
        if boost_dict is None:
            boost_dict = {"question": 3.0, "section": 0.5} if not self.context_fields else {}
        
        if filter_dict is None:
            filter_dict = {"course": self.course} if not self.context_fields else {}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            if self.context_fields:
                formatted = self._format_generic_doc(doc)
                if formatted:
                    lines.append(formatted)
                    lines.append("")
                continue

            if all(key in doc for key in ("section", "question", "answer")):
                lines.append(str(doc.get("section", "")).strip())
                lines.append("Q: " + str(doc.get("question", "")).strip())
                lines.append("A: " + str(doc.get("answer", "")).strip())
                lines.append("")
                continue


        return "\n".join(lines).strip()

    def _format_generic_doc(self, doc):
        if self.context_fields:
            parts = []
            for field in self.context_fields:
                if field in doc and doc[field] is not None:
                    parts.append(f"{field.title()}: {doc[field]}")
            if parts:
                return "\n".join(parts)

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return {
            "output_text": response.output_text,
            "input_tokens": response.usage.input_tokens,
        }

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        llm_result = self.llm(prompt)
        return llm_result