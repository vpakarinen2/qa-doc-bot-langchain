"""Document Q/A Bot using LangChain with optional LoRA adapter."""

import argparse
import torch

from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Any
from pydantic.v1 import Field
from peft import PeftModel
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM


class LocalLLM(LLM):
    """LangChain LLM wrapper."""
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)
    device: str = "cpu"
    trust_remote_code: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        lora_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.trust_remote_code = trust_remote_code

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            trust_remote_code=self.trust_remote_code,
        )

        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model = self.model.to(self.device)

    @property
    def _llm_type(self) -> str:
        return "qwen_local"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate concise answer and trim verbosity."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        gen_only = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(gen_only, skip_special_tokens=True).strip()

        if "Answer:" in text:
            text = text.split("Answer:", 1)[1].strip()

        for sep in ["\n\n", "\r\n\r\n"]:
            if sep in text:
                text = text.split(sep, 1)[0].strip()

        for end in [".", "!", "?"]:
            idx = text.find(end)
            if idx != -1:
                text = text[: idx + 1]
                break

        return text.strip()


def load_document(path: Path):
    """Load PDF or text file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from langchain.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(path))
    elif suffix in {".txt", ".text"}:
        from langchain.document_loaders import TextLoader

        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt.")

    documents = loader.load()
    return documents


def build_vectorstore(documents):
    """Chunk document, embed it, and build vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Document Q/A Bot using LangChain with local HF model (optional LoRA)."
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        required=False,
        default="What is this document about?",
        help="Question to ask about the document.",
    )
    parser.add_argument(
        "-d",
        "--doc",
        type=str,
        required=True,
        help="Path to the document (.pdf or .txt).",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        required=False,
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="Hugging Face base model id to load (e.g. Qwen/Qwen3-4B-Thinking-2507).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of custom remote code.",
    )
    parser.add_argument(
        "-l",
        "--lora-name",
        type=str,
        required=False,
        help=("Optional LoRA adapter to apply on top of the base model."),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    doc_path = Path(args.doc)

    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    print(f"Loading document: {doc_path}")
    documents = load_document(doc_path)

    print(f"Splitting and indexing document ({len(documents)} base document(s))...")
    vectorstore = build_vectorstore(documents)

    print(f"Loading model: {args.model_name}")

    if args.lora_path:
        print(f"Loading LoRA adapter: {args.lora_path}")

    llm = LocalLLM(
        model_name=args.model_name,
        trust_remote_code=args.trust_remote_code,
        lora_path=args.lora_path,
    )

    rag_prompt = PromptTemplate(
        template=(
            "You are a helpful assistant.\n\n"
            "Use the provided context to answer the user's question.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer in 2–3 sentences.\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )
    document_chain = create_stuff_documents_chain(llm, rag_prompt)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def add_context(inputs: dict) -> dict:
        question = inputs["input"]
        docs = retriever.invoke(question)
        return {
            "input": question,
            "context": docs,
        }

    def to_prompt_inputs(inputs: dict) -> dict:
        return {
            "context": inputs["context"],
            "question": inputs["input"],
        }

    retrieval_chain = (
        RunnableLambda(add_context)
        | RunnableLambda(to_prompt_inputs)
        | document_chain
    )

    question = args.question
    print(f"\nQuestion: {question}\n")

    result = retrieval_chain.invoke({"input": question})
    answer = result if isinstance(result, str) else result.get("answer", "")

    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()


