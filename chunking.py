from loadfile import fileload
import logging


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# UPDATED IMPORTS
from langchain_text_splitters import (
    CharacterTextSplitter,
    TokenTextSplitter,
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker


path = r"C:\Users\aditya\Desktop\pracsesh\sample.txt"


class chunk:
    def __init__(self, embed=None):   # changed so you can call chunk()
        self.embeddings = embed

    def fixed(
        self,
        content,
        separate_keywords="recursive",
        sep_value=None,
        chunk_size=500,
        chunk_overlap=50
    ):
        try:

            if sep_value is not None:
                splitter = CharacterTextSplitter(
                    separator=sep_value,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

            if separate_keywords == "character":
                splitter = CharacterTextSplitter(
                    separator="",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

            elif separate_keywords == "token":
                splitter = TokenTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

            elif separate_keywords == "paragraph":
                splitter = CharacterTextSplitter(
                    separator="\n\n",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

            elif separate_keywords == "recursive":

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

            else:
                raise ValueError(
                    f"Unknown splitter {separate_keywords}"
                )
            
            chunker=splitter.split_text(content)
            if not chunker:
                raise ValueError("Chunking failed. No chunks created.")
            
            logger.info(
            f"Created {len(chunker)} chunks"
        )
            return chunker
        
        except Exception as e:
            logger.exception(
            f"Unexpected chunking failure{e}"
        )
            return []

    def semantic_chunking(self, content):

        try:

            chunker = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90
            )

            chunks = chunker.split_text(content)
            if not chunks:
                raise ValueError("Chunking failed. No chunks created.")
            logger.info(
            f"Created {len(chunks)} semantic chunks"
        )
            return chunks
        
        except Exception as e:
            print(f"Error in semantic chunking: {e}")
            return []


if __name__ == "__main__":

    with open("sample.txt", "r", encoding="utf-8") as f:
        con = f.read()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    chunker = chunk(embeddings)

    print(chunker.semantic_chunking(con))