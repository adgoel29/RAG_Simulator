from loadfile import fileload

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

        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        return splitter.split_text(content)

    def semantic_chunking(self, content):

        chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90
        )

        chunks = chunker.split_text(content)

        return chunks


if __name__ == "__main__":

    with open("sample.txt", "r", encoding="utf-8") as f:
        con = f.read()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    chunker = chunk(embeddings)

    print(chunker.semantic_chunking(con))