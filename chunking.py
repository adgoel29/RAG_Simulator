from loadfile import fileload
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_experimental.text_splitter import SemanticChunker



path=r"C:\Users\aditya\Desktop\pracsesh\sample.txt"

class chunk:
    def __init__(self,embed):
        self.embeddings=embed
    
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

        elif separate_keywords == "character":
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

    def semantic_chunking(self,content):

        # embeddings=self.embeddings
        chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90   # higher = fewer, larger chunks
        )

        chunks=chunker.split_text(content)

        return chunks


if __name__=="__main__":
     
    with open("sample.txt",'r',encoding="utf-8") as f:
            con=f.read()
    chunker=chunk()



    print(chunker.semantic_chunking(con))







