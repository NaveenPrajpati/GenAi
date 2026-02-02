from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(
    " kjhlkh lj oh kjkb kbj kbkbkbkb klb kb lkblkblkblkb lk bk"
)


docs = [[1, 2, 3], [4, 5], [6, 7, 8]]

doc2 = [item for doc in docs for item in doc]
print(doc2)
