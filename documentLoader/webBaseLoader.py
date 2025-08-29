from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    WebBaseLoader,
)

page_url = "https://python.langchain.com/docs/tutorials/"

loader = WebBaseLoader(web_paths=[page_url])
print("loaded doc-", type(loader.lazy_load()))
# print("loaded doc length-", len(loader.lazy_load()))
print("loaded doc-", loader.load())
# print("loaded doc-", loader.lazy_load())
docs = []

for doc in loader.lazy_load():
    docs.append(doc)
print("-" * 20)
print("modified doc-", type(docs))
print("modified doc length", len(docs))
print("modified doc-", docs)
doc = docs[0]

# print(f"{doc.metadata}\n")
# print(doc.page_content[:500].strip())
