from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,WebBaseLoader


loader =WebBaseLoader(' web url')
doc=loader.load()
# print(doc)
print(doc[0].page_content)
print(doc[0].metadata)