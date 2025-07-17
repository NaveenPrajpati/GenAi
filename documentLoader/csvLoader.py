from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,CSVLoader


loader =CSVLoader(file_path='path of csv file')
doc=loader.load()
# print(doc)
print(doc[0].page_content)
print(doc[0].metadata)