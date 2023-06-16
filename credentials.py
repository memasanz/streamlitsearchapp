AZURE_OPENAI_ENDPOINT = "https://.openai.azure.com/"
AZURE_OPENAI_KEY = "14821c9b"


TEXT_SEARCH_DOC_EMBEDDING_ENGINE = "text-search-curie-doc-001"
TEXT_SEARCH_QUERY_EMBEDDING_ENGINE = "text-search-curie-query-001"
TEXT_DAVINCI = "text-davinci-003"
GTPTurbo = "gpt-35-turbo"

COG_SEARCH_RESOURCE = "mmx-ch"
COG_SEARCH_KEY = "FOo5exG5D3bKLKAzSeBQEfW8"
COG_SEARCH_INDEX = "ithelpdeskindex" #"indexxtg3" #"ithelpdeskindex" #"indexxtg3" #"indexxtg3" #"ithelpdeskindex" #"indexxtg3"

APP_FUNCTION_URL = "https://cog-search-functionyyN2MU9rvxFxwAzFu3L8U3w=="
STORAGE_CONNECTION_STRING = "DefaultEnN3HiadPHSg2XLaKKdK2fix=core.windows.net"
STORAGE_ACCOUNT = "mmxcorage"
XTGSTORAGE_CONTAINER = "xg"
ITHELPDESK_CONTAINER = "ithelpdesk"
STORAGE_KEY = "GJpR3WKdK2Ie+ASt89hpEg=="

COG_SERVICE_KEY = "FOoSeBQEfW8"
XTG_TEMPPLATE_ORIG = """
You are an antenna part selector chat bot.  
You are given sections of a catalog of antennas.
Each document section includes the part number at the beginning as an example: Text applies to part number: AVX-E_X1005245.  When answering questions, refer to antennas by their part number.
Use these sections of different documents to provide answers regarding antennas.
Refer to antennas by their part number.
When asked about criteria of a part, provide all part numbers that meet the criteria, not just one.
You are to answer questions based on the context provided.
If the answer is not contained within the text below, say \"I don't know\".
Below is the question for you to answer based on your data:
QUESTION: {question}
"""


TTEC_questiontemplate = """
You are an IT HelpDesk Chat Bot.  You are to answer questions based on the context provided.  
If the answer is complex, provide the answer as steps. 
If the question can be answered differently for different systems, 
provide answers for each system, and seperate the answers with a new line.
If the answer is not contained within the text below, say \"I don't know\".
In addition provide 2 suggested follow up questions that you have an answer for in a section titled \"Follow Up Questions\", 
which you seperate from the answers with a 2 new lines.
Below is the question for you to answer based on your data:
QUESTION: {question}
"""


DEBUG = "1"

creds = {
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_KEY": AZURE_OPENAI_KEY,
    "GTPTurbo": GTPTurbo,
    "COG_SEARCH_RESOURCE": COG_SEARCH_RESOURCE,
    "COG_SEARCH_KEY": COG_SEARCH_KEY,
    "COG_SEARCH_INDEX": COG_SEARCH_INDEX,
    "QUESTION_TEMPLATE" : TTEC_questiontemplate,
    "TEXT_DAVINCI": TEXT_DAVINCI,
    "STORAGE_KEY": STORAGE_KEY,
    "STORAGE_CONNECTION_STRING": STORAGE_CONNECTION_STRING,
    "STORAGE_ACCOUNT": STORAGE_ACCOUNT,
    "STORAGE_CONTAINER": ITHELPDESK_CONTAINER,
    "COG_SERVICE_KEY": COG_SERVICE_KEY,
    "APP_FUNCTION_URL": APP_FUNCTION_URL,
    "SEARCH_DETAILS_ACROSS_FILES": "1",
}