import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
from os.path import exists
import numpy as np
import math

import urllib.request

@st.cache
def download_file():
    url = "https://drive.google.com/uc?export=download&id=1e_bneSaNGhY77Nt07RhTjcMekvwHRGjS"
    path = "file.json"

    # Use urllib.request.urlretrieve to download the file from the given URL
    urllib.request.urlretrieve(url, path)

    # Return the path to the downloaded file
    return path

# Download the file and get the path to the downloaded file
path = download_file()

# The file should now be downloaded and saved to the server


@st.cache(allow_output_mutation=True)
def load_model():
    model1 = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    return model1

model = load_model()

  
def get_embeddings(texts):
    if type(texts) == str:
        texts = [texts]
    texts = [text.replace("\n", " ") for text in texts]
    return model.encode(texts)
  
def read_json(json_path):
    print('Loading embeddings from "{}"'.format(json_path))
    with open(json_path, 'r') as f:
        values = json.load(f)
    return (values['chapters'], np.array(values['embeddings']))

def read_epub(book_path, json_path, preview_mode, first_chapter, last_chapter):
    chapters = get_chapters(book_path, preview_mode, first_chapter, last_chapter)
    if preview_mode:
        return (chapters, None)
    print('Generating embeddings for chapters {}-{} in "{}"\n'.format(first_chapter, last_chapter, book_path))
    paras = [para for chapter in chapters for para in chapter['paras']]
    embeddings = get_embeddings(paras)
    try:
        with open(json_path, 'w') as f:
            json.dump({'chapters': chapters, 'embeddings': embeddings.tolist()}, f)
    except:
        print('Failed to save embeddings to "{}"'.format(json_path))
    return (chapters, embeddings)
  
def process_file(path, preview_mode=False, first_chapter=0, last_chapter=math.inf):
    values = None
    if path[-4:] == 'json':
        values = read_json(path)
    elif path[-4:] == 'epub':
        json_path = 'embeddings-{}-{}-{}.json'.format(first_chapter, last_chapter, path)
        if exists(json_path):
            values = read_json(json_path)
        else:
            values = read_epub(path, json_path, preview_mode, first_chapter, last_chapter) 
    else:
        print('Invalid file format. Either upload an epub or a json of book embeddings.')        
    return values
  
chapters, embeddings = process_file(path)
  
def print_and_write(text, f):
    print(text)
    f.write(text + '\n')

def index_to_para_chapter_index(index, chapters):
    for chapter in chapters:
        paras_len = len(chapter['paras'])
        if index < paras_len:
            return chapter['paras'][index], chapter['title'], index
        index -= paras_len
    return None

def search(query, embeddings, n=3):
    query_embedding = get_embeddings(query)[0]
    scores = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
    results = sorted([i for i in range(len(embeddings))], key=lambda i: scores[i], reverse=True)[:n]

    f = open('result.text', 'a')
    header_msg ='Results for query "{}"'.format(query)
    print_and_write(header_msg, f)
    for index in results:
        para, title, para_no = index_to_para_chapter_index(index, chapters)
        result_msg = '\n"{}"'.format(para)
        print_and_write(result_msg, f)
    print_and_write('\n', f)


st.title("Search App")

query = st.text_input("Enter your query:")

if st.button("Find"):
  with open('result.text', 'w') as f:
      f.truncate()
  results2 = search(query,embeddings)
  st.write("Results:")
# Open the result.text file in read mode
  f = open('result.text', 'r')

# Read the contents of the file
  file_content = f.read()

# Close the file
  f.close()

# Use the st.text() method to display the contents of the file
  st.text(file_content)
