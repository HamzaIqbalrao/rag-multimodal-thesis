# Multimodal RAG with Elasticseach and geoqueries 🏔️

Using Elasticsearch as a vector database to store, embed, and retrieve enterprise knowledge enabling more accurate, context-aware, and hallucination-reduced responses.




### Web Application
A front end based on streamlit that handles the user input and runs the search pipeline to obtain the LLM final response and displays images from the search results with their descriptions.

- An Elastic deployment
- Ollama running locally and the model cogito:3b running
- To copy this directory locally


### Indexing:
- Run the upload_documents.py file, it will take the images and metadata from the images, generate the necessary embeddings and upload all the data to Elastic.

Simply run:
```
python upload_documents.py
```
### Search:

You can run the LLM_conversation.py file to get the responses in the console

You can do this by running:
```
python LLM_conversation.py
```


Or simply run the streamlit app and use the UI in your browser using:

```
streamlit run streamlit_app.py
```
