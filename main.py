from pinecone import Pinecone
import os
from dotenv import load_dotenv
from groq import Groq
from semantic_router.encoders import HuggingFaceEncoder
import gradio as gr
import time

load_dotenv()

gr.close_all()

#### pinecone setup
PINECONE_API_KEY = os.getenv('pinecone')
index_name = "course-app"
pc = Pinecone(api_key="your-api-key")

index = pc.Index(index_name)

groq_key = os.getenv('groq')
client = Groq(api_key=groq_key)

chat_history = []

#### embedding initialization
encoder = HuggingFaceEncoder(name = "dwzhu/e5-base-4k")

def retrieve_context(query):
    query_embedding = encoder(query)
    results = index.query(vector=query_embedding, top_k=1, include_metadata=True)
    # if results != {}:
    #     return results['matches'][0]['metadata']['text']
    # else:
    #     return "No records found !!!"
    
    return results.get('matches', [{}])[0].get('metadata', {}).get('text', "No records found !!!")
    
def generate_response(query):
    context = retrieve_context(query)

    prompt = f"""
    Restructure the following information to provide a clearer, well-formatted response using the context below
    Retrieved Context:
    {context}
    Provide a concise and well-structured response.
    """
    restructure = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content":  prompt
        }
        ,
        {
            "role": "user",
            "content": query
        }],
    temperature=0.5,
    max_tokens=1024,
    top_p=1,
    stop=None,
    stream=False,
    )
    return restructure.choices[0].message.content

# demo = gr.Interface(fn=generate_response,
#                     inputs=[gr.Textbox(label="Input text to summarize",lines=6)],
#                     outputs=[gr.Textbox(label="Summarized text",lines=4)],
#                     title="Text Summarizer",
#                     description="THIS APPLICATION WILL BE USED TO SUMMARIZE THE TEXT")

# demo.launch()


history = []

with gr.Blocks() as demo:
    gr.Markdown("## COURSE APP CHATBOT")
    
    chatbot = gr.Chatbot(history, type="messages")
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    state = gr.State([])

    def respond(message, chat_history):
    
        bot_message = generate_response(message)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, state], [msg, chatbot])
    state.change(lambda x: x, state, chatbot)

demo.launch(share = True)