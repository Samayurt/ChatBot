import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
import time
import re
import json
from datetime import datetime, timedelta

def set_custom_css():
    st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 15%;
    }
    .chat-message .avatar img {
        max-width: 60px;
        max-height: 60px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 85%;
        padding: 0 1.5rem;
        color: #fff;
    }
    </style>
    """, unsafe_allow_html=True)

def get_llama2_response(user_input, llm, chat_history, user_info):
    user_context = "User Information: "
    if user_info:
        user_context += f"Name: {user_info.get('name', 'N/A')}, "
        user_context += f"Email: {user_info.get('email', 'N/A')}, "
        user_context += f"Age: {user_info.get('age', 'N/A')}, "
        user_context += f"Phone: {user_info.get('phone', 'N/A')}\n"
    else:
        user_context += "No user information available.\n"
    
    prompt = user_context + "\n".join(chat_history) + f"\nUser input: {user_input}\nAssistant response: "
    try:
        response = llm(prompt)
        return response.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=0,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    if not text_chunks:
        st.error("No text chunks to process. Please check if the PDFs contain extractable text.")
        return None

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None

def get_conversation_chain(vectorstore, memory):
    if vectorstore is None:
        return None

    llm = CTransformers(
        model='model/llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        config={'max_new_tokens': 200, 'temperature': 0.01}
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def validate_age(age):
    try:
        age = int(age)
        return 0 < age < 120
    except ValueError:
        return False

def validate_phone(phone):
    pattern = r'^\+?1?\d{9,15}$'
    return re.match(pattern, phone) is not None

def authenticate_user(memory):
    st.header("Welcome to AI Chat Assistant")
    st.write("Let's get to know each other! I'll ask you a few questions for authentication.")

    if 'auth_step' not in st.session_state:
        st.session_state.auth_step = 0
    
    if 'user_info' not in st.session_state:
        st.session_state.user_info = {}

    questions = [
        "What's your name?",
        "What's your email address?",
        "How old are you?",
        "What's your phone number? (Include country code if applicable)"
    ]

    validators = [
        lambda x: len(x) > 0,  # Name
        validate_email,        # Email
        validate_age,          # Age
        validate_phone         # Phone
    ]

    chat_history = st.empty()

    def update_chat_history():
        messages = []
        for i, (q, a) in enumerate(zip(questions[:st.session_state.auth_step], st.session_state.user_info.values())):
            messages.append(f"Assistant: {q}")
            messages.append(f"You: {a}")
        chat_history.markdown('\n\n'.join(messages))

    update_chat_history()

    if st.session_state.auth_step < len(questions):
        st.write(questions[st.session_state.auth_step])
        user_input = st.chat_input("Your response:")
        
        if user_input:
            if validators[st.session_state.auth_step](user_input):
                question_key = ['name', 'email', 'age', 'phone'][st.session_state.auth_step]
                st.session_state.user_info[question_key] = user_input
                memory.save_context({"input": questions[st.session_state.auth_step]}, {"output": user_input})
                st.session_state.auth_step += 1
                update_chat_history()
                st.experimental_rerun()
            else:
                st.error("Invalid input. Please try again.")
    
    if st.session_state.auth_step == len(questions):
        st.success("Great! I've collected all the information. Let's start our chat!")
        save_user_info(st.session_state.user_info)
        return True

    return False

def save_user_info(user_info):
    with open("user_info.json", "w") as f:
        json.dump(user_info, f)
    st.success("Your information has been saved. You can now proceed to the chat interface.")

def load_user_info():
    try:
        with open("user_info.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def load_llm():
    @st.cache_resource
    def _load_llm():
        return CTransformers(
            model='model/llama-2-7b-chat.ggmlv3.q8_0.bin',
            model_type='llama',
            config={'max_new_tokens': 256, 'temperature': 0.7}
        )
    return _load_llm()

def process_documents(pdf_docs):
    with st.spinner("Processing..."):
        raw_text = get_pdf_text(pdf_docs)
        if not raw_text.strip():
            st.error("No text could be extracted from the uploaded PDFs. Please check if the files contain readable text.")
        else:
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            if vectorstore:
                st.session_state.conversation = get_conversation_chain(vectorstore, st.session_state.memory)
                st.success("Documents processed successfully!")
            else:
                st.error("Failed to process documents. Please try again or check the console for more information.")

def save_appointment(user_info, date, time, reason):
    appointments = load_appointments()
    appointment = {
        "user": user_info['name'],
        "date": date,
        "time": time,
        "reason": reason
    }
    appointments.append(appointment)
    with open("appointments.json", "w") as f:
        json.dump(appointments, f)

def load_appointments():
    try:
        with open("appointments.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def get_available_slots(date):
    appointments = load_appointments()
    booked_times = [appt['time'] for appt in appointments if appt['date'] == date]
    all_slots = [f"{h:02d}:00" for h in range(9, 18)]  # 9 AM to 5 PM
    available_slots = [slot for slot in all_slots if slot not in booked_times]
    return available_slots

def get_user_appointments(user_info):
    appointments = load_appointments()
    user_appointments = [appt for appt in appointments if appt['user'] == user_info['name']]
    return user_appointments

def chat_interface(llm, memory):
    user_info = load_user_info()
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What's on your mind?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = ""
            
            try:
                if "book appointment" in prompt.lower():
                    st.session_state.booking_state = "date"
                    assistant_response = "Sure, I can help you book an appointment. Let's start with the date. Please enter the date for your appointment (YYYY-MM-DD format):"
                elif "my appointments" in prompt.lower():
                    user_appointments = get_user_appointments(user_info)
                    if user_appointments:
                        assistant_response = "Your appointments:\n" + "\n".join([f"Date: {appt['date']}, Time: {appt['time']}, Reason: {appt['reason']}" for appt in user_appointments])
                    else:
                        assistant_response = "You have no upcoming appointments."
                elif hasattr(st.session_state, 'booking_state'):
                    if st.session_state.booking_state == "date":
                        try:
                            date = datetime.strptime(prompt, "%Y-%m-%d").date()
                            available_slots = get_available_slots(str(date))
                            if available_slots:
                                st.session_state.booking_date = str(date)
                                st.session_state.booking_state = "time"
                                assistant_response = f"Great! For {date}, we have the following time slots available: {', '.join(available_slots)}. Please choose a time:"
                            else:
                                assistant_response = f"I'm sorry, there are no available slots for {date}. Please choose another date (YYYY-MM-DD format):"
                        except ValueError:
                            assistant_response = "Invalid date format. Please use YYYY-MM-DD format:"
                    elif st.session_state.booking_state == "time":
                        if prompt in get_available_slots(st.session_state.booking_date):
                            st.session_state.booking_time = prompt
                            st.session_state.booking_state = "reason"
                            assistant_response = "Excellent! Now, please provide a reason for your appointment:"
                        else:
                            assistant_response = f"I'm sorry, that time is not available. Please choose from the available slots: {', '.join(get_available_slots(st.session_state.booking_date))}"
                    elif st.session_state.booking_state == "reason":
                        save_appointment(user_info, st.session_state.booking_date, st.session_state.booking_time, prompt)
                        assistant_response = f"Your appointment has been booked for {st.session_state.booking_date} at {st.session_state.booking_time}. Reason: {prompt}"
                        del st.session_state.booking_state
                        del st.session_state.booking_date
                        del st.session_state.booking_time
                elif st.session_state.chat_mode == "General":
                    chat_history = [f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}" for msg in st.session_state.chat_history]
                    assistant_response = get_llama2_response(prompt, llm, chat_history[-10:], user_info)
                else:  
                    if st.session_state.conversation:
                        response = st.session_state.conversation({'question': prompt})
                        assistant_response = response['answer']
                    else:
                        assistant_response = "I'm sorry, but the conversation model is not initialized. Please process your documents first."

                if "my information" in prompt.lower() or "my details" in prompt.lower():
                    assistant_response = f"Here's the information you provided:\n\n"
                    for key, value in user_info.items():
                        assistant_response += f"{key.capitalize()}: {value}\n"
            
            except Exception as e:
                assistant_response = f"An error occurred: {str(e)}"

            if assistant_response:
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            else:
                full_response = "I'm sorry, I couldn't generate a response."
                message_placeholder.markdown(full_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        memory.save_context({"input": prompt}, {"output": full_response})

def main():
    load_dotenv()
    st.set_page_config(page_title="AI Chat Assistant", page_icon=":speech_balloon:")
    set_custom_css()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if not st.session_state.authenticated:
        st.session_state.authenticated = authenticate_user(st.session_state.memory)

    if st.session_state.authenticated:
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "chat_mode" not in st.session_state:
            st.session_state.chat_mode = "General"

        with st.sidebar:
            st.subheader("Chat Mode")
            chat_mode = st.radio("Select chat mode:", ("General", "Document QnA", "Appointment Booking"))
            st.session_state.chat_mode = chat_mode

            if chat_mode == "Document QnA":
                st.subheader("Your documents")
                pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
                if st.button("Process"):
                    process_documents(pdf_docs)
            elif chat_mode == "Appointment Booking":
                st.subheader("Appointment Booking")
                st.write("Use the chat interface to book appointments or check your existing appointments.")

        st.header(f"AI Chat Assistant - {st.session_state.chat_mode} Mode :speech_balloon:")

        llm = load_llm()
        chat_interface(llm, st.session_state.memory)

if __name__ == '__main__':
    main()