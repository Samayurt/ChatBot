# ChatBot
### LLM_PROJECT using llama2 model 

# Detailed explanation of the architecture design based on the diagram:

### User Interface: This is the main entry point where the user interacts with the application. It handles user input and displays the application's output.

### Authentication: This component is responsible for verifying the user's identity and authorizing their access to the application's functionalities.

### Chat Modes: This is the central component that manages the different modes of the chat interface. It acts as a router, directing the user's input and requests to the appropriate sub-components.

### Appointment Booking: This mode allows users to book appointments. It interacts with the Appointment Management component to handle appointment-related tasks, such as scheduling, rescheduling, and cancellations.

### Document QnA: This mode enables users to interact with PDF documents stored in the system. It leverages the PDF Processing component to extract and analyze the document content, and the Vector Store to store and retrieve relevant information efficiently.

### Appointment Management: This component is responsible for managing the appointment-related data, including booking, updating, and retrieving appointment details.

### PDF Processing: This component handles the processing of PDF documents, such as extracting text, analyzing the content, and preparing the data for the Vector Store.

### Vector Store: This is a data storage and retrieval system, likely a vector database, that efficiently stores and retrieves information related to the uploaded PDF documents. This enables the Document QnA mode to quickly access and respond to user queries.

## General Chat: This mode provides a general-purpose chat interface for users, allowing them to engage in open-ended conversations. It utilizes the LLM Model to generate appropriate responses based on the user's input.
## LLM Model: This is the core component that employs a Large Language Model (LLM) to understand and generate natural language responses. It powers the conversational abilities of the application, including the General Chat mode, Appointment Booking, and Document QnA.
