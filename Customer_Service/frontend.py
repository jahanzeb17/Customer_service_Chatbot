import streamlit as st
import uuid
from customer_service_agent import CustomerServiceAgent

# Initialize the agent
if 'agent' not in st.session_state:
    st.session_state.agent = CustomerServiceAgent()

# Initialize session state
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False

# Page configuration
st.set_page_config(
    page_title="Customer Service Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
        color: #333333;
    }
    
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 5px solid #2196f3;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        color: #1a1a1a;
        font-size: 16px;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 5px solid #9c27b0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        color: #1a1a1a;
        font-size: 16px;
    }
    
    .user-message strong {
        color: #1976d2;
        font-weight: 600;
    }
    
    .assistant-message strong {
        color: #7b1fa2;
        font-weight: 600;
    }
    
    .sentiment-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin-left: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .positive { 
        background-color: #4caf50; 
        color: white; 
        border: 1px solid #388e3c;
    }
    
    .negative { 
        background-color: #f44336; 
        color: white; 
        border: 1px solid #d32f2f;
    }
    
    .neutral { 
        background-color: #ff9800; 
        color: white; 
        border: 1px solid #f57c00;
    }
    
    .category-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin-left: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .technical { 
        background-color: #2196f3; 
        color: white; 
        border: 1px solid #1976d2;
    }
    
    .billing { 
        background-color: #4caf50; 
        color: white; 
        border: 1px solid #388e3c;
    }
    
    .general { 
        background-color: #ff9800; 
        color: white; 
        border: 1px solid #f57c00;
    }
    
    /* Chat container styling */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 10px;
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
    }
    
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🤖 Customer Service Agent")
    st.write("AI-powered customer support with memory")
    
    # Thread management
    st.subheader("Session Management")
    
    if st.button("🔄 New Conversation", type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.conversation_started = False
        st.rerun()
    
    
    # Display thread info
    st.info(f"Thread ID: {st.session_state.thread_id[:8]}...")
    
    # Statistics
    st.subheader("Session Stats")
    message_count = len(st.session_state.messages)
    st.metric("Messages", message_count)
    

# Main chat interface
st.title("Customer Service Chat")

# Display conversation history
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display metadata if available
            metadata_html = ""
            if "metadata" in message:
                sentiment = message["metadata"].get("sentiment", "").lower()
                category = message["metadata"].get("category", "").lower()
                
                if sentiment:
                    metadata_html += f'<span class="sentiment-badge {sentiment}">{sentiment.title()}</span>'
                if category:
                    metadata_html += f'<span class="category-badge {category}">{category.title()}</span>'
            
            st.markdown(f"""
            <div class="assistant-message">
                <strong>Agent:</strong> {message["content"]}
                {metadata_html}
            </div>
            """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Create a placeholder for the streaming response
    with st.spinner("Agent is thinking..."):
        # Process the query
        response_placeholder = st.empty()
        
        # Use streaming if available
        try:
            full_response = ""
            category = ""
            sentiment = ""
            
            # Stream the response
            for chunk in st.session_state.agent.process_query_stream(user_input, st.session_state.thread_id):
                if chunk:
                    # Extract the latest values from the chunk
                    for node, values in chunk.items():
                        if "response" in values and values["response"]:
                            full_response = values["response"]
                        if "category" in values and values["category"]:
                            category = values["category"]
                        if "sentiment" in values and values["sentiment"]:
                            sentiment = values["sentiment"]
                    
                    # Update the placeholder with current response
                    if full_response:
                        with response_placeholder.container():
                            # Display metadata
                            metadata_html = ""
                            if sentiment:
                                sentiment_clean = sentiment.lower().strip()
                                metadata_html += f'<span class="sentiment-badge {sentiment_clean}">{sentiment_clean.title()}</span>'
                            if category:
                                category_clean = category.lower().strip()
                                metadata_html += f'<span class="category-badge {category_clean}">{category_clean.title()}</span>'
                            
                            st.markdown(f"""
                            <div class="assistant-message">
                                <strong>Agent:</strong> {full_response}
                                {metadata_html}
                            </div>
                            """, unsafe_allow_html=True)
            
            # Add assistant response to chat history
            if full_response:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "metadata": {
                        "category": category,
                        "sentiment": sentiment
                    }
                })
        
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            
            # Fallback to non-streaming
            try:
                result = st.session_state.agent.process_query(user_input, st.session_state.thread_id)
                
                # Display the response
                metadata_html = ""
                if result.get("sentiment"):
                    sentiment_clean = result["sentiment"].lower().strip()
                    metadata_html += f'<span class="sentiment-badge {sentiment_clean}">{sentiment_clean.title()}</span>'
                if result.get("category"):
                    category_clean = result["category"].lower().strip()
                    metadata_html += f'<span class="category-badge {category_clean}">{category_clean.title()}</span>'
                
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>Agent:</strong> {result["response"]}
                    {metadata_html}
                </div>
                """, unsafe_allow_html=True)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["response"],
                    "metadata": {
                        "category": result.get("category", ""),
                        "sentiment": result.get("sentiment", "")
                    }
                })
                
            except Exception as e2:
                st.error(f"Fallback also failed: {str(e2)}")
    
    # Mark conversation as started
    st.session_state.conversation_started = True
    
    # Rerun to update the chat display
    st.rerun()


# Debug information (optional - remove in production)
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.subheader("Debug Information")
    st.sidebar.json({
        "thread_id": st.session_state.thread_id,
        "message_count": len(st.session_state.messages),
        "conversation_started": st.session_state.conversation_started
    })