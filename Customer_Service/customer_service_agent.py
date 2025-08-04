import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_groq import ChatGroq

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str
    messages: List[BaseMessage]
    thread_id: str

def categorize(state: State) -> State:
    query = state["query"]
    
    template = """
    You are an expert customer support agent responsible for categorizing incoming queries.
    Analyze the following customer query and provide only one word from the list below that best describes its category. Do not include any other text or explanation.
    Categories: Technical, General, Billing

    Query: {query}
    """

    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm
    category = chain.invoke({"query": query}).content
  
    return {"category": category}

def analyze_sentiment(state: State) -> State:
    query = state["query"]
    
    template = """
    You are an AI sentiment analysis model.
    Analyze the sentiment of the following customer query.
    Respond with only one word: 'Positive', 'Negative', or 'Neutral'. Do not include any other text.

    Query: {query}
    """

    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm
    sentiment = chain.invoke({"query": query}).content

    return {"sentiment": sentiment}

def handle_technical(state: State) -> State:
    query = state["query"]
    messages = state.get("messages", [])
    
    # Build context from previous messages
    context = ""
    if messages:
        context = "Previous conversation:\n"
        for msg in messages[-6:]:  # Keep last 6 messages for context
            if isinstance(msg, HumanMessage):
                context += f"Customer: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                context += f"Agent: {msg.content}\n"
        context += "\n"
    
    template = """
    You are a helpful and knowledgeable technical support agent.
    {context}
    Based on the customer's current query and any previous conversation context, provide a concise and clear technical support response.
    If you cannot fully resolve the issue, guide the user on the next steps or where to find more information.

    Current Query: {query}
    """

    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = prompt | llm
    response = chain.invoke({"query": query, "context": context}).content

    # Update messages
    updated_messages = messages + [
        HumanMessage(content=query),
        AIMessage(content=response)
    ]

    return {"response": response, "messages": updated_messages}

def handle_billing(state: State) -> State:
    query = state["query"]
    messages = state.get("messages", [])
    
    # Build context from previous messages
    context = ""
    if messages:
        context = "Previous conversation:\n"
        for msg in messages[-6:]:  # Keep last 6 messages for context
            if isinstance(msg, HumanMessage):
                context += f"Customer: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                context += f"Agent: {msg.content}\n"
        context += "\n"

    template = """
    You are a helpful and accurate billing support agent.
    {context}
    Based on the customer's current query and any previous conversation context, provide a clear and precise response regarding their billing inquiry.
    If you require more information to resolve the query, ask clarifying questions.

    Current Query: {query}
    """

    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm
    response = chain.invoke({"query": query, "context": context}).content

    # Update messages
    updated_messages = messages + [
        HumanMessage(content=query),
        AIMessage(content=response)
    ]

    return {"response": response, "messages": updated_messages}

def handle_general(state: State) -> State:
    query = state["query"]
    messages = state.get("messages", [])
    
    # Build context from previous messages
    context = ""
    if messages:
        context = "Previous conversation:\n"
        for msg in messages[-6:]:  # Keep last 6 messages for context
            if isinstance(msg, HumanMessage):
                context += f"Customer: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                context += f"Agent: {msg.content}\n"
        context += "\n"

    template = """
    You are a friendly and helpful general customer support agent.
    {context}
    Based on the customer's current query and any previous conversation context, address the customer's query directly and provide a clear, concise, and polite response.

    Current Query: {query}
    """

    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm
    response = chain.invoke({"query": query, "context": context}).content

    # Update messages
    updated_messages = messages + [
        HumanMessage(content=query),
        AIMessage(content=response)
    ]

    return {"response": response, "messages": updated_messages}

def escalate(state: State) -> State:
    query = state["query"]
    messages = state.get("messages", [])
    
    response = "This query has been escalated to a human agent due to the negative sentiment. A specialist will contact you shortly to address your concerns."
    
    # Update messages
    updated_messages = messages + [
        HumanMessage(content=query),
        AIMessage(content=response)
    ]

    return {"response": response, "messages": updated_messages}

def route_query(state: State) -> str:
    sentiment = state["sentiment"]
    category = state["category"]

    if sentiment == "Negative":
        return "escalate"
    elif category == "Technical":
        return "handle_technical"
    elif category == "Billing":
        return "handle_billing"
    else:
        return "handle_general"

# Create workflow with memory
workflow = StateGraph(State)

workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

workflow.add_edge(START, "categorize")
workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query,
    {
        "handle_technical": "handle_technical",
        "handle_billing": "handle_billing",
        "handle_general": "handle_general",
        "escalate": "escalate"
    }
)
workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)

# Initialize memory saver
memory = MemorySaver()

# Compile graph with memory
graph = workflow.compile(checkpointer=memory)

class CustomerServiceAgent:

    def __init__(self):
        self.graph = graph
    
    
    def process_query_stream(self, query: str, thread_id: str = "default"):
        """Process a query with streaming response"""
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get current state to retrieve existing messages
        try:
            current_state = self.graph.get_state(config)
            existing_messages = current_state.values.get("messages", []) if current_state.values else []
        except:
            existing_messages = []
        
        initial_state = {
            "query": query,
            "category": "",
            "sentiment": "",
            "response": "",
            "messages": existing_messages,
            "thread_id": thread_id
        }
        
        for chunk in self.graph.stream(initial_state, config):
            yield chunk
    
    
    def clear_conversation(self, thread_id: str = "default"):
        """Clear conversation history for a specific thread"""
        config = {"configurable": {"thread_id": thread_id}}
        # Reset the state by invoking with empty messages
        initial_state = {
            "query": "",
            "category": "",
            "sentiment": "",
            "response": "",
            "messages": [],
            "thread_id": thread_id
        }
        self.graph.invoke(initial_state, config)
