import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# LangChain Agent components
from langchain.agents import create_react_agent, AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool, BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.retriever import create_retriever_tool
from langchain import hub

# LangGraph for orchestration
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# ================================
# Load environment variables
# ================================
load_dotenv()

# ================================
# Enhanced State for Agent System
# ================================
class AgentState(TypedDict, total=False):
    question: str
    context: str
    draft_answer: str
    final_answer: str
    retrieved_docs: List[Dict[str, Any]]
    agent_thoughts: List[str]
    tools_used: List[str]
    reasoning_steps: List[str]

# ================================
# Custom Tools for Agents
# ================================
class DocumentRetrieverTool(BaseTool):
    """Custom tool for document retrieval"""
    name: str = "document_retriever"
    description: str = "Searches through uploaded documents to find relevant information based on a query"
    
    def __init__(self, vectorstore: FAISS):
        super().__init__()
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.1}
        )
    
    def _run(self, query: str) -> str:
        """Execute the tool"""
        try:
            docs = self.retriever.get_relevant_documents(query)
            if not docs:
                return "No relevant documents found for the query."
            
            # Format results with metadata
            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                results.append(f"Document {i} (Source: {source}, Page: {page}):\n{doc.page_content}")
            
            return "\n\n---\n\n".join(results)
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"

class ContextAnalyzerTool(BaseTool):
    """Tool to analyze and summarize context"""
    name: str = "context_analyzer"
    description: str = "Analyzes retrieved context to identify key themes, topics, and relevant information"
    
    def _run(self, context: str) -> str:
        """Analyze the context"""
        if not context:
            return "No context provided to analyze."
        
        # Simple analysis (in real implementation, could use another LLM)
        lines = context.split('\n')
        word_count = len(context.split())
        doc_count = context.count('Document ')
        
        analysis = f"""
Context Analysis:
- Total words: {word_count}
- Number of document chunks: {doc_count}
- Content length: {'Long' if word_count > 1000 else 'Medium' if word_count > 500 else 'Short'}
- Key themes: {'Technical' if any(word in context.lower() for word in ['algorithm', 'method', 'system']) else 'General'}
"""
        return analysis.strip()

# ================================
# Agent Creation Functions
# ================================
def create_research_agent(vectorstore: FAISS) -> AgentExecutor:
    """Create a research agent using LangChain's built-in functions"""
    
    # Initialize LLM
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.1,
        max_tokens=1000
    )
    
    # Create tools
    document_tool = DocumentRetrieverTool(vectorstore=vectorstore)
    context_analyzer = ContextAnalyzerTool()
    
    # Create retriever tool using LangChain's built-in function
    retriever_tool = create_retriever_tool(
        retriever=vectorstore.as_retriever(),
        name="document_search",
        description="Search through uploaded documents for relevant information"
    )
    
    tools = [document_tool, context_analyzer, retriever_tool]
    
    # Get ReAct prompt from LangChain hub
    try:
        prompt = hub.pull("hwchase17/react")
    except:
        # Fallback prompt if hub is not accessible
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful research assistant. You have access to tools to search and analyze documents.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""),
        ])
    
    # Create ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor

def create_answer_agent() -> AgentExecutor:
    """Create an answer generation agent"""
    
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.0
    )
    
    # Answer generation tool
    def answer_generator(context_and_question: str) -> str:
        """Generate answer from context and question"""
        try:
            parts = context_and_question.split("QUESTION:", 1)
            if len(parts) != 2:
                return "Please provide context and question in format: CONTEXT: ... QUESTION: ..."
            
            context = parts[0].replace("CONTEXT:", "").strip()
            question = parts[1].strip()
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer the question based ONLY on the provided context. If you cannot find the answer in the context, say 'I don't have enough information to answer this question.'"),
                ("human", f"Context:\n{context}\n\nQuestion: {question}")
            ])
            
            chain = prompt | llm
            response = chain.invoke({})
            return getattr(response, 'content', str(response))
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    answer_tool = Tool(
        name="answer_generator",
        description="Generate an answer based on provided context and question. Input should be in format: 'CONTEXT: ... QUESTION: ...'",
        func=answer_generator
    )
    
    tools = [answer_tool]
    
    # Simple prompt for answer agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an answer generation agent. Your job is to create accurate answers based on provided context.

Use the answer_generator tool to create responses. Always format your input as:
CONTEXT: [the context information]
QUESTION: [the question to answer]

Think step by step:
1. Review the context and question
2. Use the answer_generator tool
3. Provide the final answer

Question: {input}
{agent_scratchpad}"""),
    ])
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor

def create_critic_agent() -> AgentExecutor:
    """Create a critic agent for answer refinement"""
    
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )
    
    def answer_critic(answer_and_context: str) -> str:
        """Critique and improve an answer"""
        try:
            parts = answer_and_context.split("CONTEXT:", 1)
            if len(parts) != 2:
                return "Please provide in format: ANSWER: ... CONTEXT: ..."
            
            answer_part = parts[0].replace("ANSWER:", "").strip()
            context = parts[1].strip()
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a critic agent. Review the provided answer and improve it for clarity, accuracy, and completeness. Do not add information not present in the context."),
                ("human", f"Original Answer:\n{answer_part}\n\nContext:\n{context}\n\nProvide an improved version:")
            ])
            
            chain = prompt | llm
            response = chain.invoke({})
            return getattr(response, 'content', str(response))
        except Exception as e:
            return f"Error critiquing answer: {str(e)}"
    
    critic_tool = Tool(
        name="answer_critic",
        description="Critique and improve an answer. Input format: 'ANSWER: ... CONTEXT: ...'",
        func=answer_critic
    )
    
    tools = [critic_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a critic agent that improves answers for clarity and accuracy.

Use the answer_critic tool to review and enhance answers. Format your input as:
ANSWER: [the answer to improve]
CONTEXT: [the original context]

Steps:
1. Analyze the provided answer
2. Use the answer_critic tool
3. Return the improved answer

Question: {input}
{agent_scratchpad}"""),
    ])
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor

# ================================
# Agent Orchestration with LangGraph
# ================================
def research_agent_node(research_agent: AgentExecutor):
    """Research agent node for LangGraph"""
    def _node(state: AgentState) -> AgentState:
        try:
            # Add reasoning step
            reasoning = state.get("reasoning_steps", [])
            reasoning.append("🔍 Research Agent: Starting document search and analysis")
            state["reasoning_steps"] = reasoning
            
            # Execute research agent
            result = research_agent.invoke({"input": state["question"]})
            
            # Extract information from agent output
            output = result.get("output", "")
            
            # Store context and update state
            state["context"] = output
            state["tools_used"] = state.get("tools_used", []) + ["document_retriever", "context_analyzer"]
            
            reasoning.append("✅ Research Agent: Completed document analysis")
            state["reasoning_steps"] = reasoning
            
        except Exception as e:
            state["context"] = f"Research agent error: {str(e)}"
            
        return state
    
    return _node

def answer_agent_node(answer_agent: AgentExecutor):
    """Answer agent node for LangGraph"""
    def _node(state: AgentState) -> AgentState:
        try:
            reasoning = state.get("reasoning_steps", [])
            reasoning.append("💭 Answer Agent: Generating initial answer")
            state["reasoning_steps"] = reasoning
            
            # Format input for answer agent
            context = state.get("context", "")
            question = state["question"]
            agent_input = f"Based on this context: {context}\n\nAnswer this question: {question}"
            
            result = answer_agent.invoke({"input": agent_input})
            state["draft_answer"] = result.get("output", "No answer generated")
            
            state["tools_used"] = state.get("tools_used", []) + ["answer_generator"]
            
            reasoning.append("✅ Answer Agent: Generated initial response")
            state["reasoning_steps"] = reasoning
            
        except Exception as e:
            state["draft_answer"] = f"Answer agent error: {str(e)}"
            
        return state
    
    return _node

def critic_agent_node(critic_agent: AgentExecutor):
    """Critic agent node for LangGraph"""
    def _node(state: AgentState) -> AgentState:
        try:
            reasoning = state.get("reasoning_steps", [])
            reasoning.append("✨ Critic Agent: Refining and improving answer")
            state["reasoning_steps"] = reasoning
            
            # Format input for critic agent
            draft_answer = state.get("draft_answer", "")
            context = state.get("context", "")
            agent_input = f"Please improve this answer: {draft_answer}\n\nUsing this context: {context}"
            
            result = critic_agent.invoke({"input": agent_input})
            state["final_answer"] = result.get("output", state["draft_answer"])
            
            state["tools_used"] = state.get("tools_used", []) + ["answer_critic"]
            
            reasoning.append("🎯 Critic Agent: Completed answer refinement")
            state["reasoning_steps"] = reasoning
            
        except Exception as e:
            state["final_answer"] = state.get("draft_answer", f"Critic agent error: {str(e)}")
            
        return state
    
    return _node

def build_agent_graph(vectorstore: FAISS):
    """Build the multi-agent graph using LangChain built-in agents"""
    
    # Create agents
    research_agent = create_research_agent(vectorstore)
    answer_agent = create_answer_agent()
    critic_agent = create_critic_agent()
    
    # Build graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("research", research_agent_node(research_agent))
    graph.add_node("answer", answer_agent_node(answer_agent))
    graph.add_node("critic", critic_agent_node(critic_agent))
    
    # Add edges
    graph.add_edge(START, "research")
    graph.add_edge("research", "answer")
    graph.add_edge("answer", "critic")
    graph.add_edge("critic", END)
    
    return graph.compile()

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="🤖 LangChain Built-in Agents RAG")
st.title("🤖 Multi-Agent RAG with LangChain Built-in Agents")
st.write("Upload a PDF and ask questions using LangChain's built-in agent framework with ReAct reasoning!")

# Sidebar with agent information
with st.sidebar:
    st.header("🔧 Agent System")
    st.write("**Research Agent**: Uses document retrieval and analysis tools")
    st.write("**Answer Agent**: Generates responses using built-in reasoning")
    st.write("**Critic Agent**: Refines and improves answers")
    st.write("---")
    st.write("**Framework**: LangChain ReAct Agents + LangGraph")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save temp PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    with st.spinner("📚 Processing document and initializing agents..."):
        # Load & split PDF
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Create embeddings & FAISS store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Build agent graph
        agent_graph = build_agent_graph(vectorstore)
    
    st.success(f"✅ Document processed! Created {len(chunks)} chunks for agent analysis.")

    # Question input
    user_question = st.text_input("Ask your question:")
    
    if user_question:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.spinner("🤖 Agents working together..."):
                # Initialize state
                state_in: AgentState = {
                    "question": user_question,
                    "reasoning_steps": [],
                    "tools_used": []
                }
                
                # Process through agent graph
                state_out = agent_graph.invoke(state_in)
            
            # Display final answer
            st.subheader("📝 Final Answer:")
            st.write(state_out.get("final_answer", "No answer generated"))
            
            # Show agent reasoning
            with st.expander("🧠 Agent Reasoning Process"):
                reasoning_steps = state_out.get("reasoning_steps", [])
                for step in reasoning_steps:
                    st.write(step)
            
            # Show draft vs final
            with st.expander("📊 Answer Evolution"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Draft Answer:**")
                    st.write(state_out.get("draft_answer", "N/A"))
                with col_b:
                    st.write("**Final Answer:**")
                    st.write(state_out.get("final_answer", "N/A"))
        
        with col2:
            # Agent activity summary
            st.subheader("🔧 Agent Activity")
            tools_used = state_out.get("tools_used", [])
            for tool in set(tools_used):
                st.write(f"✅ {tool}")
            
            st.metric("Tools Used", len(set(tools_used)))
            st.metric("Reasoning Steps", len(state_out.get("reasoning_steps", [])))
            
            # Context preview
            with st.expander("📄 Retrieved Context"):
                context = state_out.get("context", "")
                if len(context) > 500:
                    st.write(context[:500] + "...")
                else:
                    st.write(context)

else:
    st.info("👆 Please upload a PDF document to start using the agent system!")
    
    # Show agent capabilities
    st.subheader("🚀 Agent Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🔍 Research Agent**")
        st.write("• Document retrieval")
        st.write("• Context analysis")
        st.write("• Information synthesis")
    
    with col2:
        st.write("**💭 Answer Agent**")
        st.write("• Response generation")
        st.write("• Context-based reasoning")
        st.write("• Structured thinking")
    
    with col3:
        st.write("**✨ Critic Agent**")
        st.write("• Answer refinement")
        st.write("• Quality improvement")
        st.write("• Clarity enhancement")

# Cleanup
try:
    if 'temp_pdf_path' in locals():
        os.unlink(temp_pdf_path)
except:
    pass