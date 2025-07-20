import json
import time
from typing import Any, Dict, List, Optional, TypedDict

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langgraph.graph import END, Graph, StateGraph

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.core.rag import rag_system


class AgentState(TypedDict):
    """State for the email marketing agent."""

    messages: List[Dict[str, Any]]
    customer_data: Optional[Dict[str, Any]]
    strategy_context: Optional[str]
    retrieved_documents: List[Dict[str, Any]]
    current_step: str
    iteration_count: int
    final_strategy: Optional[Dict[str, Any]]
    error: Optional[str]


class EmailMarketingAgent:
    """LangGraph agent for email marketing strategy recommendations."""

    def __init__(self):
        self.logger = get_logger("email_agent")

        # Initialize Langfuse client for manual tracing
        self.langfuse_client = Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host,
        )

        # Initialize LLM without callback (will be set per request)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            api_key=settings.openai_api_key,
        )

        # Build the agent graph
        self.graph = self._build_graph()

    def _build_graph(self) -> Graph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_customer", self._analyze_customer_node)
        workflow.add_node("retrieve_knowledge", self._retrieve_knowledge_node)
        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("generate_strategy", self._generate_strategy_node)
        workflow.add_node("finalize_strategy", self._finalize_strategy_node)

        # Define the flow
        workflow.set_entry_point("analyze_customer")

        # Add edges
        workflow.add_edge("analyze_customer", "retrieve_knowledge")
        workflow.add_conditional_edges(
            "retrieve_knowledge",
            self._should_search_web,
            {"search": "web_search", "generate": "generate_strategy"},
        )
        workflow.add_edge("web_search", "generate_strategy")
        workflow.add_edge("generate_strategy", "finalize_strategy")
        workflow.add_edge("finalize_strategy", END)

        return workflow.compile()

    async def _analyze_customer_node(self, state: AgentState) -> AgentState:
        """Analyze customer data and requirements."""
        try:
            messages = state["messages"]
            latest_message = messages[-1]["content"] if messages else ""

            # Simple customer analysis for demo
            analysis_prompt = f"""
                Extract key information from this customer inquiry:
                {latest_message}

                Return JSON with:
                - business_type: type of business
                - goals: main email marketing goals
                """

            response = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content="Extract key business info from user messages."
                    ),
                    HumanMessage(content=analysis_prompt),
                ]
            )

            try:
                customer_data = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                customer_data = {"analysis": response.content}

            state["customer_data"] = customer_data
            state["current_step"] = "customer_analyzed"

            self.logger.info("Customer analysis completed", customer_data=customer_data)

            return state

        except Exception as e:
            self.logger.error("Customer analysis failed", error=str(e))
            state["error"] = f"Customer analysis failed: {str(e)}"
            return state

    async def _retrieve_knowledge_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant marketing knowledge."""
        # Create Langfuse span for knowledge retrieval
        span = (
            self.current_trace.span(
                name="retrieve_knowledge", input={"step": "knowledge_retrieval"}
            )
            if hasattr(self, "current_trace")
            else None
        )

        try:
            customer_data = state.get("customer_data", {})
            latest_message = (
                state["messages"][-1]["content"] if state["messages"] else ""
            )

            # Create search query
            search_query = f"Email marketing for {customer_data.get('business_type', 'business')} {customer_data.get('goals', latest_message)}"

            # Retrieve relevant documents
            context_data = await rag_system.get_context_for_query(
                query=search_query,
                max_context_tokens=1000,
                k=settings.max_retrieval_results,
                similarity_threshold=settings.similarity_threshold,
            )

            state["retrieved_documents"] = context_data["sources"]
            state["strategy_context"] = context_data["context"]
            state["current_step"] = "knowledge_retrieved"

            self.logger.info(
                "Knowledge retrieval completed",
                documents_used=context_data["documents_used"],
                total_tokens=context_data["total_tokens"],
            )

            # End Langfuse span with results
            if span:
                span.end(
                    output={
                        "documents_found": len(context_data["sources"]),
                        "tokens_used": context_data["total_tokens"],
                        "query": search_query,
                    }
                )

            return state

        except Exception as e:
            self.logger.error("Knowledge retrieval failed", error=str(e))
            state["error"] = f"Knowledge retrieval failed: {str(e)}"
            # End span with error
            if span:
                span.end(output={"error": str(e)})
            return state

    def _should_search_web(self, state: AgentState) -> str:
        """Decide whether to search web or use retrieved knowledge."""
        documents_used = len(state.get("retrieved_documents", []))

        # If we have relevant documents, use them to generate strategy
        # Otherwise, search the web for additional information
        if documents_used > 0:
            return "generate"
        else:
            return "search"

    async def _web_search_node(self, state: AgentState) -> AgentState:
        """Search web for additional information when knowledge base is insufficient."""
        # Create Langfuse span for web search
        span = (
            self.current_trace.span(name="web_search", input={"step": "web_search"})
            if hasattr(self, "current_trace")
            else None
        )

        try:
            # Get the original user message
            messages = state.get("messages", [])
            original_question = messages[-1]["content"] if messages else ""
            customer_data = state.get("customer_data", {})

            # Create search query for web
            search_query = f"email marketing {customer_data.get('business_type', '')} {original_question}"

            self.logger.info(
                "No relevant knowledge found, searching web", query=search_query
            )

            # Simulate web search results (in a real implementation, you'd use a web search tool)
            web_context = f"""
            Web search results for: {search_query}

            General email marketing best practices:
            - Personalize subject lines to increase open rates
            - Segment your audience based on behavior and demographics
            - Use A/B testing to optimize campaign performance
            - Ensure mobile-friendly design
            - Include clear call-to-action buttons
            - Monitor key metrics like open rates, click rates, and conversions
            """

            state["strategy_context"] = web_context
            state["retrieved_documents"] = [
                {"source": "web_search", "content": "General best practices"}
            ]
            state["current_step"] = "web_searched"

            self.logger.info("Web search completed")

            # End Langfuse span with results
            if span:
                span.end(
                    output={
                        "search_query": search_query,
                        "results_found": "simulated_web_results",
                        "context_length": len(web_context),
                    }
                )

            return state

        except Exception as e:
            self.logger.error("Web search failed", error=str(e))
            # Continue with empty context if web search fails
            state["strategy_context"] = "No additional information available."
            state["current_step"] = "web_search_failed"
            # End span with error
            if span:
                span.end(output={"error": str(e)})
            return state

    async def _generate_strategy_node(self, state: AgentState) -> AgentState:
        """Generate email marketing strategy."""
        try:
            customer_data = state.get("customer_data", {})
            context = state.get("strategy_context", "")
            iteration = state.get("iteration_count", 0)

            # Get the original user message
            messages = state.get("messages", [])
            original_question = messages[-1]["content"] if messages else ""

            # Check if context comes from knowledge base or web search
            context_source = (
                "knowledge base"
                if state.get("current_step") == "knowledge_retrieved"
                else "web search"
            )

            strategy_prompt = f"""
                Answer this email marketing question directly and concisely:

                Question: {original_question}

                Business Type: {customer_data.get('business_type', 'unknown')}
                Goals: {customer_data.get('goals', 'general marketing')}

                Information from {context_source}:
                {context[:800] if context else "No specific information available"}

                Return JSON with:
                - answer: Direct answer to the user's question
                - recommendations: 3-4 actionable recommendations
                - source: "{context_source}"
                """

            response = await self.llm.ainvoke(
                [
                    SystemMessage(content="Create simple email marketing strategies."),
                    HumanMessage(content=strategy_prompt),
                ]
            )

            try:
                strategy = json.loads(response.content)
            except json.JSONDecodeError:
                strategy = {"strategy": response.content}

            state["final_strategy"] = strategy
            state["iteration_count"] = iteration + 1
            state["current_step"] = "strategy_generated"

            self.logger.info("Strategy generation completed", iteration=iteration + 1)

            return state

        except Exception as e:
            self.logger.error("Strategy generation failed", error=str(e))
            state["error"] = f"Strategy generation failed: {str(e)}"
            return state

    async def _finalize_strategy_node(self, state: AgentState) -> AgentState:
        """Finalize the strategy with summary and next steps."""
        try:
            strategy = state.get("final_strategy", {})

            # Add metadata to final strategy
            finalized_strategy = {
                **strategy,
                "metadata": {
                    "generated_at": time.time(),
                    "agent_version": "1.0",
                    "iterations": state.get("iteration_count", 1),
                    "documents_used": len(state.get("retrieved_documents", [])),
                    "customer_analysis": state.get("customer_data", {}),
                },
            }

            state["final_strategy"] = finalized_strategy
            state["current_step"] = "completed"

            self.logger.info(
                "Strategy finalization completed",
                iterations=state.get("iteration_count", 1),
                documents_used=len(state.get("retrieved_documents", [])),
            )

            return state

        except Exception as e:
            self.logger.error("Strategy finalization failed", error=str(e))
            state["error"] = f"Strategy finalization failed: {str(e)}"
            return state

    async def process_request(
        self,
        message: str,
        session_id: str,
        conversation_history: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a user request and generate email marketing strategy."""
        # Create ONE main trace for everything
        self.current_trace = self.langfuse_client.trace(
            name="email_agent_request",
            session_id=session_id,
            input={"message": message, "session_id": session_id},
        )

        # Create callback handler and associate it with our trace
        self.langfuse_handler = CallbackHandler(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host,
            session_id=session_id,
        )

        # Set the trace manually on the handler
        self.langfuse_handler.trace = self.current_trace

        # Update LLM callbacks for this request
        self.llm.callbacks = [self.langfuse_handler]

        try:
            # Initialize state
            messages = conversation_history or []
            messages.append(
                {"role": "user", "content": message, "timestamp": time.time()}
            )

            initial_state: AgentState = {
                "messages": messages,
                "customer_data": None,
                "strategy_context": None,
                "retrieved_documents": [],
                "current_step": "initialized",
                "iteration_count": 0,
                "final_strategy": None,
                "error": None,
            }

            # Run the agent workflow
            result = await self.graph.ainvoke(initial_state)

            if result.get("error"):
                return {
                    "success": False,
                    "error": result["error"],
                    "session_id": session_id,
                }

            # Prepare response
            response = {
                "success": True,
                "strategy": result["final_strategy"],
                "session_id": session_id,
                "metadata": {
                    "iterations": result.get("iteration_count", 1),
                    "documents_used": len(result.get("retrieved_documents", [])),
                    "processing_steps": result.get("current_step", "unknown"),
                },
                "sources": result.get("retrieved_documents", []),
            }

            self.logger.info(
                "Request processing completed",
                session_id=session_id,
                success=True,
                iterations=result.get("iteration_count", 1),
            )

            # Update the main trace with final results
            self.current_trace.update(output=response)

            return response

        except Exception as e:
            self.logger.error(
                "Request processing failed", session_id=session_id, error=str(e)
            )

            error_response = {
                "success": False,
                "error": f"Request processing failed: {str(e)}",
                "session_id": session_id,
            }

            # Update trace with error
            self.current_trace.update(output=error_response)

            return error_response


# Global agent instance
email_agent: EmailMarketingAgent = EmailMarketingAgent()
