"""Demo 6: LangChain + LangGraph helpdesk agent with tools, routing, memory and approval.

Run: python demo_06_langgraph_helpdesk.py
Requires GROQ_API_KEY in .env. This educational demo changes local sample JSON only.
"""

import json
import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

load_dotenv()
DB = Path(__file__).parent / "data" / "tickets.json"


def read_all_tickets():
    return json.loads(DB.read_text(encoding="utf-8"))


@tool
def list_tickets() -> list[dict]:
    """Return the current helpdesk tickets and their ID, issue, status and priority."""
    print("  [TOOL] list_tickets")
    return read_all_tickets()


@tool
def get_ticket(ticket_id: str) -> dict:
    """Get a specific helpdesk ticket by its ID, such as T101."""
    print(f"  [TOOL] get_ticket({ticket_id})")
    ticket = next((t for t in read_all_tickets() if t["id"] == ticket_id.upper()), None)
    return ticket or {"error": f"Ticket {ticket_id} not found"}


@tool
def update_ticket_priority(ticket_id: str, priority: Literal["low", "medium", "high"]) -> dict:
    """Change the priority of an open ticket. This action requires human approval in the graph."""
    print(f"  [TOOL] update_ticket_priority({ticket_id}, {priority})")
    tickets = read_all_tickets()
    ticket = next((t for t in tickets if t["id"] == ticket_id.upper()), None)
    if ticket is None:
        return {"updated": False, "error": "Ticket not found"}
    if ticket["status"] != "open":
        return {"updated": False, "error": "Only open tickets can be updated"}
    old = ticket["priority"]
    ticket["priority"] = priority
    DB.write_text(json.dumps(tickets, indent=2) + "\n", encoding="utf-8")
    return {"updated": True, "id": ticket["id"], "previous_priority": old, "new_priority": priority}


TOOLS = [list_tickets, get_ticket, update_ticket_priority]
WRITE_TOOL = "update_ticket_priority"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    approval_granted: bool


def make_graph():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY in the .env file")

    model = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), temperature=0)
    model_with_tools = model.bind_tools(TOOLS)
    system = SystemMessage(content=(
        "You are a concise IT helpdesk assistant. Use tools for ticket facts. "
        "Before updating a ticket, read it to verify it exists and is open. "
        "Do not claim any update succeeded without a successful tool response. "
        "The graph handles human approval of write actions."
    ))

    def agent_node(state: AgentState):
        print("[NODE] agent (LangChain ChatGroq)")
        reply = model_with_tools.invoke([system, *state["messages"]])
        return {"messages": [reply], "approval_granted": False}

    def decide_next(state: AgentState) -> Literal["approval", "tools", "end"]:
        latest = state["messages"][-1]
        if not isinstance(latest, AIMessage) or not latest.tool_calls:
            print("[ROUTE] agent -> END")
            return "end"
        if any(call["name"] == WRITE_TOOL for call in latest.tool_calls):
            print("[ROUTE] agent -> approval")
            return "approval"
        print("[ROUTE] agent -> tools")
        return "tools"

    def approval_node(state: AgentState):
        print("[NODE] approval (LangGraph interrupt)")
        calls = state["messages"][-1].tool_calls
        # interrupt suspends execution; the CLI resumes it with Command(resume=...).
        response = interrupt({
            "question": "Approve the following tool requests?",
            "pending_calls": [{"name": c["name"], "args": c["args"]} for c in calls],
        })
        approved = str(response).strip().upper() == "YES"
        if approved:
            return {"approval_granted": True}
        # Every requested tool call must receive a ToolMessage, even when refused.
        denied = [ToolMessage(content="Action not executed: human approval denied.", tool_call_id=c["id"])
                  for c in calls]
        return {"approval_granted": False, "messages": denied}

    def after_approval(state: AgentState) -> Literal["tools", "agent"]:
        return "tools" if state["approval_granted"] else "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("approval", approval_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", decide_next,
                                  {"approval": "approval", "tools": "tools", "end": END})
    builder.add_conditional_edges("approval", after_approval,
                                  {"tools": "tools", "agent": "agent"})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=InMemorySaver())


def print_reply(state):
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and not message.tool_calls:
            print("\nAGENT:", message.content)
            return


def main():
    graph = make_graph()
    # The same thread_id preserves message state while this Python process remains alive.
    config = {"configurable": {"thread_id": "classroom-demo-1"}, "recursion_limit": 20}
    print("\nLangGraph Helpdesk Demo. Type 'exit' to quit.")
    print("Examples: 'List tickets'; 'What is priority of T101?';")
    print("          'Set T101 priority to high'; 'What did we just do?'\n")

    while True:
        user_text = input("YOU: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue
        state = graph.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)
        # An interrupted graph returns pending interruptions in __interrupt__.
        while "__interrupt__" in state:
            for pause in state["__interrupt__"]:
                request = pause.value
                print("\nAPPROVAL REQUEST:", json.dumps(request, indent=2))
            decision = input("Type YES to approve or NO to reject: ").strip()
            state = graph.invoke(Command(resume=decision), config=config)
        print_reply(state)
        print()


if __name__ == "__main__":
    main()
