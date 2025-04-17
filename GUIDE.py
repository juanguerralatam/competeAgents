import getpass
import os
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from typing import List, Optional, Literal
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, trim_messages
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent




def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

_set_if_undefined("DEEPSEEK_API_KEY")

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=10, 
)

def analyse_vendor(llm, question):
    return llm.invoke(question)

@tool
def analyse_vendor_tool(question: str) -> str:
    """Use this tool to answer decide how many units to buy."""
    return analyse_vendor(llm, question)

def analyse_demand(llm, question):
    return llm.invoke(question)

@tool
def analyse_demand_tool(question: str) -> str:
    """Use this tool to decide how many units to produce."""
    return analyse_vendor(llm, question)

class State(MessagesState):
    next: str
    visited: List[str]

def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following companies: {members}. Your job is to coordinate their work.\n"
        " Rules:\n"
        " 1. You must route to EACH company EXACTLY ONCE before finishing\n"
        " 2. Track which companies have already participated\n"
        " 3. Only choose FINISH when ALL companies have participated\n"
        " 4. Choose the next company that hasn't participated yet"
    )

    # Middleware router that ensures all members are visited once
    def supervisor_node(state: State) -> Command:
        # Initialize visited list
        if "visited" not in state:
            state["visited"] = []

        messages = [
            {
                "role": "system",
                "content": system_prompt + f"\nAlready visited: {state['visited']}",
            },
        ] + state["messages"]

        # Directly invoke LLM and parse 'next' from its JSON-like reply
        response = llm.invoke(messages)
        goto = response["next"]

        # If all members have been visited, then we can finish
        if len(state["visited"]) >= len(members):
            goto = "FINISH"

        if goto == "FINISH":
            goto = END
        else:
            state["visited"] = state["visited"] + [goto]

        return Command(goto=goto, update={"next": goto, "visited": state["visited"]})

    # End of make_supervisor_node
    return supervisor_node


companyA_agent = create_react_agent(llm, tools=[analyse_vendor_tool, analyse_demand_tool])
def companyA_node(state: State) -> Command[Literal["supervisor"]]:
    result = companyA_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="companyA")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

companyB_agent = create_react_agent(llm, tools=[analyse_vendor_tool, analyse_demand_tool])
def companyB_node(state: State) -> Command[Literal["supervisor"]]:
    result = companyB_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="companyB")
            ]
        },
        goto="supervisor",
    )

research_supervisor_node = make_supervisor_node(llm, ["companyA", "companyB"])

research_builder = StateGraph(State)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("companyA", companyA_node)
research_builder.add_node("companyB", companyB_node)
research_builder.add_edge(START, "supervisor")

research_graph = research_builder.compile()

initial_message = HumanMessage(content="how many OLT will you buy")
for s in research_graph.stream(
    {"messages": [initial_message]},
    {"recursion_limit": 10},
):
    print(s)
    print("---")

ISP_agent = create_react_agent(
    llm,tools=[analyse_vendor_tool, analyse_demand_tool], 
    prompt=(
        "You are ISP and decide how many OLT buy"
    ),
)

def ISP_A_node(state: State) -> Command[Literal["supervisor"]]:
    result = ISP_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="ISP_A")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

def ISP_B_node(state: State) -> Command[Literal["supervisor"]]:
    result = ISP_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="ISP_B")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

companyA_supervisor_node = make_supervisor_node(
    llm, ["ISP_A","ISP_B"]
)

paper_writing_builder = StateGraph(State)
paper_writing_builder.add_node("supervisor", companyA_supervisor_node)
paper_writing_builder.add_node("ISP_A", ISP_A_node)
paper_writing_builder.add_node("ISP_B", ISP_B_node)

paper_writing_builder.add_edge(START, "supervisor")
paper_writing_graph = paper_writing_builder.compile()

seen_states = set()
for s in paper_writing_graph.stream(
    {
        "messages": [
            (
                "user",
                "How many OLt will you buy",
            )
        ]
    },
    {"recursion_limit": 100},
):
    print(s)
    print("---")

from langchain_core.messages import BaseMessage
teams_supervisor_node = make_supervisor_node(llm, ["call_companyA", "call_companyB"])

def call_companyA(state: State) -> Command[Literal["supervisor"]]:
    response = research_graph.invoke({"messages": state["messages"][-1]})
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="companyA"
                )
            ]
        },
        goto="supervisor",
    )


def call_companyB(state: State) -> Command[Literal["supervisor"]]:
    response = paper_writing_graph.invoke({"messages": state["messages"][-1]})
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="companyB"
                )
            ]
        },
        goto="supervisor",
    )

# Define the graph.
super_builder = StateGraph(State)
super_builder.add_node("supervisor", teams_supervisor_node)
super_builder.add_node("call_companyA", call_companyA)
super_builder.add_node("call_companyB", call_companyB)

super_builder.add_edge(START, "supervisor")
super_graph = super_builder.compile()

for s in super_graph.stream(
    {
        "messages": [
            ("user", "You are in interent industry decide how many OLT buy")
        ],
    },
    {"recursion_limit": 150},
):
    print(s)
    print("---")



