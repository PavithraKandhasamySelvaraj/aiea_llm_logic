import os
import re
from pyswip import Prolog
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict

api_key = os.getenv("OPENAI_API_KEY")
client = ChatOpenAI(model="gpt-4o", api_key=api_key)

class State(TypedDict):
    question: str
    classifier: str
    context: str
    csp_response: str
    logic_response: str
    result_interpreted: str


query_format = """
 Format the output exactly as:
            FACTS:
            <Prolog facts>
            <Prolog rules>

            QUERY:
            ?- <Prolog query>.

            Do not include any additional explanations or comments or headings.

"""

# used these problem and goal to test both CSP, logic program and error condition
nl_problem_goal = """
Monica, Rachel and Phoebe are Female
Ross, Chandler and Joey are male
Ross, Rachel, Monica, Chandler and Joey are all friends

Is the below following statements is true, false or uncertain?
Amy is rachel's sister
"""
# Chandler and Monica are not only married but also best friends which is similar to the relationship between phoebe and mike.
# phoebe and mike know each other from childhood

document_loader = TextLoader("prolog_kb_context.txt")
documnets = document_loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
document_split = splitter.split_documents(documnets)

openAIEmbeddings = OpenAIEmbeddings()
db = Chroma.from_documents(document_split, openAIEmbeddings)

db = db.as_retriever()
documents = db.get_relevant_documents(nl_problem_goal)
context = "\n".join([d.page_content for d in documents])


# classifies the problem as logical programming or CSP
def problem_classifier(state: State) -> State:
    nl_problem = state["question"]
    chat_messages = [
        SystemMessage(
            content="You are a Symbolic solver expert who is experienced in classifying natural language problems into their specific Symbolic formulations."
        ),
        HumanMessage(
            content=f"""
                Given the following natural language problem, read through the problem statement carefully and 
                    identify whether it is a logic programming problem or Constraint satisfaction problem. 

                    If it is a logic programming problem output "Logic Program"
                    if it is a Constraint satisfaction problem output "Constraint Satisfaction Problem"
                    if you are not sure about the classification, output "Cannot classify"
                    Do not include any additional explanations or comments.

                    Problem: {nl_problem}
                """
        ),
    ]

    llm_response = client.invoke(chat_messages).content
    state["classifier"] = llm_response.strip()
    return state


# MImic CSP solver - problem formulator
def constraint_solver(state: State) -> State:
    nl_problem_goal = state["question"]
    chat_messages = [
        SystemMessage(
            content="You are an expert in solving constraint satisfaction problems. Your task is to analyze the given problem and provide a structured solution."
        ),
        HumanMessage(
            content=f"Solve this constraint satisfaction problem by identifying the correct option and give the output {nl_problem_goal}"
        ),
    ]

    csp_response = client.invoke(chat_messages).content

    final_messages = [
        SystemMessage(
            content="From the response below, output only the correct option (e.g., either 'A', 'B' or 'C'):"
        ),
        HumanMessage(content=csp_response),
    ]

    final_output = client.invoke(final_messages).content.strip()

    state["csp_response"] = final_output
    return state


# mimic logic programming solver - problem formulator
def logic_solver(state: State) -> State:
    nl_problem = state["question"]
    chat_messages = [
        SystemMessage(
            content="""You are a Prolog expert who is experienced in converting natural language problems into Prolog facts, rules and queries. 
            You should create valid Prolog syntax that can be executed directly.
            """
        ),
        HumanMessage(
            content=f"""
            Given the following natural language problem, convert it into Prolog facts, rules, and query to check if the claim is true, false, or uncertain.
            Use the following knowledge base as a reference for additional context: {context}
            
            {query_format}
            
            Important:
            1. Make sure all facts and rules are valid Prolog syntax
            2. Use proper Prolog predicates and variables
            3. Ensure the query is properly formatted with a period at the end
            4. Keep the facts and rules simple and direct
            5. For marriage relationships, use husband(X,Y) where X is the husband of Y
            6. Generate queries that exactly match the facts in the knowledge base
            
            Problem:
            {nl_problem}
            """
        ),
    ]

    llm_response = client.invoke(chat_messages).content
    print("####Generated Prolog code:####")
    # print(llm_response)
    facts, query = symbolic_formulator(llm_response)
    prolog_kb_creater(facts)
    result = classify_result("output.pl", query)

    if result == "uncertain":
        r_facts, r_query, result, attempts = self_refinement(nl_problem_goal, llm_response, result, max_count=6)

        if r_facts:
            print(f"\nRefined result", result)
        else:
            print(f"Refinement failed")
    else:
        print(f"\nrefined result", result)
        print(f"Query:", query)

    state["logic_response"] = result
    return state


# conversion of the llm output to prolog KB - mimic symbolic reasonor
def symbolic_formulator(llm_response):
    llm_response = str(llm_response)
    # Extract facts and query using more robust regex patterns
    facts_match = re.search(r"FACTS:\s*(.*?)(?:QUERY:|$)", llm_response, re.DOTALL)
    query_match = re.search(r"QUERY:\s*\?\-\s*([^\.\n]+)", llm_response)

    if not facts_match or not query_match:
        raise ValueError("Could not parse facts or query from response")

    facts = facts_match.group(1).strip()
    query = query_match.group(1).strip()

    # Ensure query ends with a period
    if not query.endswith("."):
        query += "."

    return facts, query


def prolog_kb_creater(facts, filename="output.pl"):
    with open(filename, "w") as file:
        file.write(facts)

# mimic result interpreter
def classify_result(fact_file, query):
    prolog = Prolog()
    try:
        print("###classify_result", fact_file, query)
        prolog.consult(fact_file)
        result = list(prolog.query(query))

        if result:
            return "true"
        else:
            return "false"
    except Exception as e:
        print(f"Error in Prolog execution: {str(e)}")
        return "uncertain"


def result_interpreter(state: State) -> State:
    result = state["logic_response"]
    if result == "true":
        final_result =  "The statement is true."
    elif result == "false":
        final_result =  "The statement is false."
    elif result == "uncertain":
        final_result =  "The answer cannot be found with the provided information"
    else:
        final_result =  "Unknown result/error"

    state["result_interpreted"] = final_result
    return state


# mimic self refinement process
def self_refinement(nl_problem, original_response, query_result, max_count=3):
    if query_result != "uncertain":
        return None, None, query_result, 0

    last_response = original_response
    refine_counter = 0
    while refine_counter < max_count:
        refine_counter += 1

        chat_messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(
                content=f"""
                You previously gave the below symbolic formulation as response for the problem:
                problem: {nl_problem}
                response : {last_response}

                But, the symbolic solver was not able to produce the correct result and outputted 'uncertain'.
                Please revise your Prolog facts, rules, and query so that the problem can be correctly resolved using logic programming.
                Use the following knowledge base as a reference for additional context: {context}
                {query_format}

                Do not include any additional explanations or comments or headings.
                """
            ),
        ]
        print(HumanMessage(content=chat_messages[1].content))
        refined_response = client.invoke(chat_messages).content
        last_response = refined_response

        print(f"\n Refinement Attempt completed ", refine_counter)

        facts, query = symbolic_formulator(refined_response)
        print(refined_response)

        prolog_kb_creater(facts)

        query_result = classify_result("output.pl", query)

        if query_result != "uncertain":
            return facts, query, query_result, refine_counter

    return None, None, "uncertain", refine_counter


builder = StateGraph(State)

builder.add_node("problem_classifier", problem_classifier)
builder.add_node("constraint_solver", constraint_solver)
builder.add_node("logic_solver", logic_solver)
builder.add_node("result_interpreter", result_interpreter)

builder.set_entry_point("problem_classifier")
builder.add_conditional_edges(
    "problem_classifier",
    lambda state: state["classifier"],
    {
        "Logic Program": "logic_solver",
        "Constraint Satisfaction Problem": "constraint_solver",
        "Cannot classify": END
    }
)
builder.add_edge("logic_solver", "result_interpreter")


graph = builder.compile()

if __name__ == "__main__":
    input_state = {"question":'''Monica, Rachel and Phoebe are Female
Ross, Chandler and Joey are male
Ross, Rachel, Monica, Chandler and Joey are all friends

Is the below following statements is true, false or uncertain?
Amy is rachel's sister'''}

#     input_state = {"question":'''In an antique car show, there are 3 vehicles, a tractor, a convertible, and a minivan.
# The tractor is the second newest vehicle.
# The minivan is newer than the convertible.

# which the following statements is true?
# A) The tractor is the oldest vehicle.
# B) The convertible is the oldest vehicle.
# C) The minivan is the oldest vehicle.'''}

    result = graph.invoke(input_state)
    print(result["result_interpreted"]) 