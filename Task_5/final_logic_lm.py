import os
import re
from pyswip import Prolog

api_key = os.getenv("OPENAI_API_KEY")
from openai import OpenAI
client = OpenAI(
    api_key=api_key
) 

# used these problem and goal to test both CSP, logic program and error condition
nl_problem_goal = """
In an antique car show, there are 3 vehicles, a tractor, a convertible, and a minivan.
The tractor is the second newest vehicle.
The minivan is newer than the convertible.

which the following statements is true?
A) The tractor is the oldest vehicle.
B) The convertible is the oldest vehicle.
C) The minivan is the oldest vehicle.
"""

# """
# metals conduct electricity.
# insulators don't conduct electricity.
# if something is made of iron, then it is a metal.
# nails are made of iron.

# is the following statement true, false, or uncertain?
# nails cannot conduct electricity.
# """ 

# """
# Based on the context below, is the following statement true, false, or uncertain?
# All birds can fly. Penguins are birds. Penguins cannot fly.
# Statement: Penguins can fly.
# Do not provide any explanation, just the answer.
# """ 

# """
# A man can either be a doctor or an engineer.
# Google is a search engine.
# Only birds can fly.
# Laptops are portable computers.
# """
 
# classifies the problem as logical programming or CSP
def problem_classifier(nl_problem_goal):
    text_prompt = f"""
    You are a Symbolic solver expert who is experienced in classifying natural language problems into their specific Symbolic formulations.

    Given the following natural language problem, read through the problem statement carefully and 
    identify whether it is a logic programming problem or Constraint satisfaction problem. 

    If it is a logic programming problem output "Logic Program"

    if it is a Constraint satisfaction problem output "Constraint Satisfaction Problem"

    if you are not sure about the classification, output "Cannot classify"

    Do not include any additional explanations or comments.

    Problem:
    {nl_problem_goal}
    """

    llm_response = client.responses.create(
        model="gpt-4o",
        input=text_prompt
    ).output_text
    return llm_response

# MImic CSP solver - problem formulator
def constraint_solver(nl_problem_goal):
    csp_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in solving constraint satisfaction problems. Your task is to analyze the given problem and provide a structured solution."},
            {"role": "user", "content": f"Solve this constraint satisfaction problem by identifying the correct option and give the output {nl_problem_goal}"}
        ]
    )
    csp_result = csp_response.choices[0].message.content
    final_output = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "From the response below, output only the correct option (e.g., either 'A', 'B' or 'C'):\n"},
            {"role": "user","content": csp_result}
        ]
    )
    answer = final_output.choices[0].message.content.strip()
    return answer

# mimic logic programming solver - problem formulator
def logic_solver(nl_problem):
    text_prompt = f"""
        You are a Prolog expert who is experienced in converting natural language problems into Prolog facts, rules and queries.

        Given the following natural language problem, convert it into Prolog facts, rules, and  query to check if the claim is true, false, or uncertain.

        Format the output exactly as:
        FACTS:
        <Prolog facts and rules>

        QUERY:
        ?- <Prolog query>.

        Do not include any additional explanations or comments or headings.

        Problem:
        {nl_problem}
        """

    llm_response = client.responses.create(
        model="gpt-4o",
        input=text_prompt
    ).output_text
    return llm_response

# conversion of the llm output to prolog KB - mimic symbolic reasonor
def symbolic_formulator(llm_response):
    llm_response = str(llm_response)
    facts = re.search(r"FACTS:\s*(.*?)QUERY:", llm_response, re.DOTALL).group(1).strip()
    query = re.search(r"QUERY:\s*\?\-\s*([^\.\n]+)", llm_response).group(1).strip()
    return facts, query


def prolog_kb_creater(facts, filename="output.pl"):
    with open(filename, "w") as file:
        file.write(facts)

# mimic result interpreter
def classify_result(fact_file, query):
    prolog = Prolog()
    try:
        prolog.consult(fact_file)
        result = list(prolog.query(query))
        if result:
            return "true"
        else:
            return "false"
    except Exception:
        return "uncertain"

def result_interpreter(result):
    if result == "true":
        return "The statement is true."
    elif result == "false":
        return "The statement is false."
    elif result == "uncertain":
        return "The answer cannot be found with the provided information"
    else:
        return "Unknown result/error"

# mimic self refinement process
def self_refinement(nl_problem, original_response, query_result, max_count=3):
    if query_result != "uncertain":
        return None, None, query_result, 0

    last_response = original_response
    refine_counter = 0
    while refine_counter < max_count:
        refine_counter += 1

        feedback_prompt = f"""
        You previously gave the below symbolic formulation as response for the problem:
        problem: {nl_problem}
        response : {last_response}

        But, the symbolic solver was not able to produce the correct result and outputted 'uncertain'.

        Please revise your Prolog facts, rules, and query so that the problem can be correctly resolved using logic programming.

        Format the output exactly as:
        FACTS:
        <Prolog facts and rules>

        QUERY:
        ?- <Prolog query>.

        Do not include any additional explanations or comments or headings.
        """

        refined_response = client.responses.create(
            model="gpt-4o",
            input=feedback_prompt
        ).output_text

        last_response = refined_response
        
        print(f"\n Refinement Attempt completed ", refine_counter)

        # refined_response = logic_solver(nl_problem)
        facts, query = symbolic_formulator(refined_response)
        prolog_kb_creater(facts)
        query_result = classify_result("output.pl", query)

        if query_result != "uncertain":
            return facts, query, query_result, refine_counter

    return None, None, "uncertain", refine_counter

# main function that call the functions for classifying problem, solver, result interpreter and refinement
def problem_formulator(nl_problem_goal, response):
    if response.strip() == "Logic Program":
        llm_output = logic_solver(nl_problem_goal)
        facts, query = symbolic_formulator(llm_output)
        prolog_kb_creater(facts)
        result = classify_result("output.pl", query)

        if result == "uncertain":
            r_facts, r_query, result, attempts = self_refinement(nl_problem_goal, llm_output, result, max_count=6)
            
            if r_facts:
                print(f"\nRefined result", result)
                print(f"Answer:", result_interpreter(result))
            else:
                print(f"Refinement failed")
        else:
            print(f"\nrefined result", result)
            print(f"Query:", query)
            print(f"Answer:", result_interpreter(result))
    elif response.strip() == "Constraint Satisfaction Problem":
        result = constraint_solver(nl_problem_goal)
        print(nl_problem_goal)
        print(f"\nAnswer:", result)
    elif response.strip() == "Cannot classify":
        print("The problem could not be classified")
        return None
    return result
        

if __name__ == "__main__":
    response = problem_classifier(nl_problem_goal)
    problem_formulator(nl_problem_goal, response)
