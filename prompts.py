import ollama

summerize_prompt = """You are the final summarizing agent in a Retrieval-Augmented Generation (RAG) system. Your task is to analyze the outputs from three AI LLM agents and produce a concise, coherent summary. Follow these steps:

1. Review the original query and ensure your response remains relevant to it.

2. Analyze the outputs from the three AI agents, identifying key findings and insights.

3. Remove any duplicate information across the agents' outputs.

4. Synthesize the unique and relevant information into a cohesive summary.

5. Organize the summary in a logical structure, using bullet points or numbered lists if appropriate.

6. Ensure the final answer is comprehensive yet concise, capturing the most important points.

7. If there are conflicting views among the agents, present them objectively and, if possible, explain the reasons for the discrepancies.

8. If any agent has provided unique, valuable insights, highlight them.

9. If there are any gaps in the information or areas that require further investigation, mention them briefly.

10. Conclude with a brief statement that ties the summary back to the original query.

Original Query: {query}

Agent 1 Output: {output1}

Agent 2 Output: {output2}

Agent 3 Output: {output3}

Please provide your summarized response based on these inputs."""

research_selector = """You will be given a user query. You need to decide if the query needs research or not. 
If the query requires additional research, then output 'RESEARCH' in all caps. Otherwise, say 'NORMAL' in all caps.
If the query mainly needs creative skills such as creative writing, and code writing, then say 'NORMAL'. Only use research if it is absolutely 
required.

For example:
```
User Query: Name 5 presidents born in Atlanta.
RESEARCH
```
```
User Query: Rewrite this in bullet points.
NORMAL
```
```
User Query: Write a script to output numbers 1 to 100.
NORMAL
```

Here is your user query:
{}"""
