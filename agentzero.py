# Mixture-of-Agents in 50 lines of code
import asyncio
import os
from together import AsyncTogether, Together

client = Together(api_key=("b2ceded2a0405d5f78a6d31e0004fee9f23c073fc2c05926be188188da0570a7"))
async_client = AsyncTogether(api_key=("b2ceded2a0405d5f78a6d31e0004fee9f23c073fc2c05926be188188da0570a7"))

user_prompt = input("Please tell me what you want: ")
reference_models = [
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]
aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability also your task is to tell me updated information and always create well structured data outputs and make a clean list of stuff and try to explain in simple term without effecting the original message.

Responses from models:"""


async def run_llm(model):
    """Run a single LLM call with a reference model."""
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content


async def main():
    results = await asyncio.gather(*[run_llm(model) for model in reference_models])

    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {"role": "system", "content": aggreagator_system_prompt},
            {"role": "user", "content": ",".join(str(element) for element in results)},
        ],
        stream=True,
    )

    for chunk in finalStream:
        # Remove asterisks from the output
        output = chunk.choices[0].delta.content or ""
        cleaned_output = output.replace('*', '')  # Remove asterisks
        print(cleaned_output, end="", flush=True)




asyncio.run(main())