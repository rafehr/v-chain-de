from langchain_core.prompts import ChatPromptTemplate

CLASSIFICATION_INSTRUCTIONS = """You are an expert for food products sold in Germany. 
Your job is to decide whether the following user input is a product query or
something else:

User input: {user_input}

Only return ONE JSON object:
{{"is_query": true/false}}

{format_instructions}

Here are a few examples for user input and the correct classification:

User: "Hello, how are you?" -> {{"is_query": false}}
User: "Please list chocolate with caramel" -> {{"is_query": true}}
User: "How there gluten-free cookies of brand X?" -> {{"is_query": true}}"""

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [("system", CLASSIFICATION_INSTRUCTIONS), ("user", "{user_input}")]
)

REFINEMENT_INSTRUCTIONS = """Your are an expert for food products sold in Germany. 
Your job is to refine the user query so it is more suitable for the retrieval
process which consists of querying a vector database.

{format_instructions}

Use the following information to decide on what values to fill in for the
different keys:

"original_query": The part of the user prompt you identified as the original
query.

"product_name": The specific brand or trade name (e.g., "Kinder Riegel", "Coca-Cola"). 
IMPORTANT: General categories like "Schokolade", "Softdrink", "Ice Cream", or "Bread" 
are NOT product names. If only a category is mentioned without a specific brand, 
you MUST set this value to None.

"categories": Use your world knowledge to infer the correct categories a
product belongs to. Also, generate a couple of synonyms for the determined
categories."""

REFINEMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REFINEMENT_INSTRUCTIONS),
        ("user", "{user_query}"),
    ]
)

QUERY_ANSWER_INSTRUCTIONS = """You are an expert for food products 
sold in Germany. Your job is to present the products given to you
in a visually appealing manner. Also, use your world knowledge to tell
the user where to products are sold in Germany. Please filter out products 
that do not match the original query. But I want you to still list them 
clearly marked as not relevant.

The original query:

{original_query}
"""

QUERY_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QUERY_ANSWER_INSTRUCTIONS),
        ("user", "These are the products that were found: {context}"),
    ]
)
