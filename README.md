# LangChain Cookbook 👨‍🍳👩‍🍳

This cookbook provides an introductory understanding of the components and use cases of LangChain, a framework for developing applications powered by language models. It aims to explain the concepts and features of LangChain through ELI5 examples and code snippets. For detailed use cases, please refer to Part 2 (coming soon).

## Table of Contents

1. [Links](#links)
2. [What is LangChain?](#what-is-langchain)
3. [Why LangChain?](#why-langchain)
4. [LangChain Components](#langchain-components)
   - [Schema](#schema)
     - [Text](#text)
     - [Chat Messages](#chat-messages)
     - [Documents](#documents)
   - [Models](#models)
     - [Language Model](#language-model)
     - [Chat Model](#chat-model)
     - [Text Embedding Model](#text-embedding-model)
   - [Prompts](#prompts)
     - [Prompt](#prompt)
     - [Prompt Template](#prompt-template)
     - [Example Selectors](#example-selectors)
5. [Getting Started](#getting-started)
6. [Contributing](#contributing)
7. [License](#license)

## Links
- [LC Conceptual Documentation](https://link-to-conceptual-docs)
- [LC Python Documentation](https://link-to-python-docs)
- [LC Javascript/Typescript Documentation](https://link-to-js-docs)
- [LC Discord](https://discord-link)
- [www.langchain.com](https://www.langchain.com)
- [LC Twitter](https://twitter.com/langchain)

## What is LangChain?
LangChain is a framework for developing applications powered by language models. It simplifies the complex aspects of working and building with AI models by providing two main features:

1. Integration: LangChain enables the integration of external data, such as files, other applications, and API data, into your language models (LLMs).
2. Agency: LangChain allows LLMs to interact with their environment by making decisions. It enables LLMs to assist in decision-making processes and helps determine the next action to take.

In summary, LangChain makes working with AI models easier by providing a flexible framework for integrating external data and enabling decision-making capabilities.

## Why LangChain?
There are several reasons why you should consider using LangChain:

1. Components: LangChain offers an easy way to swap out abstractions and components required to work with language models.
2. Customized Chains: LangChain provides out-of-the-box support for using and customizing "chains," which are a series of actions strung together.
3. Speed: The LangChain team ships updates and new features at a fast pace, ensuring that you have access to the latest advancements in LLMs.
4. Community: LangChain has a vibrant community with active support on Discord, meetups, hackathons, and more.

While working with simple LLMs (text-in, text-out) may be straightforward initially, LangChain becomes valuable as your applications become more complex and encounter friction points that LangChain can help resolve.

**Note:** This cookbook provides a curated set of examples and information to get you started quickly. For more details, please refer to the LangChain Conceptual Documentation.

## LangChain Components

### Schema
The Schema component forms the foundation of working with LLMs.

#### Text
Text is the natural language way to interact with LLMs. It typically involves working with simple strings that can grow in complexity as your applications evolve.

Example:
```python
# You'll be

working with simple strings (that'll soon grow in complexity!)
my_text = "What day comes after Friday?"
```

#### Chat Messages
Chat Messages allow you to specify different types of messages, such as System, Human, and AI, when interacting with a Chat Model. These messages provide contextual information to guide the AI's behavior.

Example:
```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

chat = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
chat(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
        HumanMessage(content="I like tomatoes, what should I eat?")
    ]
)
# Output: AIMessage(content='You could eat a caprese salad with fresh mozzarella and basil.', additional_kwargs={}, example=False)
```

You can pass more chat history with responses from the AI to have a conversation.

Example:
```python
chat(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"),
        HumanMessage(content="I like the beaches, where should I go?"),
        AIMessage(content="You should go to Nice, France"),
        HumanMessage(content="What else should I do when I'm there?")
    ]
)
# Output: AIMessage(content='While in Nice, you can also explore the charming Old Town and enjoy some delicious Mediterranean cuisine.', additional_kwargs={}, example=False)
```

#### Documents
Documents are objects that hold a piece of text and metadata. They can be used to store text with additional information for reference.

Example:
```python
from langchain.schema import Document

document = Document(
    page_content="This is my document. It is full of text that I've gathered from other places",
    metadata={
        'my_document_id' : 234234,
        'my_document_source' : "The LangChain Papers",
        'my_document_create_time' : 1680013019
    }
)
```

### Models
The Models component provides the interface to the AI brains.

#### Language Model
A Language Model is a model that takes text as input and produces text as output.

Example:
```python
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-ada-001", openai_api_key=openai_api_key)
llm("What day comes after Friday?")
# Output: 'Saturday'
```

#### Chat Model
A Chat Model takes a series of messages as input and returns a message as output. It enables interactive conversations with the AI.

Example:
```python
chat = ChatOpenAI(temperature=1, openai_api_key=openai_api_key)
chat(
    [
        SystemMessage(content="You are an unhelpful AI bot that makes a joke at whatever the user says"),
        HumanMessage(content="I would like to go to New York, how should I do this?")
    ]
)
# Output: AIMessage(content="Well, have you considered using your teleportation powers? Oh, wait, you don't have any? I guess you'll have to settle for an airplane or a really fast pogo stick. Good luck with that!", additional_kwargs={}, example=False)
```

#### Text Embedding Model
A Text Embedding Model converts text into a vector representation, capturing the semantic meaning of the text. It is often used for comparing and analyzing textual data.

Example:
```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
text = "Hi! It's time for the beach"
text

_embedding = embeddings.embed_query(text)
print(f"Your embedding is length {len(text_embedding)}")
print(f"Here's a sample: {text_embedding[:5]}...")
```

### Prompts
Prompts are text instructions provided to the underlying model to guide its response.

#### Prompt
A Prompt is the text passed to the model as input.

Example:
```python
llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)

prompt = """
Today is Monday, tomorrow is Wednesday.

What is wrong with that statement?
"""

llm(prompt)
# Output: '\nThe statement is incorrect because tomorrow is Tuesday, not Wednesday.'
```

#### Prompt Template
A Prompt Template is an object that helps create prompts based on a combination of user input, other non-static information, and a fixed template string. It allows for dynamic prompts generation.

Example:
```python
llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)

template = """
I really want to travel to {location}. What should I do there?

Respond in one short sentence
"""

prompt = PromptTemplate(
    input_variables=["location"],
    template=template,
)

final_prompt = prompt.format(location='Rome')

print(f"Final Prompt: {final_prompt}")
print("-----------")
print(f"LLM Output: {llm(final_prompt)}")
```

#### Example Selectors
Example Selectors provide a way to select examples from a series of predefined examples to incorporate contextual information into the prompt. This is useful when dealing with nuanced tasks or a large list of examples.

Example:
```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Example Input: {input}\nExample Output: {output}",
)

examples = [
    {"input": "pirate", "output": "ship"},
    {"input": "pilot", "output": "plane"},
    {"input": "driver", "output": "car"},
    {"input": "tree", "output": "ground"},
    {"input": "bird", "output": "nest"},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(openai_api_key=openai_api_key),
    FAISS,
    ...
)
```

## Getting Started
To get started with LangChain, follow these steps:

1. Install the LangChain package:
   ```
   pip install langchain
   ```

2. Import the necessary components and models:
   ```python
   from langchain.chat_models import ChatOpenAI
   from langchain.schema import HumanMessage, SystemMessage, AIMessage
   from langchain.llms import OpenAI
   from langchain.embeddings import OpenAIEmbeddings
   from langchain.prompts import PromptTemplate
   ```

3. Set up your API key:
   ```python
   openai_api_key = 'YourAPIKey'
   ```

4. Start using LangChain components and models in your applications. Refer to the examples and code snippets provided in this README for guidance.

## Contributing
Contributions to the LangChain project are welcome! If you have any improvements, bug fixes, or new features to add, please submit a pull request on the LangChain GitHub repository.

## License
The Lang

Chain framework is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](https://link-to-license-file) file for more information.
