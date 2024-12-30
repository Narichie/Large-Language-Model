# Large Language Model ( Data Science Profiles and Interview Preparation)

# Introduction
The goal of this project is to connect with a large language model (LLM) to extract actionable insights on preparing for data science interviews and understanding relevant profiles.
# GOAL
1.	Create a connection to an LLM for knowledge extraction and HuggingFace.
2.	Prepare prompts to investigate:
*	Specific steps to prepare for a data science interview.
*	The top five questions recruiters commonly ask entry-level data scientists.
*	Descriptive profiles for a set of three data scientists.

# Necessary installations and imports

This project discusses an advanced prompt engineering technique involving the Llama2-13B model, which is noteworthy due to its training on 13 billion parameters. The process begins with the installation of the GPU-compatible Llama-cpp-python library, followed by the configuration of its parameters. Additionally, the huggingface_hub was installed; this platform serves as a repository for a diverse range of large language models (LLMs). Finally, both the huggingface_hub and the Llama model were imported for use in the project.

# Model Development
The following section outlines the model configuration and connection process. A specific path has been established to connect Hugging Face to the LLM in Llama. This includes specifying the base name and the repository ID, which correspond to the model name and path. Lastly, the filename is indicated in relation to the model's base name.

# Model configuration
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
model_basename = "llama-2-13b-chat.Q5_K_M.gguf"
model_path = hf_hub_download(
    repo_id=model_name_or_path,
    filename=model_basename
    )

The connection was subsequently defined by various factors, including the number of threads corresponding to the CPU cores, the number of batches processed, the structure of the layers, and the size of the context window.

lcpp_llm = Llama(
        model_path=model_path,
        n_threads=2,  
        n_batch=512,  
        n_gpu_layers=43,
        n_ctx=4096,  
)

Following the deployment of the technical configuration, a tailored functionality was created by specifying the User prompt. This setup facilitates the analysis of the prompt and sends it to the LLM (Large Language Model) to obtain a response. The function is defined as follows:

```python
def generate_llama_response(user_prompt):
```
To begin with, a system message is established to provide the language model with its initial instructions. Following this, the user prompt is combined with the system message to formulate the complete prompt. This process can be represented as: prompt = f"{user_prompt}\n{system_message}”.

The next step involves configuring the response by adjusting key parameters such as max tokens, temperature, top P, top K, repeat penalty, as well as defining the stop and echo settings. Each of these settings plays a crucial role in shaping the output's quality and style.


 # Model Check

user_prompt = "Provide the top ten steps to prepare effectively for an entry-level data science interview"
response_interview_steps = generate_llama_response(user_prompt)
print(response_interview_steps)


In my approach, I established clear outcomes by defining specific output expectations. I created well-structured and concise prompts to facilitate effective interactions with large language models (LLMs). For instance, I utilized prompts such as, "List 10 steps to prepare for a data science interview."

To formulate these prompts, I incorporated relevant terminology and frameworks associated with data science. This included specific phrases like "specific tools and technologies" and "attributes of senior data scientists," ensuring that the prompts aligned with industry standards.

After generating responses from the LLM, I compared them against my predefined objectives. This involved verifying the completeness of the provided steps and evaluating their coherence and relevance to ensure they met the expected criteria.

# Follow me and let's connect
www.linkedin.com/in/richard-asiamah-007b20242
