"""
In-Context Learning (ICL) Tutorial Script
==========================================

This script demonstrates the main logic of using Large Language Models (LLMs)
for prediction tasks with and without In-Context Learning (ICL).

ICL is a technique where you provide examples (demonstrations) to the LLM
within the prompt, helping it understand the task pattern and make better predictions.

Reference paper: https://arxiv.org/abs/2509.17552

Author: Based on the LLMxFM_ICRL project
"""

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Note: torch or transformers not installed. Install with:")
    print("  pip install -r requirements_icl.txt")
    print("")


def _get_model_device(model):
    """Helper function to get the device of a model (handles multi-device case)."""
    if hasattr(model, 'device'):
        return model.device
    return next(model.parameters()).device


# ==============================================================================
# STEP 1: Load the LLM Model
# ==============================================================================

def load_llm_model(model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
                   device: str = "auto",
                   load_in_4bit: bool = False):
    """
    Download and load a pre-trained LLM model from Hugging Face.
    
    Args:
        model_name: The Hugging Face model identifier (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")
        device: Device mapping strategy. Use "auto" for automatic GPU/CPU selection.
        load_in_4bit: Whether to load the model in 4-bit quantization (saves memory).
    
    Returns:
        model: The loaded language model
        tokenizer: The corresponding tokenizer
    
    Notes:
        - Uses bfloat16 precision which requires modern GPUs (e.g., Ampere or newer).
          For older GPUs, consider using torch_dtype=torch.float16.
        - Uses left padding which is required for causal LMs during generation.
        - With device_map="auto", model may be split across multiple devices.
    
    Example:
        >>> model, tokenizer = load_llm_model("meta-llama/Meta-Llama-3.1-8B-Instruct")
    """
    print(f"Loading model: {model_name}")
    
    # Load the tokenizer (converts text to tokens and vice versa)
    # Left padding is required for causal language models during generation
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    
    # Set pad token (required for batch processing)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model with bfloat16 precision (requires modern GPUs)
    if load_in_4bit:
        # 4-bit quantization for memory efficiency (requires bitsandbytes library)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16
        )
    
    # Set pad token id in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Get device info for display (handle multi-device case)
    if hasattr(model, 'hf_device_map'):
        device_info = f"devices: {set(model.hf_device_map.values())}"
    else:
        device_info = f"device: {next(model.parameters()).device}"
    
    print(f"Model loaded successfully on {device_info}")
    return model, tokenizer


# ==============================================================================
# STEP 2: Predict WITHOUT ICL (Zero-Shot)
# ==============================================================================

def predict_without_icl(model, tokenizer, question: str, 
                        system_prompt: str = None,
                        max_new_tokens: int = 100,
                        temperature: float = 0.7):
    """
    Make a prediction using the LLM WITHOUT in-context learning examples.
    This is called "zero-shot" prediction.
    
    Args:
        model: The loaded language model
        tokenizer: The corresponding tokenizer
        question: The question/task to predict
        system_prompt: Optional system instruction for the model
        max_new_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (lower = more deterministic)
    
    Returns:
        str: The model's generated response
    
    Example:
        >>> response = predict_without_icl(model, tokenizer, 
        ...     "What is the solubility of aspirin?")
    """
    # Build the prompt using chat template format (for instruction-tuned models)
    messages = []
    
    # Add system message if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add the user question
    messages.append({"role": "user", "content": question})
    
    # Convert messages to the model's expected format
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize the prompt and move to model's device
    device = _get_model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the generated tokens (skip the input prompt)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                 skip_special_tokens=True)
    
    return response


# ==============================================================================
# STEP 3: Predict WITH ICL (Few-Shot)
# ==============================================================================

def predict_with_icl(model, tokenizer, question: str,
                     icl_examples: list,
                     system_prompt: str = None,
                     max_new_tokens: int = 100,
                     temperature: float = 0.7):
    """
    Make a prediction using the LLM WITH in-context learning examples.
    This is called "few-shot" prediction.
    
    The ICL examples teach the model the expected input-output pattern,
    helping it generalize to new inputs.
    
    Args:
        model: The loaded language model
        tokenizer: The corresponding tokenizer
        question: The question/task to predict
        icl_examples: List of example dictionaries with 'input' and 'output' keys
                      Example: [{"input": "Drug: Aspirin", "output": "-3.5"},
                                {"input": "Drug: Caffeine", "output": "-1.2"}]
        system_prompt: Optional system instruction for the model
        max_new_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (lower = more deterministic)
    
    Returns:
        str: The model's generated response
    
    Example:
        >>> examples = [
        ...     {"input": "Drug SMILES: CCO", "output": "-0.77"},
        ...     {"input": "Drug SMILES: CCCO", "output": "-0.50"},
        ... ]
        >>> response = predict_with_icl(model, tokenizer, 
        ...     "Drug SMILES: CCCCO", examples)
    """
    # Build the ICL prompt by combining examples with the question
    icl_prompt_parts = []
    
    # Add each example in a consistent format
    for i, example in enumerate(icl_examples, 1):
        example_text = f"Example {i}:\n{example['input']}\nAnswer: {example['output']}\n"
        icl_prompt_parts.append(example_text)
    
    # Add the current question (without the answer)
    icl_prompt_parts.append(f"Now answer the following:\n{question}\nAnswer:")
    
    # Combine all parts into the user message
    user_content = "\n".join(icl_prompt_parts)
    
    # Build messages
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": user_content})
    
    # Convert to model format
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize the prompt and move to model's device
    device = _get_model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the generated tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                 skip_special_tokens=True)
    
    return response


# ==============================================================================
# STEP 4: Helper function to format ICL examples from data
# ==============================================================================

def create_icl_examples(input_data: list, output_labels: list, 
                        input_template: str = "Input: {}", 
                        num_shots: int = 5):
    """
    Create ICL examples from input data and labels.
    
    Args:
        input_data: List of input values (e.g., drug SMILES strings)
        output_labels: List of corresponding output values (e.g., solubility scores)
        input_template: Template string for formatting inputs
        num_shots: Number of examples to include (k-shot learning)
    
    Returns:
        list: List of formatted ICL example dictionaries
    
    Example:
        >>> smiles = ["CCO", "CCCO", "CCCCO"]
        >>> labels = [-0.77, -0.50, -0.25]
        >>> examples = create_icl_examples(smiles, labels, 
        ...     input_template="Drug SMILES: {}", num_shots=2)
    """
    examples = []
    
    # Limit to available data
    num_examples = min(num_shots, len(input_data), len(output_labels))
    
    for i in range(num_examples):
        examples.append({
            "input": input_template.format(input_data[i]),
            "output": str(output_labels[i])
        })
    
    return examples


# ==============================================================================
# MAIN: Demonstration
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("In-Context Learning (ICL) Tutorial")
    print("=" * 60)
    
    # Note: To run this example, you need:
    # 1. A Hugging Face account with access to the model
    # 2. Set your HF token: huggingface-cli login
    # 3. Sufficient GPU memory (8B model needs ~16GB)
    
    # For demonstration without running, here's how you would use it:
    
    print("\n--- Example Usage (Not Executed) ---\n")
    
    print("1. Load the model:")
    print('   model, tokenizer = load_llm_model("meta-llama/Meta-Llama-3.1-8B-Instruct")')
    
    print("\n2. Zero-shot prediction (without ICL):")
    print('   question = "What is the solubility (log mol/L) of the molecule with SMILES: CCO?"')
    print('   response = predict_without_icl(model, tokenizer, question)')
    
    print("\n3. Few-shot prediction (with ICL):")
    print("   # Prepare ICL examples")
    print('   examples = [')
    print('       {"input": "Drug SMILES: CCO", "output": "-0.77"},')
    print('       {"input": "Drug SMILES: CCCO", "output": "-0.50"},')
    print('       {"input": "Drug SMILES: CC(=O)O", "output": "-1.33"},')
    print('   ]')
    print('   question = "Drug SMILES: CCCCO"')
    print('   response = predict_with_icl(model, tokenizer, question, examples)')
    
    print("\n4. Create examples from data:")
    print('   smiles_list = ["CCO", "CCCO", "CC(=O)O"]')
    print('   labels_list = [-0.77, -0.50, -1.33]')
    print('   examples = create_icl_examples(smiles_list, labels_list, ')
    print('       input_template="Drug SMILES: {}", num_shots=3)')
    
    print("\n" + "=" * 60)
    print("Key Concepts:")
    print("=" * 60)
    print("""
    1. Zero-Shot: Ask the model directly without examples.
       - Relies entirely on the model's pre-trained knowledge.
       - May struggle with specialized or domain-specific tasks.
    
    2. Few-Shot (ICL): Provide examples in the prompt.
       - Examples show the model the expected input-output pattern.
       - More examples usually improve accuracy (up to a point).
       - The model learns the task "in-context" without fine-tuning.
    
    3. Benefits of ICL:
       - No training required (uses frozen model weights).
       - Easy to adapt to new tasks by changing examples.
       - Quick to experiment and iterate.
    
    4. Tips for Better ICL:
       - Choose diverse, representative examples.
       - Use consistent formatting across examples.
       - Order examples from simple to complex or by similarity.
       - Include examples that cover edge cases.
    """)
