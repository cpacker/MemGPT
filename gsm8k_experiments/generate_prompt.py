import yaml
from typing import Dict, List, Any
import jinja2

class PromptGenerator:
    def __init__(self, yaml_config: Dict[str, Any]):
        self.config = yaml_config
        self.environment = jinja2.Environment()
        
    def render_template(self, template_str: str, context: Dict[str, Any]) -> str:
        """Render a template string with the given context."""
        template = self.environment.from_string(template_str)
        return template.render(**context)
        
    def format_fewshot_example(self, example: Dict[str, str]) -> str:
        """Format a single few-shot example using the doc_to_text and doc_to_target templates."""
        # Generate the question part
        question_text = self.render_template(self.config['doc_to_text'], example)
        
        # Generate the answer part using doc_to_target if specified
        if 'doc_to_target' in self.config:
            target = self.render_template(self.config['doc_to_target'], example)
        else:
            target = example['target']
            
        return f"{question_text} {target}"

    def generate_few_shot_context(self) -> List[str]:
        """Generate the few-shot context from the examples in the config."""
        if 'fewshot_config' not in self.config:
            return ""
            
        examples = self.config['fewshot_config']['samples']
        formatted_examples = [self.format_fewshot_example(example) for example in examples]
        return formatted_examples

    def generate_prompt(self, question: str) -> str:
        """Generate a complete prompt including few-shot examples and the target question."""
        # Generate few-shot context
        few_shot_context = self.generate_few_shot_context()
        
        # Create context for the target question
        question_context = {'question': question}
        
        # Generate the target question using doc_to_text template
        target_question = self.render_template(self.config['doc_to_text'], question_context)
        
        # Combine everything into the final prompt
        if few_shot_context:
            return f"{few_shot_context}\n\n{target_question}"
        return target_question

def load_yaml_config(yaml_string: str) -> Dict[str, Any]:
    """Load YAML configuration from a string."""
    return yaml.safe_load(yaml_string)

# Example usage
def main():
    # Example question to generate a prompt for
    test_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and sells the rest to her neighbors for $2 per egg. How much money does she make per day?"

    # Create the prompt generator
    yaml_config = load_yaml_config(yaml_string)  # yaml_string would be your YAML content
    generator = PromptGenerator(yaml_config)
    
    # Generate the prompt
    prompt = generator.generate_prompt(test_question)
    
    return prompt

# Example test function
def test_prompt_generator():
    # Test YAML content (truncated version)
    test_yaml = '''
    dataset_name: main
    doc_to_text: 'Q: {{question}}

      A:'
    doc_to_target: '{{answer.split("####")[-1].strip() if answer is defined else target}}'
    fewshot_config:
      sampler: first_n
      samples:
      - question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
        target: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
    '''

    with open("gsm8k-cot.yaml", "r") as f:
        test_yaml = f.read()
    
    # Test question
    test_question = "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"
    
    # Generate prompt
    config = load_yaml_config(test_yaml)
    generator = PromptGenerator(config)
    prompt = generator.generate_prompt(test_question)
    
    print("Generated prompt:")
    print(prompt)

if __name__ == "__main__":
    test_prompt_generator()