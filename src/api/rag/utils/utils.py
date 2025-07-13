import yaml
from jinja2 import Template
from langsmith import Client


# Read the yaml file and return the template
def prompt_template_config(yaml_file, prompt_key):
    
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)    
    template_content = config["prompts"][prompt_key] # a string
    template = Template(template_content) # convert the string to jinja template
    
    return template


# Load from registry
ls_client = Client()
def prompt_template_regstry(prompt_name):
    # get_promt return metadata, pull - template; 0- system, 1 user
    # returns a string
    template_content = ls_client.pull_prompt(prompt_name).messages[1].prompt.template 
    template = Template(template_content) # convert the string to jinja template
    return template





