from utils import load_config
from langchain.llms import GooglePalm

# Load config
config = load_config('config.yaml')
model_API_keys = config['model_API_keys']

# Google PaLM Model
api_file = open(model_API_keys['palm'], 'r')
api_key = api_file.readlines()[1] # read second line
api_key = api_key.strip() # remove newline (\n)
LLM = GooglePalm(google_api_key=api_key)