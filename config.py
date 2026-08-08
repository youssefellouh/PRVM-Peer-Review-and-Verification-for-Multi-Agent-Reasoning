# -*- coding: UTF-8 -*-
"""
Configuration - Load environment variables
"""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()
MODEL_NAME = 'meta-llama/llama-3.1-8b-instruct'
#MODEL_NAME = 'qwen/qwen3.6-35b-a3b'
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')