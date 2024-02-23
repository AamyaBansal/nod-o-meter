from flask import Flask, Response, render_template, jsonify, send_file
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import requests
from openai import OpenAI

OPENAI_API_KEY = None #'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'

client = OpenAI(api_key = OPENAI_API_KEY)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    # {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Generate a witty, curious question which can be answered in only yes or no."}
  ]
)

print(completion.choices[0].message)