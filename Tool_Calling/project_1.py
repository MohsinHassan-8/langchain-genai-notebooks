from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import requests
from langchain_core.messages import HumanMessage
import os
from langchain_core.tools import InjectedToolArg
from typing import Annotated

import json

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

@tool
def get_conv_factor(base_curr : str, target_curr : str) -> float:
       '''this function fetches the conversion factor between base currency and target currency'''
       url = f"https://v6.exchangerate-api.com/v6/{os.getenv('EXCHANGE_RATE_API_KEY')}/pair/{base_curr}/{target_curr}"
       response = requests.get(url)
       return response.json()

@tool
def convert(base_curr_value : int, conversion_rate : Annotated[float, InjectedToolArg] ) -> float:
       '''given currency value and conversion rate, find the target currency value'''
       return base_curr_value * conversion_rate

#get_conv_factor.invoke({'base_curr' : 'USD', 'target_curr': 'PKR'})

#convert.invoke({'base_curr_calue' : 10, 'conversion_rate' : 280.05})

model_with_tools = model.bind_tools([get_conv_factor,convert])

messages = []

query = HumanMessage('What is conversion factor between USD and PKR, and based on that convert 10 USD to PKR')
messages.append(query)

ai_msg = model_with_tools.invoke(messages)

for tool_call in ai_msg.tool_calls:
    # Execute the 1st tool and get the value of conversion rate
    if tool_call['name'] == 'get_conv_factor':
        tool_message1 = get_conv_factor.invoke(tool_call)
        # Fetch this conversion rate
        conversion_rate = json.loads(tool_message1.content)['conversion_rate']
        # Append this tool message to messages list
        messages.append(tool_message1)
        
        # Execute the 2nd tool using the conversion rate from tool 1
    if tool_call['name'] == 'convert':
        # Fetch the current arg
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)

ai_msg_1 = model_with_tools.invoke(messages)
messages.append(ai_msg_1)
print(ai_msg_1.content)
