import os 
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision
)
import openai 
import requests
import json 
import uuid
import pandas as pd 

def evaluate_answer_gpt4(df):

    url='http://ai-tools.prod.bhasai.samagra.io/llm/openai/answer_evaluator/'
    header = {'Content-Type':'application/json'}

    output_data=[]

    for index,row in df.iterrows():

        data={
            'actual_answer':row['actual_answer'],
            'LLM_answer':row['LLM_answer'],
            'input_lang':'en'
        }


        data_json=json.dumps(data)

        try:
            response=requests.post(url,headers=header,data=data_json)

            if response.status_code==200:
                response_data=response.json()
                output_data.append(response_data)
            else:
                print(f'Error: {response.status_code}-{response.text}')
        
        except Exception as e:
            print(f'Error: {e}')

    if len(output_data)>0:
        output_file=f'output_{uuid.uuid4()}.json'

        with open(output_file,'w') as f:
            json.dump(output_data,f)
    else:
        print("Warning: no response was generated")





if __name__=="__main__":
    data = {
    'actual_answer': ["- For maize, the recommended seed rate is 6-8 kg per acre.", "- The capital of France is Paris."],
    'LLM_answer': ["AI: The recommended seed rate for maize per acre is around 7-8 kg. Thank you.", "AI: The capital of France is Paris."],
    }
    df = pd.DataFrame(data)

    evaluate_answer(df)


