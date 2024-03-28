import sys 
import pandas as pd 
import random
from tqdm import tqdm 
from openai import OpenAI



def add_chunks_to_dataset(chunks,chunk,question,answer,num_distract,p=1.0):

  i=chunks.index(chunk)
  docs=[chunk]
  indices=list(range(0,len(chunks)))
  indices.remove(i)
  for j in random.sample(indices,num_distract):
    docs.append(chunks[j])
  oracle=random.uniform(0,1)<p
  if not oracle:
    docs[0]=chunks[random.sample(indices,1)[0]]
  random.shuffle(docs)
  d={
      "title":[],
      "sentences":[],
  }
  d["title"].append(["placeholder_title"]*(num_distract+1))
  d["sentences"].append(docs)

  distractor=d

  #instructions
  context=""
  for doc in docs:
    context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
  context += question

  instruction=context

  return distractor,instruction

def generate_dataset(df:pd.DataFrame,save_path:str,num_distract=3,p=1.0):
  
  chunks=list(df["oracle_context"])
  instructions=[]
  distractors=[]
  
  for i in tqdm(range(0,len(df))):
        distractor,instruction=add_chunks_to_dataset(chunks,df["oracle_context"][i],df["question"][i],df["answer"][i],num_distract=num_distract,p=p)
        instructions.append(instruction)
        distractors.append(distractor)
  
  df["instructions"]=instructions
  df["distractors"]=distractors 

  df.to_csv(save_path)

  print(f"Dataset generated and saved at {save_path}")

  return df 

#prompt for chain of thought 
def cot_prompt(question, chunk, answer) -> list[str]:
    """
    Encode multiple prompt instructions into a single string for genrating
    """

    prompts = []

    prompt = """
        Question: {question}\nContext: {context}\nAnswer: {answer}\n
        Given question, context and answer above provide step-by-step reasoning how the answer is generated using context above. Here is things to pay attention to:
        - First provide step-by-step reasoning on how to answer the question.
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context.
        - End your response with final answer in the form <ANSWER>: {answer}\n
    """.format(question=question, context=str(chunk),answer=answer)
    prompts.append({"role": "system", "content": "You are a helpful question answerer who can provide an answer given a question and relevant context."})
    prompts.append({"role": "user", "content": prompt})
    return prompts

#chain of thought response 
def generate_cot(question, context, answer) -> str:
    """
    Generates the label / answer to `question` using `context` and GPT-4.
    """
    question = cot_prompt(question, context, answer)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=question,
        n=1,
        temperature=0
    )
    response = response.choices[0].message.content
    return response


def add_cot(df,open_ai_key):
   
   global client 
   client= OpenAI(
        api_key=open_ai_key,
    )
   
   cot_answers=[]

   for i in tqdm(range(0,len(df))):
      response=generate_cot(df["questions"][i],df["oracle_context"][i],df["answer"][i])
      cot_answers.append(response)

   df["cot_answers"]=cot_answers

   return df 


   
   
   


