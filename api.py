import transformers
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

from typing import Optional, Union
from fastapi import FastAPI

from pydantic import BaseModel
from utils import *

app = FastAPI()


class Request(BaseModel):
    """
        Class handles the structure of the user's query.
        query: user's query
        focus: Particular drug attributes to check for
        threshold: float number that determines whether to ignore predictions of model based on probability score
        
    """
    query: str
    focus: Optional[Union[list, str]] = None
    threshold: Optional[float] = 0.7
    use_max: Optional[bool] = False
    

# Load finetuned NER Model
model, tokenizer = load_model_tokenizer(MODEL_PATH, len(label_names))


@app.post("/results")
def fetch_predictions(request_query: Request):
    query = request_query.query
    focus = request_query.focus
    threshold = request_query.threshold
    use_max = request_query.use_max
    
    if threshold > 1:
        threshold = 0.7
        
    results = get_response(model, tokenizer, query, PARQUET_PATH ,focus=focus, threshold=threshold, use_max= use_max)
    
    return results