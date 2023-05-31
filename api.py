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
    
    results = get_response(model, tokenizer, query, PARQUET_PATH ,focus=focus, threshold=threshold, use_max= use_max)
    
    return results