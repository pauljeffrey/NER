DRUG_CATEGORY = ['B-GENERIC', 'B-DRUGNAME', 'I-BRAND', 'I-GENERIC', 'B-BRAND', 'I-DRUGNAME']
DRUG_FEATURES = ["drug_name","brand_name", "generic_name", "related_drugs"]

INFORMATION_CATEGORY = [ 'I-DOSE', 'I-SYMPTOMS', 'B-ILL', 'B-BRANDNAME', 
                        'I-BRANDNAME',  'I-ILL_NAME', 'B-ILL_NAME', 'I-DRUGCLASS',
                        'B-GENERICNAME', 'I-CLASS', 'I-SIDE', 'I-GENERICNAME','B-CLASS',
                        'I-ILL', 'B-SYMPTOMS', 'O', 'B-DRUG', 'B-SIDE', 
                        'B-DRUGCLASS']

TAGS_TO_FEATURES = {'B-GENERIC': "generic name", 'B-DRUGNAME': ["related_drugs","drug_name"],
'I-DOSE': ["dose","dosage", "dosages"], 'I-SYMPTOMS':"symptoms", 'B-ILL':"medical condition",  'B-BRANDNAME':"brand_names",  
'I-BRANDNAME': "brand_names", 'I-BRAND': "drug brand", 'I-ILL_NAME':"medical_conditions", 
'B-ILL_NAME':"medical_conditions", 'I-DRUGCLASS' : "drug class", 'B-GENERICNAME': "generic_name", 
'I-GENERIC':"generic name", 'B-BRAND':"drug band", 'I-CLASS':"drug_class",
'I-SIDE':"side_effects", 'I-GENERICNAME': "generic_name", 'I-DRUGNAME': ["related_drugs","drug_name"], 
'B-CLASS':"drug_class", 'I-ILL':"medical condition", 'B-SYMPTOMS':"symptoms", 'O': "others",
'B-DRUG': "drug", 'B-SIDE': "side_effects", 'B-DRUGCLASS': "drug class"}

label_names = ['I-GENERICNAME',
 'B-DRUG',
 'B-DRUGNAME',
 'I-DOSE',
 'B-GENERIC',
 'I-GENERIC',
 'I-DRUGNAME',
 'I-SYMPTOMS',
 'B-DRUGCLASS',
 'B-SIDE',
 'B-BRAND',
 'B-GENERICNAME',
 'I-ILL',
 'B-SYMPTOMS',
 'I-DRUGCLASS',
 'I-ILL_NAME',
 'I-BRAND',
 'B-BRANDNAME',
 'I-CLASS',
 'O',
 'I-BRANDNAME',
 'B-ILL',
 'B-ILL_NAME',
 'I-SIDE',
 'B-CLASS']


MODEL_PATH = ""
PARQUET_PATH = ""


def tokenize_function(examples):
    return tokenizer(examples["tokens"], padding="max_length", truncation=True, is_split_into_words=True)



#Get the values for input_ids, token_type_ids, attention_mask
def tokenize_adjust_labels(all_samples_per_split):
  tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], truncation=True, is_split_into_words=True,)
  #tokenized_samples is not a datasets object so this alone won't work with Trainer API, hence map is used 
  #so the new keys [input_ids, labels (after adjustment)]
  #can be added to the datasets dict for each train test validation split
  total_adjusted_labels = []
  #print(len(tokenized_samples["input_ids"]))
  for k in range(0, len(tokenized_samples["input_ids"])):
    prev_wid = -1
    word_ids_list = tokenized_samples.word_ids(batch_index=k)
    existing_label_ids = all_samples_per_split["ner_tags"][k]
    #print(existing_label_ids)
    i = -1
    adjusted_label_ids = []
   
    for wid in word_ids_list:
      if(wid is None):
        adjusted_label_ids.append(-100)
      elif(wid!=prev_wid):
        i = i + 1
        adjusted_label_ids.append(existing_label_ids[i])
        prev_wid = wid
      else:
        
        adjusted_label_ids.append(existing_label_ids[i])
        
    total_adjusted_labels.append(adjusted_label_ids)
  tokenized_samples["labels"] = total_adjusted_labels
  return tokenized_samples



def join_words_tags(wp):
    """
        Function aggregates words together based on the predicted tags.
    """
    subword =False
    # Create a temporary empty list to store words, predictions, scores.
    temp_word = []
    temp_pred = []
    temp_score = []
    
    # Create lists to store eventual processsed words, tags and corresponding probability score.
    input_words = []
    tags = []
    prob_scores = []

    tag_processed = [False for i in range(len(wp["ner"]))]
    
    # Iterate through each words in the list
    for ind,(word, tag, score) in enumerate(zip(wp["words"],wp["ner"], wp["prob. score"])):
        if not tag_processed[ind] and tag.startswith("B-"):
          
            subword = True                                      
            temp_word.append(word)
            #temp_pred.append(pred)
            temp_score.append(score)
            tag_processed[ind] = True
            
            if ind + 1 >= len(wp["ner"]):
                input_words.append(word)
                tags.append(tag)
                prob_scores.append(score)
                subword = False
                break
            else:
                while subword == True:

                    for i in range(ind+1, len(wp["ner"])):
                        if wp["ner"][i].startswith("I-") and wp['ner'][i][2:] == tag[2:]:
                            temp_word.append(wp["words"][i])
                            temp_score.append(wp["prob. score"][i])
                            tag_processed[i] = True
                        else:
                            input_words.append(" ".join(temp_word))
                            tags.append(tag)
                            prob_scores.append(np.mean(temp_score))
                            temp_word = []
                            temp_pred = []
                            temp_score = []    

                            subword = False
                            break
            
            continue
         
        if not tag_processed[ind] and tag.startswith("I-"):
            
            input_words.append(word)
            tags.append(tag)
            prob_scores.append(score)
            tag_processed[ind] = True
            

        elif not tag_processed[ind] and tag.startswith("O"):
            
            input_words.append(word)
            tags.append(tag)
            prob_scores.append(score)
            tag_processed[ind] = True
            continue
            
        else:
            continue
            

    return {"ner": tags, "words": input_words, 'prob. score': prob_scores}


def aggregate(words,predictions,prob_score,use_max= True):
   
    assert (len(words) == len(predictions))and (len(predictions) == len(prob_score))
    prob_score.pop(0)
    prob_score.pop(-1)
    reversed_prob_score = prob_score[::-1]
 
    words.pop(0)
    words.pop(-1)
    reversed_words= words[::-1]
    
    predictions.pop(0)
    predictions.pop(-1)    
    reversed_predictions = predictions[::-1]
           

    input_words = []
    tags = []
    prob_scores = []

    subword =False
    temp_word = []
    temp_pred = []
    temp_score = []

    for ind ,(word, pred, score) in enumerate(zip(reversed_words, reversed_predictions, reversed_prob_score)):
        if word.startswith("##"):
            subword = True                                      
            temp_word.insert(0, word.replace("##",""))
            temp_pred.insert(0, pred)
            temp_score.insert(0, score)
        else:
            if subword:
                temp_word.insert(0, word)
                temp_pred.insert(0, pred)
                temp_score.insert(0,score)
                if use_max:                    
                    input_words.insert(0,"".join(temp_word))
                    tags.insert(0, temp_pred[np.argmax(temp_score)])
                    prob_scores.insert(0, np.max(temp_score))
                else:
                    input_words.insert(0,"".join(temp_word))
                    if len(temp_pred) <= 2:
                        tags.insert(0, temp_pred[np.argmax(temp_score)])
                        prob_scores.insert(0, np.max(temp_score))
                    else:
                        modals, counts = np.unique(temp_pred, return_counts=True)
                        index = np.argmax(counts)
                        tags.insert(0, modals[index])
                        sum_pred = []
                        for ind, t in enumerate(temp_pred):
                            if t == modals[index]:
                                sum_pred.append(temp_score[ind])
                        prob_scores.insert(0, np.mean(sum_pred))
                    
                temp_word=[]
                temp_pred=[]
                temp_score=[]
                subword = False
            else:
                input_words.insert(0, word)
                tags.insert(0, pred)
                prob_scores.insert(0, score)
                #subword =False
                
    words_prediction  = join_words_tags({"ner": tags, "words": input_words, 'prob. score': prob_scores})

    return words_prediction



def load_model_tokenizer(path, num_labels):
    # Load trained model and tokenizer
    tokenizer =  AutoTokenizer.from_pretrained(path)
    model =  AutoModelForTokenClassification.from_pretrained(path, num_labels=num_labels)
    return model , tokenizer


def get_response(model, tokenizer, query, parquet_path ,focus=None, threshold=0.1, use_max=False):
    """
        Retrieves drug information from parquet database using the model's predicted output.
    """
    # Tokenize query
    tokens = tokenizer(query)      
    # Fetch predictions. Shape == (1, n_subwords, len(label_names))
    predictions = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0),
                    attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
    # Get the probability score for each predictions
    prob_score = list(torch.max(torch.softmax(predictions.logits, axis=-1).squeeze(),
                                axis=1).values.detach().numpy())
    #Get the maximum predicted values from these predictions.
    # Shape = (1, n_subwords, 1)
    predictions = torch.argmax(predictions.logits.squeeze(), axis=1)

    # Get tokens ids
    input_ids = torch.tensor(tokens["input_ids"])
#     pred_for_word = [label_names[i] for i in predictions]
#     each_word = tokenizer.batch_decode(input_ids)
    # Post process the generated outputs and their input_ids using the tokenizer.    
    words_prediction = post_process(tokenizer , input_ids, predictions, prob_score, use_max)
    
    # Retrieve results from the parquet database using predicted entity tags.
    results = retrieve_results(words_prediction,parquet_path, focus, threshold=threshold)
    
    return results #words_prediction #

def post_process(tokenizer, input_ids, pred, prob_score,use_max=False):
    
    """
        This function aggregates the word ids and corresponding tags into full words and then into 
        compound words if necessary based on the generated entity tag.
    """
    # Change all generated label to theirtags (e.g 1  -> "B-DRUG")
    pred_for_word = [label_names[i] for i in pred]
    # Tokenize input ids to words (e.g 1 -> "##xy")
    each_word = tokenizer.batch_decode(input_ids)
    # Aggregate subwords into full words and then into compound words based on tag predictions.
    words_tags = aggregate(each_word, pred_for_word, prob_score, use_max)
        
    return words_tags

def retrieve_results(words_prediction,path, focus=None,threshold=0.7):
    """
        Function retrieves results from the parquet database and provide the results.
    
    """
    #  Create result object
    results = {"results": []}
    # create list to store extracted drug names
    drugs = []    
    # Create list to store extracted attributes
    if not focus:
        focus = []
     
   # Loop through each word and corresponding predicted tags 
    for each_word, pred, score in zip(words_prediction['words'], words_prediction["ner"], words_prediction["prob. score"]):
        # if the model's predicted tag is in the drug_category append to the drug list
        if pred in DRUG_CATEGORY and score >= threshold:
            drugs.append(each_word)   
        # Else append to features / attribute list
        elif pred in INFORMATION_CATEGORY and score >= threshold:
            feat = TAGS_TO_FEATURES[pred]
            
            if type(feat)== list:
                focus.extend(feat)
            else:
                focus.append(feat)   
        # If the model does not predict a drugname or drug feature for a particular word, ignore it.
        else:
            continue
            
    drugs = set(drugs)
    focus = set(focus)
   
    
    # if drug list is empty, return an empty dictionary
    if len(drugs) == 0:
        return {"results": None}
        

    # Open database:
    db = pd.read_parquet(path)

    # Iterate through each found drug.
    for drug in drugs:
        drug = drug.lower()
        drug_info_found = False
        # if the drug is in the index
        if drug in [each.lower() for each in db.index]:
            # Get drug info
            drug_info, drug_info_found = get_values(db, drug, focus) 
            #print(drug_info)
            #results["results"].append(drug_info)    
            
        # if drug is in the "generic_name"
        if drug in db["generic_name"] and (not drug_info_found):
            drug_info, drug_info_found = get_values(db.set_index("generic_name"), drug, focus)
            #results["results"].append(drug_info) 
        # If drug is in the "brand_name"
        if drug in  db["brand_names"] and (not drug_info_found):
            drug_info, drug_info_found = get_values(db.set_index("brand_name"), drug, focus)
            #results["results"].append(drug_info) 
        # If drug is in the "related_drugs" category
        if drug in db["related_drugs"] and (not drug_info_found):
            drug_info, drug_info_found = get_values(db.set_index("related_drugs"), drug, focus)
            #results["results"].append(drug_info) 
        
    
        # If no drug was found at all then return None for all extracted feature attribute.
        if not drug_info_found and len(focus) > 0:
            drug_info = {"drug_name": drug}            
            for feature in set(focus):
                drug_info[feature] = None
#         else:
#             drug_info = {"drug_name": drug}
#             for feature in df.columns:
#                 drug_info[feature] = None
                
        results["results"].append(drug_info)

    return results
        
        
def get_values(df,drug,features):
    """
        This function gets feature values of the queried drug.
    """
    drug_info = {"drug_name": drug}
    #features = set(features)      
    
    if len(features) > 0:
        for feature in features:
            # Check if focus feature is in the database column
            if feature in df.columns:
                if feature == "generic_name":
                    drug_info[feature] = df.loc[drug, feature].split(",")[0]        
                elif feature == 'CSA':
                    drug_info[feature]= df.loc[drug, feature]
                elif feature == 'pregnancy_category':
                    drug_info[feature]= df.loc[drug, feature]
                else:
                    drug_info[feature] = df.loc[drug, feature].split(',')

    # Else just return all the attributes of the index drug
    else:
        for feature in db.columns:
            if feature == "generic_name":
                    drug_info[feature] = df.loc[drug, feature].split(",")[0]        
            elif feature == 'CSA':
                drug_info[feature]= df.loc[drug, feature]
            elif feature == 'pregnancy_category':
                drug_info[feature]= df.loc[drug, feature]
            else:
                drug_info[feature] = df.loc[drug, feature].split(',')

    return drug_info , True

