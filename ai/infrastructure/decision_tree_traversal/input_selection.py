#match the node question

from similarity import SapBERTEmbedder, match_medical_terms
from ..models.system_one_models import master_decision as MD,slave_decision as SD
from similarity import MedCPTRelevanceScorer
from train_system_one import train_laya as TL

def make_decision(document: dict,node_question:str,parameter_ranges: dict)->dict:

    #initialize embedder object

    #embedder=SapBERTEmbedder()


    # dict
    # {
    #     "document_title": str | None,
    #     "source_pdf": str,
    #     "pages": list[dict],
    #     "tables": list[dict],
    #     "image_directory": str,
    #     "json_path": str
    # }

    #__________________________________________________________________

    #ask laya to decide which table is most relevant based on its title

    table_titles=[]

    for table in document["tables"]:
        title=table["title"]
        table_titles.append(title)

    laya_question="Select the table(s) that are most likely to contain the medical data relevant to the data requirements for answering the provided question"

    laya_decision=SD(question=laya_question,input_text=node_question,options=table_titles)

    selected_option=laya_decision["selected_option"]
    prediction_set=laya_decision["prediction_set"]

    #__________________________________________________________________

    #extract useful data from these tables

    scorer = MedCPTRelevanceScorer()
    required_data={}

    for table in prediction_set:
        column_names=table["columns"]
        row_names=table["rows"]
        data=table["data"]

        #use lexical similarity to select distinct entries in the rows and columns

        #required_columns=match_medical_terms(words=column_names,master_word=node_question, embedder=embedder)
        #required_rows=match_medical_terms(words=row_names,master_word=node_question, embedder=embedder)

        #use cross encoder

        #standardization of names improves its accuracy and makes it easier to fine tune cross encoder in future

        required_columns=scorer.rank(master_query=node_question, candidates=column_names)
        required_rows=scorer.rank(master_query=node_question, candidates=row_names)

        for row in required_rows:
            for column in required_columns:
                required_data[f"{row}:{column}"]=data[row][column]

    #__________________________________________________________________

    #ask med-gemma to make final decision
    SD(question=laya_question,input_text=node_question,options=table_titles)

    #MODELS NEED TO KNOW THE MEDICAL CONTEXT AND NORMAL RANGE VALUES

    context=f'''
    
    Permissible range of parameters: {parameter_ranges}
    
    Provided Medical Data of the Patient:
    
    {required_data}
    
    '''

    route_options=["yes","no"]

    laya_decision=SD(question=node_question,input_text=context,options=route_options)
    medgemma_decision=MD(question=node_question,input_text=context,options=route_options)

    #initiate training of the models
    TL(laya_decision,medgemma_decision)

    return medgemma_decision





#What OCR returns

'''
Returns
-------
dict
{
    "document_title": str | None,
    "source_pdf": str,
    "pages": list[dict],
    "tables": list[dict],
    "image_directory": str,
    "json_path": str
}
'''

#What we do:
#IMAGE PROCESSING extracts the necessary content for each image

"""

   Processes extracted medical images using MedGemma batch inference.

   Returns
   -------
   dict
   {
       "document_title": str | None,
       "source_pdf": str,
       "total_images": int,
       "processed_images": int,
       "findings": list[dict],
       "errors": list[dict]
   }

"""



#pass these to the laya model and ask it to find out which entities are needed

#database query pulls out the required values

#have an XGBoost model decide which model, laya or medgemma should handle the situation

#laya->only laya. Medgemma->have laya learn alongside MedGemma

