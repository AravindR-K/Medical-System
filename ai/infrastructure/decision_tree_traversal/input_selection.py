#match the node question

from similarity import SapBERTEmbedder, match_medical_terms
from ..models.system_one_models import master_decision,slave_decision


def input_zone(document: dict,node_question:str)->dict:

    #initialize embedder object

    embedder=SapBERTEmbedder()

    for table in document["tables"]:
        column_names=table["columns"]
        row_names=table["rows"]
        data=table["data"]

        #use lexical similarity to select distinct entries in the rows and columns

        required_columns=match_medical_terms(words=column_names,master_word=node_question, embedder=embedder)
        required_rows=match_medical_terms(words=row_names,master_word=node_question, embedder=embedder)

        database_entries=[(x,y) for x in row_names for y in column_names]

        #ask laya and med-gemma as to which of this data is useful

        slave_decision






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

