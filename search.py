import ijson
import random


def search_edges(query_subj, query_pred, query_obj):

    results = []
    with open("./data/relationships.json", "rb") as f:
        for record in ijson.items(f, "item"):
            relationships = record['relationships']
            image_id = record['image_id']

            for relation in relationships:
                obj = relation['object']
                sub = relation['subject']
                
                if 'name' in obj and 'name' in sub:
                    if relation['predicate'] == query_pred and relation['object']['name'] == query_obj and relation['subject']['name'] == query_subj:
                        results.append(image_id)
                        break

            if len(results) > 10:
                break

    return results

def search_nodes(query_node):

    results = []
    with open("./data/relationships.json", "rb") as f:
        for record in ijson.items(f, "item"):

            relationships = record['relationships']
            image_id = record['image_id']

            for relation in relationships:
                obj = relation['object']
                sub = relation['subject']
                
                if 'name' in obj and 'name' in sub:
                    if relation['object']['name'] == query_node or relation['subject']['name'] == query_node:
                        results.append(image_id)
                        break

            if len(results) > 100:
                break
    
    final_results = random.choices(results, k=10)
    return final_results
