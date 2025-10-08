from typing import Optional
from floggit import flog
from google.genai import types
from .subagents.knowledge_graph_agent.subagents.fetch_knowledge_agent.tools import get_relevant_neighborhood_for_query


@flog
def expand_query(query: str, graph_id: str) -> Optional[types.Part]:
    """Expands a query by fetching related entities from the knowledge graph and appending them to the query."""
    nbhd = get_relevant_neighborhood_for_query(query=query, graph_id=graph_id)

    relevant_entities_str = ""
    for entity in nbhd['entities'].values():
        entity_name = entity['entity_names'][0]
        entity_str = ''
        if entity.get('properties'):
            entity_str += f"{entity_name} has properties: {str(entity['properties'])}. "
        if len(entity['entity_names']) > 1:
            entity_str += f"{entity_name} is also known as: {', '.join(entity['entity_names'][1:])}"

        if entity_str:
            relevant_entities_str += entity_str + "\n"

    relationships_str = ""
    for rel in nbhd['relationships']:
        source_name = nbhd['entities'][rel['source_entity_id']]['entity_names'][0]
        target_name = nbhd['entities'][rel['target_entity_id']]['entity_names'][0]
        relationships_str += f"{source_name} {rel['relationship']} {target_name}\n"

    if relevant_entities_str or relationships_str:
        relevant_subgraph_str = f"(FYI, according to the Knowledge Graph: {relevant_entities_str}\n{relationships_str}.)"
    else:
        relevant_subgraph_str = ''

    expanded_query = f'{relevant_subgraph_str}\ngraph_id={graph_id}'

    return types.Part(text=expanded_query)
