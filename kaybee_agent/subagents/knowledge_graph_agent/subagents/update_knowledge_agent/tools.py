import json
import os
from dotenv import load_dotenv
from typing import Optional
from floggit import flog

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.cloud import storage

from .utils import generate_random_string

load_dotenv()

def _get_bucket():
    storage_client = storage.Client()
    bucket_name = os.environ.get("KNOWLEDGE_GRAPH_BUCKET")
    if not bucket_name:
        raise ValueError("KNOWLEDGE_GRAPH_BUCKET environment variable not set.")
    return storage_client.get_bucket(bucket_name)


def _fetch_knowledge_graph(graph_id: str) -> dict:
    """Fetches the knowledge graph from the Google Cloud Storage bucket."""
    bucket = _get_bucket()
    blob = bucket.blob(f"{graph_id}.json")
    if not blob.exists():
        return {"entities": {}, "relationships": []}
    else:
        content = blob.download_as_text()
        return json.loads(content)

def _store_knowledge_graph(knowledge_graph: dict, graph_id: str) -> None:
    """Stores the knowledge graph in the Google Cloud Storage bucket."""
    bucket = _get_bucket()
    blob = bucket.blob(f"{graph_id}.json")
    blob.upload_from_string(
        json.dumps(knowledge_graph, indent=2), content_type="application/json"
    )


def _reformat_graph(graph: dict, frozen_entity_ids: set) -> dict:
    '''
    Args:
        graph (dict): A knowledge graph as a dict, with graph['entities'] as a list.

    Returns:
        dict: graph, but with new entity IDs, and with graph['entities'] now as a dict.'''

    id_mapping = {}
    for entity in graph['entities']:
        if entity['entity_id'] in frozen_entity_ids:
            id_mapping[entity['entity_id']] = entity['entity_id']
        else:
            new_id = f"{entity['entity_names'][0][:4].replace(' ','_').lower()}.{generate_random_string(length=4)}"
            id_mapping[entity['entity_id']] = new_id

    graph['entities'] = {
            id_mapping[entity['entity_id']]: entity | {'entity_id': id_mapping[entity['entity_id']]}
            for entity in graph['entities']
    }

    graph['relationships'] = [
            rel | {
                'source_entity_id': id_mapping[rel['source_entity_id']],
                'target_entity_id': id_mapping[rel['target_entity_id']]
            }
            for rel in graph['relationships']
    ]

    return graph


def update_graph(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """
    Stores the provided graph in the knowledge graph store.
    This will overwrite the existing graph.
    """
    if llm_response.partial:
        return

    existing_knowledge_subgraph = callback_context.state['existing_knowledge']
    frozen_entity_ids = {k for k, v in existing_knowledge_subgraph['entities'].items() if v.get('frozen')}
    updated_knowledge_subgraph = json.loads(llm_response.content.parts[-1].text)
    updated_knowledge_subgraph = _reformat_graph(
            graph=updated_knowledge_subgraph, frozen_entity_ids=frozen_entity_ids)

    _update_knowledge_graph(
            graph_id=callback_context._invocation_context.user_id,
            old_subgraph=existing_knowledge_subgraph,
            new_subgraph=updated_knowledge_subgraph)

@flog
def _update_knowledge_graph(
        graph_id: str,
        old_subgraph: dict,
        new_subgraph: dict):

    graph = _fetch_knowledge_graph(graph_id)

    # Excise existing_knowledge_graph
    graph['entities'] = {
            k: v
            for k, v in graph['entities'].items()
            if k not in old_subgraph['entities']
    }
    old_relationships = [
            (rel['source_entity_id'], rel['target_entity_id'])
            for rel in old_subgraph['relationships']
    ]
    graph['relationships'] = [
            rel for rel in graph['relationships']
            if (rel['source_entity_id'], rel['target_entity_id'])
            not in old_relationships
    ]

    # Insert updated_knowledge_graph
    graph['entities'].update(
            new_subgraph['entities'])
    graph['relationships'].extend(
            new_subgraph['relationships'])

    _store_knowledge_graph(knowledge_graph=graph, graph_id=graph_id)
