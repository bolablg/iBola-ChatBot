"""
Knowledge Graph Integration for Enhanced Reasoning Capabilities.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import json
import re
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import GEMINI_API_KEY

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_core.documents import Document
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KNOWLEDGE_GRAPH_AVAILABLE = False
    print("Knowledge graph service requires langchain-google-genai")

class KnowledgeGraph:
    """
    Dynamic knowledge graph for enhanced reasoning and relationships.
    """

    def __init__(self):
        self.nodes = {}  # {node_id: node_data}
        self.edges = defaultdict(list)  # {source_id: [(target_id, relationship, weight)]}
        self.node_types = {}  # {node_id: node_type}
        self.reverse_edges = defaultdict(list)  # {target_id: [(source_id, relationship, weight)]}

        if KNOWLEDGE_GRAPH_AVAILABLE:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                google_api_key=GEMINI_API_KEY
            )

            # Initialize extraction chain
            self.extraction_prompt = PromptTemplate.from_template("""
            Extract key concepts, entities, and relationships from the following text.
            Focus on technical skills, projects, experiences, and educational background.

            Text: {text}

            Extract in this format:
            ENTITIES: [list of key entities/concepts]
            RELATIONSHIPS: [entity1]->[relationship]->[entity2], [entity3]->[relationship]->[entity4]
            CATEGORIES: [entity1:category1, entity2:category2]

            Be specific and technical:
            """)

            self.extraction_chain = LLMChain(
                llm=self.llm,
                prompt=self.extraction_prompt,
                verbose=False
            )

    def add_document(self, doc: Document, doc_id: str = None):
        """Add a document to the knowledge graph."""
        if not KNOWLEDGE_GRAPH_AVAILABLE:
            return

        if not doc_id:
            doc_id = f"doc_{hash(doc.page_content)}"

        try:
            # Extract entities and relationships
            extraction_result = self.extraction_chain.run(text=doc.page_content)

            # Parse the extraction result
            entities, relationships, categories = self._parse_extraction_result(extraction_result)

            # Add entities as nodes
            for entity in entities:
                node_id = self._get_or_create_node(entity, categories.get(entity, "concept"))
                self.nodes[node_id]["documents"].add(doc_id)

            # Add relationships as edges
            for rel in relationships:
                if "->" in rel:
                    parts = rel.split("->")
                    if len(parts) == 3:
                        source, relationship, target = [p.strip() for p in parts]
                        self._add_relationship(source, target, relationship)

            print(f"🕸️  Added {len(entities)} entities and {len(relationships)} relationships to knowledge graph")

        except Exception as e:
            print(f"❌ Error adding document to knowledge graph: {e}")

    def _parse_extraction_result(self, result: str) -> Tuple[List[str], List[str], Dict[str, str]]:
        """Parse the LLM extraction result."""
        entities = []
        relationships = []
        categories = {}

        lines = result.strip().split('\n')

        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('ENTITIES:'):
                current_section = 'entities'
                content = line.replace('ENTITIES:', '').strip()
                if content:
                    entities.extend([e.strip() for e in content.split(',') if e.strip()])
            elif line.startswith('RELATIONSHIPS:'):
                current_section = 'relationships'
                content = line.replace('RELATIONSHIPS:', '').strip()
                if content:
                    relationships.extend([r.strip() for r in content.split(',') if r.strip()])
            elif line.startswith('CATEGORIES:'):
                current_section = 'categories'
                content = line.replace('CATEGORIES:', '').strip()
                if content:
                    for cat_pair in content.split(','):
                        if ':' in cat_pair:
                            entity, category = cat_pair.split(':', 1)
                            categories[entity.strip()] = category.strip()
            elif current_section and line:
                # Continue parsing multi-line sections
                if current_section == 'entities' and ',' in line:
                    entities.extend([e.strip() for e in line.split(',') if e.strip()])
                elif current_section == 'relationships' and ',' in line:
                    relationships.extend([r.strip() for r in line.split(',') if r.strip()])
                elif current_section == 'categories' and ',' in line:
                    for cat_pair in line.split(','):
                        if ':' in cat_pair:
                            entity, category = cat_pair.split(':', 1)
                            categories[entity.strip()] = category.strip()

        return entities, relationships, categories

    def _get_or_create_node(self, entity: str, node_type: str = "concept") -> str:
        """Get existing node or create new one."""
        # Create a normalized node ID
        node_id = re.sub(r'[^\w\s]', '', entity.lower()).replace(' ', '_')

        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "label": entity,
                "type": node_type,
                "created_at": datetime.now().isoformat(),
                "documents": set(),
                "metadata": {}
            }
            self.node_types[node_id] = node_type

        return node_id

    def _add_relationship(self, source_entity: str, target_entity: str, relationship: str):
        """Add a relationship between two entities."""
        source_id = self._get_or_create_node(source_entity)
        target_id = self._get_or_create_node(target_entity)

        # Add forward edge
        self.edges[source_id].append((target_id, relationship, 1.0))

        # Add reverse edge
        self.reverse_edges[target_id].append((source_id, relationship, 1.0))

    def query_related_concepts(self, concept: str, max_depth: int = 2) -> Dict[str, Any]:
        """Query related concepts in the knowledge graph."""
        concept_id = self._normalize_concept(concept)

        if concept_id not in self.nodes:
            return {"error": f"Concept '{concept}' not found in knowledge graph"}

        visited = set()
        related = defaultdict(list)

        def traverse(node_id: str, depth: int, path: List[str]):
            if depth > max_depth or node_id in visited:
                return

            visited.add(node_id)
            path.append(self.nodes[node_id]["label"])

            # Get outgoing relationships
            for target_id, relationship, weight in self.edges[node_id]:
                if target_id not in visited:
                    related[relationship].append({
                        "concept": self.nodes[target_id]["label"],
                        "type": self.nodes[target_id]["type"],
                        "path": path.copy()
                    })

            # Continue traversal
            for target_id, _, _ in self.edges[node_id]:
                if target_id not in visited:
                    traverse(target_id, depth + 1, path)

            path.pop()
            visited.remove(node_id)

        traverse(concept_id, 0, [])
        return dict(related)

    def find_paths(self, start_concept: str, end_concept: str, max_depth: int = 3) -> List[List[str]]:
        """Find paths between two concepts."""
        start_id = self._normalize_concept(start_concept)
        end_id = self._normalize_concept(end_concept)

        if start_id not in self.nodes or end_id not in self.nodes:
            return []

        paths = []

        def dfs(current_id: str, target_id: str, path: List[str], depth: int):
            if depth > max_depth:
                return

            path.append(self.nodes[current_id]["label"])

            if current_id == target_id:
                paths.append(path.copy())
                path.pop()
                return

            for neighbor_id, relationship, _ in self.edges[current_id]:
                if neighbor_id not in path:  # Avoid cycles
                    dfs(neighbor_id, target_id, path, depth + 1)

            path.pop()

        dfs(start_id, end_id, [], 0)
        return paths

    def get_concept_recommendations(self, user_query: str) -> List[str]:
        """Get concept recommendations based on user query."""
        query_terms = set(re.findall(r'\b\w+\b', user_query.lower()))

        recommendations = []
        for node_id, node_data in self.nodes.items():
            node_terms = set(re.findall(r'\b\w+\b', node_data["label"].lower()))
            overlap = len(query_terms & node_terms)

            if overlap > 0:
                recommendations.append({
                    "concept": node_data["label"],
                    "type": node_data["type"],
                    "relevance_score": overlap / len(query_terms)
                })

        # Sort by relevance score
        recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
        return [rec["concept"] for rec in recommendations[:5]]

    def _normalize_concept(self, concept: str) -> str:
        """Normalize concept to match node ID format."""
        return re.sub(r'[^\w\s]', '', concept.lower()).replace(' ', '_')

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        node_type_counts = Counter(self.node_types.values())
        edge_counts = sum(len(edges) for edges in self.edges.values())

        return {
            "total_nodes": len(self.nodes),
            "total_edges": edge_counts,
            "node_types": dict(node_type_counts),
            "most_connected_nodes": self._get_most_connected_nodes(),
            "graph_density": edge_counts / (len(self.nodes) * (len(self.nodes) - 1)) if len(self.nodes) > 1 else 0
        }

    def _get_most_connected_nodes(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get the most connected nodes in the graph."""
        node_degrees = {}

        for node_id in self.nodes:
            outgoing = len(self.edges[node_id])
            incoming = len(self.reverse_edges[node_id])
            node_degrees[node_id] = outgoing + incoming

        sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "node": self.nodes[node_id]["label"],
                "type": self.nodes[node_id]["type"],
                "connections": degree
            }
            for node_id, degree in sorted_nodes[:top_k]
        ]

    def export_graph(self, filepath: str):
        """Export the knowledge graph to a JSON file."""
        export_data = {
            "nodes": self.nodes,
            "edges": dict(self.edges),
            "node_types": self.node_types,
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"📊 Exported knowledge graph to {filepath}")

    def import_graph(self, filepath: str):
        """Import knowledge graph from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.nodes = data.get("nodes", {})
            self.edges = defaultdict(list, data.get("edges", {}))
            self.node_types = data.get("node_types", {})

            # Rebuild reverse edges
            self.reverse_edges = defaultdict(list)
            for source_id, edge_list in self.edges.items():
                for target_id, relationship, weight in edge_list:
                    self.reverse_edges[target_id].append((source_id, relationship, weight))

            print(f"📥 Imported knowledge graph from {filepath}")

        except Exception as e:
            print(f"❌ Error importing knowledge graph: {e}")


class KnowledgeGraphReasoner:
    """
    Enhanced reasoning capabilities using the knowledge graph.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

        if KNOWLEDGE_GRAPH_AVAILABLE:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.2,
                google_api_key=GEMINI_API_KEY
            )

            # Reasoning chain
            self.reasoning_prompt = PromptTemplate.from_template("""
            Use the following context from the knowledge graph to answer the question.
            Consider relationships, concepts, and connections between entities.

            Question: {question}
            Related Concepts: {concepts}
            Knowledge Context: {context}

            Provide a comprehensive answer based on the available knowledge:
            """)

            self.reasoning_chain = LLMChain(
                llm=self.llm,
                prompt=self.reasoning_prompt,
                verbose=False
            )

    def reason_about_query(self, query: str) -> Dict[str, Any]:
        """Use knowledge graph for enhanced reasoning about a query."""
        if not KNOWLEDGE_GRAPH_AVAILABLE:
            return {"answer": "Knowledge graph reasoning not available"}

        try:
            # Get related concepts
            related_concepts = self.kg.get_concept_recommendations(query)

            # Get context from knowledge graph
            context_parts = []
            for concept in related_concepts[:3]:
                related = self.kg.query_related_concepts(concept, max_depth=1)
                if related:
                    context_parts.append(f"{concept}: {related}")

            context = "; ".join(context_parts)

            # Use LLM for reasoning
            result = self.reasoning_chain.run(
                question=query,
                concepts=", ".join(related_concepts),
                context=context
            )

            return {
                "answer": result.strip(),
                "related_concepts": related_concepts,
                "knowledge_context": context,
                "reasoning_method": "knowledge_graph_enhanced"
            }

        except Exception as e:
            print(f"❌ Error in knowledge graph reasoning: {e}")
            return {"answer": "Error in reasoning process", "error": str(e)}

# Global knowledge graph instance
knowledge_graph = KnowledgeGraph() if KNOWLEDGE_GRAPH_AVAILABLE else None
reasoner = KnowledgeGraphReasoner(knowledge_graph) if knowledge_graph else None
