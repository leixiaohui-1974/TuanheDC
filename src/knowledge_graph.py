"""
TAOS V3.10 Knowledge Graph Module
知识图谱模块

Features:
- Domain knowledge representation
- Entity and relationship modeling
- Reasoning and inference engine
- Query and traversal operations
- Integration with control systems
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from enum import Enum
from datetime import datetime
import logging
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Entity types in the knowledge graph"""
    CHANNEL = "channel"
    GATE = "gate"
    SENSOR = "sensor"
    PUMP = "pump"
    RESERVOIR = "reservoir"
    JUNCTION = "junction"
    CONTROL_ZONE = "control_zone"
    ALARM = "alarm"
    OPERATOR = "operator"
    MAINTENANCE = "maintenance"
    RULE = "rule"
    PROCEDURE = "procedure"


class RelationType(Enum):
    """Relationship types"""
    CONNECTS_TO = "connects_to"
    UPSTREAM_OF = "upstream_of"
    DOWNSTREAM_OF = "downstream_of"
    CONTROLS = "controls"
    MONITORS = "monitors"
    BELONGS_TO = "belongs_to"
    TRIGGERS = "triggers"
    DEPENDS_ON = "depends_on"
    AFFECTS = "affects"
    CAUSED_BY = "caused_by"
    FOLLOWS = "follows"
    PART_OF = "part_of"
    RESPONSIBLE_FOR = "responsible_for"


@dataclass
class Entity:
    """Knowledge graph entity"""
    entity_id: str
    entity_type: EntityType
    name: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type.value,
            'name': self.name,
            'description': self.description,
            'properties': self.properties,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class Relationship:
    """Knowledge graph relationship"""
    relationship_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'relationship_id': self.relationship_id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type.value,
            'properties': self.properties,
            'weight': self.weight,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class InferenceRule:
    """Inference rule for knowledge reasoning"""
    rule_id: str
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    conclusions: List[Dict[str, Any]]
    confidence: float = 1.0
    enabled: bool = True
    priority: int = 0


class KnowledgeGraph:
    """
    Knowledge Graph for domain knowledge representation
    领域知识图谱
    """

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.inference_rules: Dict[str, InferenceRule] = {}
        
        # Indexes for efficient querying
        self._entity_by_type: Dict[EntityType, Set[str]] = defaultdict(set)
        self._outgoing_relations: Dict[str, Set[str]] = defaultdict(set)
        self._incoming_relations: Dict[str, Set[str]] = defaultdict(set)
        self._relations_by_type: Dict[RelationType, Set[str]] = defaultdict(set)
        
        self._lock = threading.RLock()

    def add_entity(self, entity: Entity) -> str:
        """Add entity to knowledge graph"""
        with self._lock:
            self.entities[entity.entity_id] = entity
            self._entity_by_type[entity.entity_type].add(entity.entity_id)
            logger.debug(f"Added entity: {entity.name} ({entity.entity_type.value})")
            return entity.entity_id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)

    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update entity properties"""
        with self._lock:
            if entity_id not in self.entities:
                return False
            entity = self.entities[entity_id]
            for key, value in updates.items():
                if key == 'properties':
                    entity.properties.update(value)
                elif hasattr(entity, key):
                    setattr(entity, key, value)
            entity.updated_at = datetime.now()
            return True

    def remove_entity(self, entity_id: str) -> bool:
        """Remove entity and its relationships"""
        with self._lock:
            if entity_id not in self.entities:
                return False
            entity = self.entities[entity_id]
            
            # Remove from type index
            self._entity_by_type[entity.entity_type].discard(entity_id)
            
            # Remove related relationships
            rels_to_remove = (
                self._outgoing_relations.get(entity_id, set()) |
                self._incoming_relations.get(entity_id, set())
            )
            for rel_id in list(rels_to_remove):
                self.remove_relationship(rel_id)
            
            del self.entities[entity_id]
            return True

    def add_relationship(self, relationship: Relationship) -> str:
        """Add relationship to knowledge graph"""
        with self._lock:
            if relationship.source_id not in self.entities:
                raise ValueError(f"Source entity not found: {relationship.source_id}")
            if relationship.target_id not in self.entities:
                raise ValueError(f"Target entity not found: {relationship.target_id}")
            
            self.relationships[relationship.relationship_id] = relationship
            self._outgoing_relations[relationship.source_id].add(relationship.relationship_id)
            self._incoming_relations[relationship.target_id].add(relationship.relationship_id)
            self._relations_by_type[relationship.relation_type].add(relationship.relationship_id)
            
            return relationship.relationship_id

    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Get relationship by ID"""
        return self.relationships.get(relationship_id)

    def remove_relationship(self, relationship_id: str) -> bool:
        """Remove relationship"""
        with self._lock:
            if relationship_id not in self.relationships:
                return False
            rel = self.relationships[relationship_id]
            
            self._outgoing_relations[rel.source_id].discard(relationship_id)
            self._incoming_relations[rel.target_id].discard(relationship_id)
            self._relations_by_type[rel.relation_type].discard(relationship_id)
            
            del self.relationships[relationship_id]
            return True

    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a type"""
        with self._lock:
            return [
                self.entities[eid]
                for eid in self._entity_by_type.get(entity_type, set())
                if eid in self.entities
            ]

    def get_outgoing_relationships(self, entity_id: str,
                                    relation_type: Optional[RelationType] = None) -> List[Relationship]:
        """Get outgoing relationships from an entity"""
        with self._lock:
            rel_ids = self._outgoing_relations.get(entity_id, set())
            relationships = [self.relationships[rid] for rid in rel_ids if rid in self.relationships]
            
            if relation_type:
                relationships = [r for r in relationships if r.relation_type == relation_type]
            
            return relationships

    def get_incoming_relationships(self, entity_id: str,
                                    relation_type: Optional[RelationType] = None) -> List[Relationship]:
        """Get incoming relationships to an entity"""
        with self._lock:
            rel_ids = self._incoming_relations.get(entity_id, set())
            relationships = [self.relationships[rid] for rid in rel_ids if rid in self.relationships]
            
            if relation_type:
                relationships = [r for r in relationships if r.relation_type == relation_type]
            
            return relationships

    def get_neighbors(self, entity_id: str,
                      relation_type: Optional[RelationType] = None,
                      direction: str = "both") -> List[Entity]:
        """Get neighboring entities"""
        with self._lock:
            neighbor_ids = set()
            
            if direction in ["out", "both"]:
                for rel in self.get_outgoing_relationships(entity_id, relation_type):
                    neighbor_ids.add(rel.target_id)
            
            if direction in ["in", "both"]:
                for rel in self.get_incoming_relationships(entity_id, relation_type):
                    neighbor_ids.add(rel.source_id)
            
            return [self.entities[nid] for nid in neighbor_ids if nid in self.entities]

    def find_path(self, source_id: str, target_id: str,
                  max_depth: int = 10) -> Optional[List[str]]:
        """Find shortest path between two entities using BFS"""
        if source_id not in self.entities or target_id not in self.entities:
            return None
        
        if source_id == target_id:
            return [source_id]
        
        visited = {source_id}
        queue = [(source_id, [source_id])]
        
        while queue and len(visited) < max_depth * 100:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            for neighbor in self.get_neighbors(current, direction="out"):
                if neighbor.entity_id == target_id:
                    return path + [target_id]
                
                if neighbor.entity_id not in visited:
                    visited.add(neighbor.entity_id)
                    queue.append((neighbor.entity_id, path + [neighbor.entity_id]))
        
        return None

    def get_upstream_entities(self, entity_id: str, max_depth: int = 10) -> List[Entity]:
        """Get all upstream entities"""
        upstream = []
        visited = {entity_id}
        queue = [entity_id]
        depth = 0
        
        while queue and depth < max_depth:
            next_queue = []
            for eid in queue:
                for rel in self.get_incoming_relationships(eid, RelationType.UPSTREAM_OF):
                    if rel.source_id not in visited:
                        visited.add(rel.source_id)
                        next_queue.append(rel.source_id)
                        if rel.source_id in self.entities:
                            upstream.append(self.entities[rel.source_id])
            queue = next_queue
            depth += 1
        
        return upstream

    def get_downstream_entities(self, entity_id: str, max_depth: int = 10) -> List[Entity]:
        """Get all downstream entities"""
        downstream = []
        visited = {entity_id}
        queue = [entity_id]
        depth = 0
        
        while queue and depth < max_depth:
            next_queue = []
            for eid in queue:
                for rel in self.get_outgoing_relationships(eid, RelationType.DOWNSTREAM_OF):
                    if rel.target_id not in visited:
                        visited.add(rel.target_id)
                        next_queue.append(rel.target_id)
                        if rel.target_id in self.entities:
                            downstream.append(self.entities[rel.target_id])
            queue = next_queue
            depth += 1
        
        return downstream

    def query(self, entity_type: Optional[EntityType] = None,
              properties: Optional[Dict[str, Any]] = None,
              name_contains: Optional[str] = None) -> List[Entity]:
        """Query entities with filters"""
        with self._lock:
            if entity_type:
                candidates = [
                    self.entities[eid]
                    for eid in self._entity_by_type.get(entity_type, set())
                    if eid in self.entities
                ]
            else:
                candidates = list(self.entities.values())
            
            results = []
            for entity in candidates:
                if name_contains and name_contains.lower() not in entity.name.lower():
                    continue
                
                if properties:
                    match = True
                    for key, value in properties.items():
                        if entity.properties.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue
                
                results.append(entity)
            
            return results

    def add_inference_rule(self, rule: InferenceRule):
        """Add inference rule"""
        self.inference_rules[rule.rule_id] = rule

    def infer(self, entity_id: str) -> List[Dict[str, Any]]:
        """Apply inference rules to derive new knowledge"""
        if entity_id not in self.entities:
            return []
        
        entity = self.entities[entity_id]
        inferences = []
        
        for rule in sorted(self.inference_rules.values(), key=lambda r: -r.priority):
            if not rule.enabled:
                continue
            
            if self._check_conditions(entity, rule.conditions):
                for conclusion in rule.conclusions:
                    inferences.append({
                        'rule_id': rule.rule_id,
                        'rule_name': rule.name,
                        'conclusion': conclusion,
                        'confidence': rule.confidence
                    })
        
        return inferences

    def _check_conditions(self, entity: Entity, conditions: List[Dict]) -> bool:
        """Check if entity satisfies rule conditions"""
        for condition in conditions:
            cond_type = condition.get('type')
            
            if cond_type == 'entity_type':
                if entity.entity_type.value != condition.get('value'):
                    return False
            
            elif cond_type == 'property':
                prop_name = condition.get('property')
                operator = condition.get('operator', '==')
                expected = condition.get('value')
                actual = entity.properties.get(prop_name)
                
                if operator == '==' and actual != expected:
                    return False
                elif operator == '>' and not (actual is not None and actual > expected):
                    return False
                elif operator == '<' and not (actual is not None and actual < expected):
                    return False
                elif operator == 'in' and actual not in expected:
                    return False
            
            elif cond_type == 'has_relationship':
                rel_type = RelationType(condition.get('relation_type'))
                direction = condition.get('direction', 'out')
                
                if direction == 'out':
                    rels = self.get_outgoing_relationships(entity.entity_id, rel_type)
                else:
                    rels = self.get_incoming_relationships(entity.entity_id, rel_type)
                
                if not rels:
                    return False
        
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        with self._lock:
            entity_counts = {
                etype.value: len(eids)
                for etype, eids in self._entity_by_type.items()
            }
            
            relation_counts = {
                rtype.value: len(rids)
                for rtype, rids in self._relations_by_type.items()
            }
            
            return {
                'total_entities': len(self.entities),
                'total_relationships': len(self.relationships),
                'entity_counts': entity_counts,
                'relation_counts': relation_counts,
                'inference_rules': len(self.inference_rules)
            }

    def export_to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary"""
        with self._lock:
            return {
                'entities': [e.to_dict() for e in self.entities.values()],
                'relationships': [r.to_dict() for r in self.relationships.values()],
                'statistics': self.get_statistics()
            }

    def import_from_dict(self, data: Dict[str, Any]):
        """Import graph from dictionary"""
        with self._lock:
            for entity_data in data.get('entities', []):
                entity = Entity(
                    entity_id=entity_data['entity_id'],
                    entity_type=EntityType(entity_data['entity_type']),
                    name=entity_data['name'],
                    description=entity_data.get('description', ''),
                    properties=entity_data.get('properties', {}),
                    metadata=entity_data.get('metadata', {})
                )
                self.add_entity(entity)
            
            for rel_data in data.get('relationships', []):
                relationship = Relationship(
                    relationship_id=rel_data['relationship_id'],
                    source_id=rel_data['source_id'],
                    target_id=rel_data['target_id'],
                    relation_type=RelationType(rel_data['relation_type']),
                    properties=rel_data.get('properties', {}),
                    weight=rel_data.get('weight', 1.0)
                )
                try:
                    self.add_relationship(relationship)
                except ValueError:
                    pass


class AqueductKnowledgeBuilder:
    """Builder for aqueduct system knowledge graph"""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def add_channel(self, channel_id: str, name: str,
                    length: float, width: float, slope: float,
                    **kwargs) -> str:
        """Add channel entity"""
        entity = Entity(
            entity_id=channel_id,
            entity_type=EntityType.CHANNEL,
            name=name,
            properties={
                'length': length,
                'width': width,
                'slope': slope,
                **kwargs
            }
        )
        return self.graph.add_entity(entity)

    def add_gate(self, gate_id: str, name: str,
                 max_opening: float, gate_type: str = "sluice",
                 **kwargs) -> str:
        """Add gate entity"""
        entity = Entity(
            entity_id=gate_id,
            entity_type=EntityType.GATE,
            name=name,
            properties={
                'max_opening': max_opening,
                'gate_type': gate_type,
                'current_position': 0.0,
                **kwargs
            }
        )
        return self.graph.add_entity(entity)

    def add_sensor(self, sensor_id: str, name: str,
                   sensor_type: str, unit: str,
                   **kwargs) -> str:
        """Add sensor entity"""
        entity = Entity(
            entity_id=sensor_id,
            entity_type=EntityType.SENSOR,
            name=name,
            properties={
                'sensor_type': sensor_type,
                'unit': unit,
                'accuracy': kwargs.get('accuracy', 0.01),
                **kwargs
            }
        )
        return self.graph.add_entity(entity)

    def connect_upstream_downstream(self, upstream_id: str, downstream_id: str):
        """Create upstream-downstream relationship"""
        rel1 = Relationship(
            relationship_id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=upstream_id,
            target_id=downstream_id,
            relation_type=RelationType.UPSTREAM_OF
        )
        rel2 = Relationship(
            relationship_id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=downstream_id,
            target_id=upstream_id,
            relation_type=RelationType.DOWNSTREAM_OF
        )
        self.graph.add_relationship(rel1)
        self.graph.add_relationship(rel2)

    def connect_gate_to_channel(self, gate_id: str, channel_id: str):
        """Connect gate to channel it controls"""
        rel = Relationship(
            relationship_id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=gate_id,
            target_id=channel_id,
            relation_type=RelationType.CONTROLS
        )
        self.graph.add_relationship(rel)

    def connect_sensor_to_entity(self, sensor_id: str, entity_id: str):
        """Connect sensor to entity it monitors"""
        rel = Relationship(
            relationship_id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=sensor_id,
            target_id=entity_id,
            relation_type=RelationType.MONITORS
        )
        self.graph.add_relationship(rel)


class KnowledgeGraphManager:
    """Knowledge Graph Manager"""

    def __init__(self):
        self.graphs: Dict[str, KnowledgeGraph] = {}
        self._lock = threading.Lock()

    def create_graph(self, name: str) -> KnowledgeGraph:
        """Create new knowledge graph"""
        with self._lock:
            graph = KnowledgeGraph()
            self.graphs[name] = graph
            return graph

    def get_graph(self, name: str) -> Optional[KnowledgeGraph]:
        """Get knowledge graph by name"""
        return self.graphs.get(name)

    def delete_graph(self, name: str) -> bool:
        """Delete knowledge graph"""
        with self._lock:
            if name in self.graphs:
                del self.graphs[name]
                return True
            return False

    def list_graphs(self) -> List[str]:
        """List all graph names"""
        return list(self.graphs.keys())


_kg_manager: Optional[KnowledgeGraphManager] = None


def get_knowledge_graph_manager() -> KnowledgeGraphManager:
    """Get singleton knowledge graph manager"""
    global _kg_manager
    if _kg_manager is None:
        _kg_manager = KnowledgeGraphManager()
    return _kg_manager


def create_sample_aqueduct_graph() -> KnowledgeGraph:
    """Create sample aqueduct knowledge graph"""
    graph = KnowledgeGraph()
    builder = AqueductKnowledgeBuilder(graph)
    
    # Add channels
    builder.add_channel("ch_main", "Main Channel", length=5000, width=10, slope=0.0001)
    builder.add_channel("ch_branch1", "Branch 1", length=2000, width=5, slope=0.0002)
    builder.add_channel("ch_branch2", "Branch 2", length=1500, width=4, slope=0.00015)
    
    # Add gates
    builder.add_gate("gate_1", "Main Gate 1", max_opening=2.0)
    builder.add_gate("gate_2", "Branch Gate 1", max_opening=1.5)
    builder.add_gate("gate_3", "Branch Gate 2", max_opening=1.0)
    
    # Add sensors
    builder.add_sensor("sensor_wl1", "Water Level Sensor 1", "water_level", "m")
    builder.add_sensor("sensor_wl2", "Water Level Sensor 2", "water_level", "m")
    builder.add_sensor("sensor_flow1", "Flow Sensor 1", "flow_rate", "m3/s")
    
    # Create relationships
    builder.connect_upstream_downstream("ch_main", "ch_branch1")
    builder.connect_upstream_downstream("ch_main", "ch_branch2")
    builder.connect_gate_to_channel("gate_1", "ch_main")
    builder.connect_gate_to_channel("gate_2", "ch_branch1")
    builder.connect_gate_to_channel("gate_3", "ch_branch2")
    builder.connect_sensor_to_entity("sensor_wl1", "ch_main")
    builder.connect_sensor_to_entity("sensor_wl2", "ch_branch1")
    builder.connect_sensor_to_entity("sensor_flow1", "ch_main")
    
    return graph
