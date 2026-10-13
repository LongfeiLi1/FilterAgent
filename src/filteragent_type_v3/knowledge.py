"""Audited static projection of the 12-card inventory; no outcome learning."""
from copy import deepcopy
from rc_agent.knowledge import KnowledgeBase

VERSION = "static_12_inventory_typed_projection_v3"
# Retain only topology-independent source facts; exclude every original derived
# SK/MFB formula. K01 is replaced by a registered-graph protocol card.
GENERAL = {"K03", "K04", "K06", "K07", "K10"}
DISABLED = {"K02", "K05", "K08", "K09", "K11", "K12"}


class TypedKnowledge:
    version = VERSION
    def __init__(self):
        self.base = KnowledgeBase()  # verifies all 12 original resource hashes

    def retrieve(self, case, candidate_id, binding, trusted):
        cards = []
        for original in self.base.cards:
            cid = original["id"]
            if cid in GENERAL:
                c = {"id": cid, "title": original["title"],
                     "claims": deepcopy([x for x in original["claims"] if x["kind"] == "source_fact"]),
                     "source_ids": original["source_ids"],
                     "source_inventory_id": cid,
                     "projection": "source facts only; original topology-derived equations excluded"}
                cards.append(c)
        mapping = {
            "id": "K01", "title": "Registered topology, SI coordinates, signed reference",
            "claims": [{"kind": "derived_protocol", "text":
                "Use the exact registered node graph and Case response profile. T2 has an s^2 numerator and finite passband reference; T3 has negative lowpass polarity. Named parameter bounds and source/load conditions are immutable."}],
            "mapping": binding, "source_ids": [],
            "implementation_sources": ["newbench/topologies/registry.py",
                "newbench/metrics/response.py", "orchestration/filteragent_v3/m1_type_v3/model.py"],
            "projection": "explicit new typed protocol mapping, not the original SKLP formula",
        }
        # Same compact retrieval envelope: mandatory mapping/stability plus 3 facts.
        by_id = {c["id"]: c for c in cards}
        selected = [mapping] + [by_id[k] for k in ("K07", "K03", "K04", "K10")]
        return {"knowledge_version": VERSION, "original_inventory_version": self.base.manifest["knowledge_version"],
                "original_resource_hashes": self.base.manifest["resources"], "original_inventory_count": 12,
                "type_id": case["type_id"], "candidate_id": str(candidate_id),
                "cards": selected, "disabled_for_this_projection": sorted(DISABLED),
                "available_general_cards": sorted(GENERAL), "model_trusted": bool(trusted),
                "no_original_topology_derived_formula_injected": True, "experience_ingestion": False,
                "hard_constraints_unchanged": True}

