from dataclasses import dataclass, asdict
from typing import Any, Dict

@dataclass
class DocumentStore:
    metaData: Dict[str, Any]
    retrieverEngine: Any
    
@dataclass
class ResponseCandidate:
    index: int
    distance: float
    core_question: str
    core_answer: str
    main_node: str
    responsible_team: str
    question_tag: str
    question_type: str

    def to_dict(self):
        """Export the dataclass to a dictionary."""
        return asdict(self)

    def to_list(self):
        """Export the dataclass to a list of values."""
        return list(asdict(self).values())
    
    def to_qa(self):
        return {
            "Question": self.core_question,
            "Answer": self.core_answer
        }