"""Data models for Label Recovery classification."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class ConversationClassification(BaseModel):
    """Pydantic model for structured LLM response parsing for label recovery."""

    industry: str = Field(
        ...,
        description="Industry category from the specific list: manufacturing, energy production, energy management, "
        "energy technology, apparel retail, retail clothing stores, apparel manufacturing, fitness apparel retail, "
        "footwear retail, safety apparel manufacturing, home decor retail, home textiles retail, manufacturing tools, "
        "retail technology solutions, gaming technology services, transportation technology, transportation services, "
        "logistics and transportation, kitchen appliances manufacturing, utility management services, audio equipment "
        "manufacturing, e-commerce grocery retail, gambling and betting, e-commerce retail baby products, furniture retail, "
        "label manufacturing, cutlery manufacturing, bicycle manufacturing, telecommunications retail, pet retail, "
        "financial services, financial software development, gaming, retail, outdoor equipment retail, e-commerce jewelry "
        "manufacturing, retail fashion accessories, automotive parts retail, fintech services, games, e-commerce retail goods, "
        "automotive retail, coatings manufacturing, sporting goods manufacturing, e-commerce, beverage retailing, "
        "computer hardware manufacturing, automotive manufacturing, e-commerce electronics retail",
    )
    problem: str = Field(
        ...,
        description="Problem type from the specific list: create_account, delete_account, edit_account, switch_account, "
        "check_cancellation_fee, delivery_options, complaint, review, check_invoice, get_invoice, newsletter_subscription, "
        "cancel_order, change_order, place_order, check_payment_methods, payment_issue, check_refund_policy, track_refund, "
        "change_shipping_address, set_up_shipping_address",
    )
    channel: str = Field(..., description="Communication channel: 'email' or 'chat'")
    agent_experience: str = Field(..., description="Agent experience level: 'junior' or 'senior'")
    agent_type: str = Field(..., description="Agent type: 'human' or 'bot'")
    explanation: str = Field(..., description="Brief justification for each classification decision")


@dataclass
class ClassificationResult:
    """Container for LLM classification results with metadata."""

    conversation_id: int
    generator_model: str
    industry: str | None = None
    problem: str | None = None
    channel: str | None = None
    agent_experience: str | None = None
    agent_type: str | None = None
    explanation: str | None = None
    error: str | None = None
    processing_time: float | None = None
    source_file: str | None = None

    def is_successful(self) -> bool:
        """Check if all classification fields are populated."""
        return all([
            self.industry,
            self.problem,
            self.channel,
            self.agent_experience,
            self.agent_type,
        ])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "generator_model": self.generator_model,
            "industry": self.industry,
            "problem": self.problem,
            "channel": self.channel,
            "agent_experience": self.agent_experience,
            "agent_type": self.agent_type,
            "explanation": self.explanation,
            "error": self.error,
            "processing_time": self.processing_time,
            "source_file": self.source_file,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClassificationResult":
        """Create from dictionary."""
        return cls(
            conversation_id=data["conversation_id"],
            generator_model=data.get("generator_model", "unknown"),
            industry=data.get("industry"),
            problem=data.get("problem"),
            channel=data.get("channel"),
            agent_experience=data.get("agent_experience"),
            agent_type=data.get("agent_type"),
            explanation=data.get("explanation"),
            error=data.get("error"),
            processing_time=data.get("processing_time"),
            source_file=data.get("source_file"),
        )
