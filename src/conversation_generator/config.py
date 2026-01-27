"""Configuration presets and random generation for conversation generator."""

import random
from typing import Literal, TypedDict

# Finno-Ugric languages + English baseline
LANGUAGES: list[tuple[str, str]] = [
    ("et", "Estonian"),
    ("fi", "Finnish"),
    ("hu", "Hungarian"),
    ("en-us", "English"),
]

INDUSTRIES: list[str] = [
    "Retail",
    "E-commerce",
    "Banking",
    "Insurance",
    "Healthcare",
    "Telecommunications",
    "Travel & Hospitality",
    "Airlines",
    "Real Estate",
    "Automotive",
    "Education",
    "Government",
    "Non-profit",
    "Media & Entertainment",
    "Food & Beverage",
    "Manufacturing",
    "Logistics",
    "Energy & Utilities",
    "Fashion",
    "Sports & Fitness",
    "Beauty & Cosmetics",
    "Home & Garden",
    "Electronics",
    "Software & SaaS",
    "Gaming",
    "Music & Audio",
    "Photography",
    "Legal Services",
    "Accounting",
    "HR & Recruitment",
    "Marketing & Advertising",
    "Event Management",
    "Wedding Services",
    "Pet Services",
    "Childcare",
    "Senior Care",
    "Cleaning Services",
    "Security Services",
    "Printing & Publishing",
    "Architecture",
    "Interior Design",
    "Construction",
    "Agriculture",
    "Mining",
    "Pharmaceuticals",
    "Biotechnology",
]

PROBLEMS: list[str] = [
    "Order tracking",
    "Refund request",
    "Product defect",
    "Shipping delay",
    "Wrong item received",
    "Account access issues",
    "Password reset",
    "Billing discrepancy",
    "Subscription cancellation",
    "Technical support",
    "Installation help",
    "Feature request",
    "Complaint",
    "General inquiry",
    "Feedback",
    "Return request",
    "Exchange request",
    "Warranty claim",
    "Appointment scheduling",
    "Service cancellation",
]

LanguageCode = Literal["et", "fi", "hu", "en-us"]
Channel = Literal["email", "chat"]
AgentExperience = Literal["junior", "senior"]
AgentType = Literal["human", "bot"]


class ConversationConfig(TypedDict, total=False):
    """Configuration for generating a conversation."""

    language: LanguageCode
    industry: str
    problem: str
    n_messages: int
    n_agents: int
    channel: Channel
    agent_experience: AgentExperience
    agent_type: AgentType


def random_config() -> ConversationConfig:
    """Generate a random configuration for conversation generation."""
    language_code, _ = random.choice(LANGUAGES)
    return ConversationConfig(
        language=language_code,  # type: ignore
        industry=random.choice(INDUSTRIES),
        problem=random.choice(PROBLEMS),
        n_messages=random.randint(4, 16),
        n_agents=random.choice([1, 2]),
        channel=random.choice(["email", "chat"]),
        agent_experience=random.choice(["junior", "senior"]),
        agent_type=random.choice(["human", "bot"]),
    )


def get_language_name(code: str) -> str:
    """Get the full language name from a language code."""
    for lang_code, lang_name in LANGUAGES:
        if lang_code == code:
            return lang_name
    return code
