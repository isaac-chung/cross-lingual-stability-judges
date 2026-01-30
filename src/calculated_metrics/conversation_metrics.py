"""
Conversation Metrics Base Module

This module provides base classes and data structures for conversation analysis.
Contains language-agnostic conversation metrics and data containers.

Classes:
    - ConversationTurn: Represents a single conversation turn
    - ConversationData: Container for conversation data with structured turns
    - ConversationMetrics: Parent class for language-agnostic conversation metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Set
import re
import logging
from dataclasses import dataclass
from rich import print as rich_print

# Configure logging
logger = logging.getLogger(__name__)

__version__ = "1.0.0"

##---------------------------------------------------- HELPER FUNCTIONS ----------------------------------------------------##
def save_model_results_to_csv(model_results: dict, language_code: str) -> None:
    """
    Save model analysis results to CSV file.
    
    Args:
        model_results: Dictionary mapping model names to their result DataFrames
        language_code: Language code ('et', 'hu', 'fi') for filename generation
    """
    if not model_results:
        return
    
    # Map language codes to full names
    language_names = {
        'et': 'estonian',
        'hu': 'hungarian',
        'fi': 'finnish'
    }
    
    language_name = language_names.get(language_code, language_code)
    
    # Combine all results for saving
    all_results = []
    for model_name, df in model_results.items():
        df_copy = df.copy()
        df_copy['model_name'] = model_name
        all_results.append(df_copy)

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        output_file = f'{language_name}_conversation_metrics.csv'
        combined_df.to_csv(output_file, index=False)
        rich_print(f"[green]💾 Results saved to: {output_file}[/green]")
##---------------------------------------------------- ConversationTurn ----------------------------------------------------##
@dataclass
class ConversationTurn:
    """Represents a single conversation turn."""
    speaker: str
    text: str
    is_agent: Optional[bool] = None  # True if agent, False if client, None if unknown
##---------------------------------------------------- ConversationData ----------------------------------------------------##
@dataclass
class ConversationData:
    """
    Container for conversation data that supports both structured turns and formatted strings.
    Avoids inefficient join-then-split pattern by maintaining both representations.
    """
    turns: List[ConversationTurn]

    @property
    def formatted_string(self) -> str:
        """Get conversation as formatted string (lazy evaluation)."""
        if not hasattr(self, '_formatted_string'):
            lines = [f"{turn.speaker}: {turn.text}" for turn in self.turns]
            self._formatted_string = "\n" + "\n".join(lines) + "\n        "
        return self._formatted_string

    @property
    def turns_as_dicts(self) -> List[Dict[str, str]]:
        """Get turns as list of dictionaries for compatibility."""
        return [{"speaker": turn.speaker, "text": turn.text} for turn in self.turns]

    @classmethod
    def from_string(cls, conversation: str, speaker_pattern: str = r'(Agent|Klient|Kasutaja|Abistaja|Klienditeenindaja):'): #Should be multiling: Klient|Kasutaja|Abistaja|Klienditeenindaja
        """Create ConversationData from formatted string (fallback for existing data)."""
        turns = []
        parts = re.split(f'({speaker_pattern})', conversation)

        current_speaker = None
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if re.match(speaker_pattern, part):
                current_speaker = part.replace(':', '')
            elif current_speaker:
                turns.append(ConversationTurn(speaker=current_speaker, text=part))

        return cls(turns=turns)

    @classmethod
    def from_comments(cls, comments: List[Dict], clean_text_func=None):
        """Create ConversationData from structured comments data."""
        turns = []
        for comment in comments:
            author_name = comment.get('authorName', 'Unknown')
            comment_text = comment.get('comment', '')
            is_agent = comment.get('is_agent', None)  # Get agent status if available

            if comment_text.strip():
                # Apply text cleaning if function provided
                if clean_text_func:
                    comment_text = clean_text_func(comment_text)
                turns.append(ConversationTurn(speaker=author_name, text=comment_text, is_agent=is_agent))

        return cls(turns=turns)
##---------------------------------------------------- ConversationMetrics ----------------------------------------------------##
class ConversationMetrics:
    """Parent class for language-agnostic conversation metrics."""

    def __init__(self):
        """Initialize base conversation metrics."""
        self.semantic_models = {}
        self.speaker_pattern = r'(Agent|Klient|Kasutaja|Abistaja|Klienditeenindaja):' #Default is Estonian
        logger.info("🌐 Initializing Conversation Metrics Base")

    def extract_turns(self, conversation_input: Union[str, ConversationData], speaker_pattern: str = None) -> List[Dict[str, str]]:
        """
        Extract conversation turns with speaker labels.
        Supports both legacy string input and new ConversationData objects.
        
        Parameters:
        - conversation_input: String or ConversationData object
        - speaker_pattern: Optional speaker pattern override (uses self.speaker_pattern if None)
        """
        # Use instance speaker_pattern if not provided
        if speaker_pattern is None:
            speaker_pattern = self.speaker_pattern
            
        # Handle ConversationData objects directly (avoid join-split pattern)
        # Use hasattr for duck typing to avoid __main__ vs module import issues
        if hasattr(conversation_input, 'turns') and hasattr(conversation_input, 'formatted_string'):
            return [
                {
                    'speaker': turn.speaker,
                    'text': self.clean_text(turn.text),
                    'is_agent': turn.is_agent  # Include is_agent field for Self-BLEU categorization
                }
                for turn in conversation_input.turns
            ]

        # Legacy string processing for backward compatibility
        if not isinstance(conversation_input, str):
            raise TypeError(f"conversation_input must be str or ConversationData, got {type(conversation_input)}")
        
        conversation = conversation_input
        turns = []

        # Split by speaker patterns
        parts = re.split(f'({speaker_pattern})', conversation)

        current_speaker = None
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if re.match(speaker_pattern, part):
                current_speaker = part.replace(':', '')
            elif current_speaker:
                turns.append({
                    'speaker': current_speaker,
                    'text': self.clean_text(part)
                })

        return turns

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize conversation text.
        Child classes should override for language-specific cleaning.
        """
        if not isinstance(text, str):
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove common conversation artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed annotations
        text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical notes

        return text

    def tokenize_with_stopwords_and_regex(self, text: str, stopwords: set = None, min_token_length: int = 1) -> List[str]:
        """
        Tokenize text using regex and filter by stopwords.
        
        This is a general-purpose tokenization method that:
        1. Extracts word tokens using regex pattern \b\w+\b
        2. Converts to lowercase
        3. Filters out stopwords (if provided)
        4. Filters out tokens shorter than min_token_length
        
        Parameters:
        - text: Text to tokenize
        - stopwords: Set of stopwords to filter out (default: None)
        - min_token_length: Minimum token length to keep (default: 1)
        
        Returns:
        - List of filtered tokens
        
        Usage example:
            tokens = self.tokenize_with_stopwords(text, self.estonian_stopwords, min_token_length=2)
        """
        if not text:
            return []
        
        if stopwords is None:
            stopwords = set()
        
        # Extract word tokens using regex
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter by stopwords and minimum length
        tokens = [token for token in tokens if token not in stopwords and len(token) >= min_token_length]
        
        return tokens

    def _initialize_semantic_models(self):
        """Initialize multilingual models."""
        logger.info("🧠 Initializing semantic models...")

        # Use intfloat/multilingual-e5-large-instruct for both sentence transformer and BERT embeddings
        try:
            from sentence_transformers import SentenceTransformer
            self.semantic_models['sentence_transformer'] = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
            logger.info("✅ Multilingual sentence transformer loaded (intfloat/multilingual-e5-large-instruct)")
        except Exception as e:
            logger.info(f"⚠️  Sentence transformer loading failed: {e}")
            self.semantic_models['sentence_transformer'] = None

        # Multilingual BERT for embeddings (same model)
        try:
            from transformers import AutoModel, AutoTokenizer
            self.semantic_models['mbert_tokenizer'] = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
            self.semantic_models['mbert_model'] = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')
            logger.info("✅ Multilingual BERT loaded (intfloat/multilingual-e5-large-instruct)")
        except Exception as e:
            logger.info(f"⚠️  Multilingual BERT loading failed: {e}")
            self.semantic_models['mbert_tokenizer'] = None
            self.semantic_models['mbert_model'] = None

    def self_bleu(self, texts: List[str], n_gram: int = 4, stopwords: Optional[Set[str]] = None) -> Dict[str, float]:
        """
        Language-agnostic Self-BLEU score calculation for text diversity analysis.
        Ref: https://arxiv.org/pdf/1802.01886 (Zhu et al. 2018)

        What it measures:
        Self-BLEU measures how similar texts are to each other within the same dataset.
        For each text, calculates BLEU score against all other texts in the collection.
        Higher Self-BLEU indicates lower diversity (more formulaic/repetitive patterns).

        Range: 0.0 - 1.0
        Better: Lower values indicate more diversity

        Reference Values (dependent on domain, text length, n_gram, number of texts):
        - 0.0-0.2: High diversity, very different texts
        - 0.2-0.4: Moderate diversity, some variation
        - 0.4-0.6: Low diversity, similar patterns
        - 0.6-0.8: Very low diversity, formulaic
        - 0.8-1.0: Extremely formulaic, nearly identical

        Parameters:
        - texts: List of text strings to analyze
        - n_gram: Maximum n-gram order for BLEU (1-4, default 4)
        - stopwords: Optional set of words to exclude. If None, no stopword filtering applied.

        Returns:
        Dictionary with Self-BLEU statistics:
        - self_bleu: Average Self-BLEU score (main metric)
        - std: Standard deviation of scores (reliability indicator)
        - median: Median score (robust central tendency)
        - min: Minimum score (most diverse pair)
        - max: Maximum score (most similar pair)
        - num_texts: Total texts compared
        - num_empty: Number of empty/invalid texts skipped
        - num_duplicates: Number of duplicate texts detected

        Raises:
        - ValueError: If insufficient valid texts, too many empty texts (>20%), 
                     invalid n_gram parameter, or NLTK BLEU scorer unavailable
        """
        # Input validation
        if not texts:
            raise ValueError("Empty texts list provided")
        
        if n_gram < 1 or n_gram > 4:
            raise ValueError(f"n_gram must be 1-4, got {n_gram}")
        
        if len(texts) > 10000:
            logger.warning(f"Large dataset ({len(texts)} texts) - Self-BLEU calculation may be slow")

        # Validate NLTK BLEU scorer availability
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        except ImportError:
            raise ValueError("NLTK BLEU scorer required for Self-BLEU calculation. Install: pip install nltk")
        
        smoothie = SmoothingFunction().method4

        # Track empty texts
        empty_indices = []
        tokenized_texts = []

        # Tokenize texts (basic word-level tokenization - language agnostic)
        for idx, text in enumerate(texts):
            if not text or not text.strip():
                empty_indices.append(idx)
                continue

            # Basic tokenization: split on whitespace and filter
            tokens = [word.lower() for word in text.split() if word.isalpha()]

            # Apply stopword filtering if provided
            if stopwords:
                tokens = [token for token in tokens if token not in stopwords]

            if tokens:
                tokenized_texts.append(tokens)

        # Validate empty ratio
        if empty_indices:
            empty_percentage = (len(empty_indices) / len(texts)) * 100
            if empty_percentage > 20:
                raise ValueError(
                    f"Too many empty texts: {len(empty_indices)}/{len(texts)} ({empty_percentage:.1f}%) are empty. "
                    f"First indices: {empty_indices[:10]}"
                    + ("..." if len(empty_indices) > 10 else "")
                )
            logger.warning(
                f"⚠️  {len(empty_indices)} empty/invalid texts skipped ({empty_percentage:.1f}% of {len(texts)})"
            )

        if len(tokenized_texts) < 2:
            raise ValueError(
                f"Insufficient valid texts: {len(tokenized_texts)} valid after filtering, need ≥2"
            )

        # Detect duplicates
        unique_texts = set(tuple(tokens) for tokens in tokenized_texts)
        num_duplicates = len(tokenized_texts) - len(unique_texts)

        if num_duplicates > 0:
            logger.warning(
                f"⚠️  {num_duplicates} duplicate texts detected. "
                f"Unique texts: {len(unique_texts)}/{len(tokenized_texts)}"
            )

        # Calculate Self-BLEU scores
        bleu_scores = []
        failed_indices = []

        # Set n-gram weights
        if n_gram == 1:
            weights = (1.0,)
        elif n_gram == 2:
            weights = (0.5, 0.5)
        elif n_gram == 3:
            weights = (1./3, 1./3, 1./3)
        else:  # n_gram >= 4
            weights = (0.25, 0.25, 0.25, 0.25)

        # For each text, calculate BLEU against all others
        for i, candidate_tokens in enumerate(tokenized_texts):
            # Get all other texts as references
            reference_tokens = [tokenized_texts[j] for j in range(len(tokenized_texts)) if i != j]

            if not reference_tokens:
                continue

            try:
                # Calculate BLEU score
                score = sentence_bleu(
                    reference_tokens,
                    candidate_tokens,
                    weights=weights,
                    smoothing_function=smoothie
                )
                bleu_scores.append(score)

            except (ValueError, ZeroDivisionError) as e:
                logger.warning(f"⚠️  Text {i} failed BLEU calculation: {e}")
                failed_indices.append(i)
                continue
            except Exception as e:
                logger.error(f"❌ Unexpected error in BLEU calculation for text {i}: {e}")
                failed_indices.append(i)
                continue

        # Validate calculation success
        if len(failed_indices) > len(tokenized_texts) * 0.1:
            raise ValueError(
                f"Too many texts failed BLEU calculation: {len(failed_indices)}/{len(tokenized_texts)} "
                f"({len(failed_indices)*100/len(tokenized_texts):.1f}%). "
                f"Check text quality and tokenization."
            )

        if not bleu_scores:
            raise ValueError(
                f"No Self-BLEU scores calculated despite {len(tokenized_texts)} valid texts. "
                "Possible issue: all texts too short or identical."
            )

        bleu_scores = np.array(bleu_scores)

        return {
            'self_bleu': float(np.mean(bleu_scores)),
            'std': float(np.std(bleu_scores)),
            'median': float(np.median(bleu_scores)),
            'min': float(np.min(bleu_scores)),
            'max': float(np.max(bleu_scores)),
            'num_texts': len(tokenized_texts),
            'num_empty': len(empty_indices),
            'num_duplicates': num_duplicates
        }

    def create_model_metrics_table(
        self,
        model_results: Dict[str, pd.DataFrame],
        language_name: str = "Language",
        language_flag: str = "",
        console: Optional['Console'] = None
    ) -> None:
        """
        Create and display a rich table comparing paper metrics across models.
        Shows only: TTR, MATTR, Full/Agent/Client Self-BLEU, and Intra Model Similarity.

        This is a generic implementation that can be used by all language-specific classes.

        Parameters:
        - model_results: Dictionary mapping model names to DataFrames of results
        - language_name: Language name for table title (e.g., "Estonian", "Hungarian", "Finnish")
        - language_flag: Optional emoji flag for table title (e.g., "🇪🇪", "🇭🇺", "🇫🇮")
        - console: Optional Rich Console object for custom rendering
        """
        from rich.console import Console
        from rich.table import Table
        
        if console is None:
            console = Console()

        # Calculate aggregate statistics per model
        model_stats = {}
        for model_name, df in model_results.items():
            model_stats[model_name] = {
                'conversations': len(df),
                'ttr': df['ttr'].mean() if 'ttr' in df.columns else 0,
                'mattr': df['mattr'].mean() if 'mattr' in df.columns else 0,
                'full_self_bleu': df['dataset_full_conversation_self_bleu_self_bleu'].iloc[0] if 'dataset_full_conversation_self_bleu_self_bleu' in df.columns and len(df) > 0 else 0,
                'agent_self_bleu': df['dataset_agent_response_self_bleu_self_bleu'].iloc[0] if 'dataset_agent_response_self_bleu_self_bleu' in df.columns and len(df) > 0 else 0,
                'client_self_bleu': df['dataset_client_response_self_bleu_self_bleu'].iloc[0] if 'dataset_client_response_self_bleu_self_bleu' in df.columns and len(df) > 0 else 0,
                'intra_avg_sim': df['avg_similarity'].iloc[0] if 'avg_similarity' in df.columns and len(df) > 0 else 0,
                'intra_std_sim': df['std_similarity'].iloc[0] if 'std_similarity' in df.columns and len(df) > 0 else 0
            }

        # Create Rich table with paper metrics only
        title = f"{language_flag} {language_name} Conversation Metrics by Model".strip()
        table = Table(title=title, show_header=True, header_style="bold magenta")
        
        # Add columns - only paper metrics
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Conversations", justify="right")
        table.add_column("TTR", justify="right", style="green")
        table.add_column("MATTR", justify="right", style="green")
        table.add_column("Full Self-BLEU", justify="right", style="red")
        table.add_column("Agent Self-BLEU", justify="right", style="red")
        table.add_column("Client Self-BLEU", justify="right", style="red")
        table.add_column("Intra Avg Sim", justify="right", style="blue")
        table.add_column("Intra Std Sim", justify="right", style="blue")

        # Add rows
        for model_name, stats in model_stats.items():
            table.add_row(
                model_name,
                str(stats['conversations']),
                f"{stats['ttr']:.3f}",
                f"{stats['mattr']:.3f}",
                f"{stats['full_self_bleu']:.3f}",
                f"{stats['agent_self_bleu']:.3f}",
                f"{stats['client_self_bleu']:.3f}",
                f"{stats['intra_avg_sim']:.3f}",
                f"{stats['intra_std_sim']:.3f}"
            )

        console.print(table)
        console.print("\n[bold cyan]Paper Metric Interpretation Guide:[/bold cyan]")
        console.print("• [green]TTR (Type-Token Ratio)[/green]: 0.0-1.0, higher = more diverse vocabulary")
        console.print("• [green]MATTR (Moving Average TTR)[/green]: 0.0-1.0, higher = more consistent lexical diversity")
        console.print("• [red]Self-BLEU[/red]: 0.0-1.0, [bold]lower = more diverse[/bold], higher = more formulaic")
        console.print("  - Full: Entire conversation diversity")
        console.print("  - Agent: Agent response diversity")
        console.print("  - Client: Client response diversity")
        console.print("• [blue]Intra-Model Similarity[/blue]: 0.0-1.0, semantic similarity within model conversations")
        console.print("\n[yellow]Note: Self-BLEU measures formulaic patterns - lower scores indicate better diversity[/yellow]")
