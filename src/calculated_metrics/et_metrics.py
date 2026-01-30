"""
Estonian Conversation Metrics Module

This module provides Estonian-specific conversation diversity and quality metrics.
Uses EstNLTK for proper Estonian language processing when available.

Requirements:
    pip install numpy pandas nltk transformers sentence-transformers torch estnltk rich
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
import re
import logging
from rich.console import Console

# Import parent class and data structures from conversation_metrics module
from conversation_metrics import ConversationMetrics, ConversationData

# Configure logging
logger = logging.getLogger(__name__)

class EstonianConversationMetrics(ConversationMetrics):
    """Main class for analyzing Estonian conversation diversity and quality."""

    def __init__(self):
        """Initialize Estonian-specific conversation metrics."""
        super().__init__()
        self.estonian_stopwords = set()
        self.estnltk_available = False
        self.estnltk_text = None
        self.estnltk_syntax = None
        
        # Set Estonian-specific speaker pattern
        self.speaker_pattern = r'(Agent|Klient|Kasutaja|Abistaja|Klienditeenindaja):'

        logger.info("🇪🇪 Initializing Estonian Conversation Diversity Analysis Tool")
        logger.info("=" * 50)

        self._setup_estnltk()
        self._setup_dependencies()
        self._initialize_semantic_models()

    def _setup_estnltk(self):
        """Setup EstNLTK for proper Estonian language processing."""
        logger.info("🇪🇪 Setting up EstNLTK for Estonian processing...")

        try:
            import estnltk
            from estnltk import Text
            from estnltk.taggers.standard.text_segmentation.paragraph_tokenizer import ParagraphTokenizer
            from estnltk.taggers.standard.text_segmentation.sentence_tokenizer import SentenceTokenizer
            from estnltk.taggers.standard.text_segmentation.whitespace_tokens_tagger import WhiteSpaceTokensTagger
            from estnltk.taggers.standard.text_segmentation.word_tagger import WordTagger
            from estnltk.taggers.standard.morph_analysis.morf import VabamorfTagger

            # Try compound tokenizer
            try:
                from estnltk.taggers import CompoundTokenTagger
                self.compound_tokenizer = CompoundTokenTagger()
                compound_available = True
            except ImportError:
                compound_available = False
            if compound_available:
                logger.info("✅ EstNLTK compound analysis available")
            else:
                logger.info("⚠️ EstNLTK compound analysis not available")
                
            self.estnltk_text = Text
            self.estnltk_available = True

            logger.info("✅ EstNLTK loaded successfully")

            # Initialize Estonian-specific taggers with correct API
            self.paragraph_tokenizer = ParagraphTokenizer()
            self.sentence_tokenizer = SentenceTokenizer()
            self.whitespace_tokenizer = WhiteSpaceTokensTagger()
            self.word_tokenizer = WordTagger()
            self.morph_analyzer = VabamorfTagger()

            # Try to enable basic syntax analysis with morph_extended
            self.estnltk_syntax = False
            self.syntax_tagger = None
            self.syntax_layer_name = 'morph_extended'  # Use morph_extended as basic syntax

            # Enable syntax analysis using morph_extended layer
            try:
                # We'll use VabamorfTagger with morph_extended as a basic syntax substitute
                self.morph_extended_tagger = VabamorfTagger(output_layer='morph_extended', use_postanalysis=True)
                self.estnltk_syntax = True
                logger.info("✅ EstNLTK basic syntax analysis enabled (morphological analysis)")
            except Exception as e:
                logger.info(f"⚠️  EstNLTK basic syntax analysis failed: {e}")
                self.estnltk_syntax = False

        except ImportError as e:
            logger.info(f"❌ EstNLTK not available: {e}")
            logger.info("    EstNLTK is required for proper Estonian language processing.")
            self.estnltk_available = False
            self.estnltk_text = None
            self.estnltk_syntax = False
            return None

    def _setup_dependencies(self):
        """Setup basic language dependencies."""
        logger.info("📚 Setting up basic language dependencies...")

        # Load comprehensive Estonian stopwords
        self.estonian_stopwords = {
            # Common conjunctions and particles
            'ja', 'või', 'ning', 'aga', 'kuid', 'ent', 'kui', 'et', 'sest',
            'siis', 'nii', 'nagu', 'kuna', 'kuigi', 'ehkki', 'vaid', 'ainult',
            # Pronouns
            'ma', 'sa', 'ta', 'me', 'te', 'nad', 'tema', 'teie', 'nende',
            'see', 'need', 'selle', 'nende', 'seda', 'neid', 'sellest', 'neist',
            'sellele', 'neile', 'selles', 'neis', 'sellega', 'nendega',
            # Question words
            'mis', 'kes', 'kus', 'kuhu', 'kust', 'millal', 'kuidas', 'miks',
            'mitu', 'milline', 'milliseid', 'kellele', 'kellega', 'kellelt',
            # Common verbs (auxiliary and modal)
            'on', 'ole', 'olema', 'olen', 'oled', 'oleme', 'olete',
            'pole', 'polegi', 'ei', 'ära', 'mitte',
            'saab', 'saama', 'saan', 'saad', 'saame', 'saate', 'saavad',
            'võib', 'võima', 'võin', 'võid', 'võime', 'võite', 'võivad',
            # Prepositions and adverbs
            'ka', 'veel', 'juba', 'alati', 'kunagi', 'mitte kunagi',
            'väga', 'palju', 'vähe', 'liiga', 'üsna', 'päris', 'peaaegu',
            'tõesti', 'kindlasti', 'ilmselt', 'võib-olla', 'loodetavasti',
            # Quantifiers
            'kõik', 'midagi', 'keegi', 'miski', 'igaüks', 'mõni', 'iga',
            'mõningad', 'paljud', 'vähesed', 'enamik', 'osad'
        }

        # Try to augment with NLTK Estonian stopwords
        try:
            try:
                with open('stopwords/et/estonian_stopwords.txt', 'r', encoding='utf-8') as et_stopwords:
                    words = et_stopwords.read().splitlines()
                self.estonian_stopwords.update(set(words))
                logger.info("✅ Estonian stopwords enhanced with Estonian stopword list by Kristel Uiboaed")
            except OSError:
                logger.info("⚠️  Using built-in Estonian stopwords list")
        except Exception as e:
            logger.info(f"⚠️  Using built-in Estonian stopwords: {e}")

        logger.info(f"📝 Loaded {len(self.estonian_stopwords)} Estonian stopwords")

    def clean_text(self, text: str) -> str:
        """Clean and normalize Estonian conversation text (overrides parent method)."""
        # Call parent clean_text first
        text = super().clean_text(text)
        if not text:
            return ""
        # Estonian-specific cleaning - remove chat artifacts
        text = re.sub(r'\b(hmmm|mhm|ahaa|aha|okei|ok)\b', '', text, flags=re.IGNORECASE)
        return text

    def _build_estnltk_pipeline(self, text: str):
        """Build complete EstNLTK pipeline with all required layers."""
        if not self.estnltk_available or not text:
            return None

        try:
            # Create EstNLTK Text object
            est_text = self.estnltk_text(text)

            # Apply the basic pipeline in correct dependency order
            # 1. WhiteSpaceTokensTagger creates tokens (no dependencies)
            est_text = self.whitespace_tokenizer.tag(est_text)

            # 2. Add compound_tokens if compound tokenizer available
            if hasattr(self, 'compound_tokenizer'):
                try:
                    est_text = self.compound_tokenizer.tag(est_text)
                except Exception:
                    # If compound tokenizer fails, create empty compound_tokens layer
                    pass

            # 3. WordTagger creates words (needs tokens and compound_tokens)
            est_text = self.word_tokenizer.tag(est_text)

            # 4. SentenceTokenizer creates sentences (needs words and compound_tokens)
            est_text = self.sentence_tokenizer.tag(est_text)

            # 5. ParagraphTokenizer creates paragraphs (needs sentences)
            est_text = self.paragraph_tokenizer.tag(est_text)

            # 6. Morphological analysis (needs words and sentences)
            est_text = self.morph_analyzer.tag(est_text)

            # 7. Apply morph_extended analysis if syntax analysis is enabled
            if self.estnltk_syntax and hasattr(self, 'morph_extended_tagger'):
                est_text = self.morph_extended_tagger.tag(est_text)
            return est_text

        except Exception as e:
            logger.info(f"⚠️  EstNLTK pipeline failed: {e}")
            return None

    def extract_estonian_turns(self, conversation_input: Union[str, ConversationData], speaker_pattern: str = None) -> List[Dict[str, str]]:
        """
        Extract conversation turns with Estonian speaker labels.
        This is an alias for extract_turns() for backward compatibility.
        """
        return self.extract_turns(conversation_input, speaker_pattern)

    def estonian_type_token_ratio(self, text: str) -> float: #USED
        """
        Calculate Type-Token Ratio (TTR) for Estonian text - measures lexical diversity.

        What it measures:
        TTR = unique words / total words. Higher values indicate greater vocabulary diversity.
        Uses EstNLTK for proper Estonian morphological analysis when available.

        Range: 0.0 - 1.0
        - 0.0: No diversity (impossible in practice)
        - 1.0: Every word is unique (maximum diversity)

        Better: Higher is better (more diverse vocabulary)

        Reference Values:
        - 0.3-0.5: Low diversity, repetitive language
        - 0.5-0.7: Moderate diversity, typical conversation
        - 0.7-0.9: High diversity, rich vocabulary
        - 0.9+: Very high diversity, academic/literary text

        Estonian-specific: Accounts for Estonian stopwords and morphological variants.
        """
        if not text:
            return 0.0

        # Use EstNLTK pipeline if available
        est_text = self._build_estnltk_pipeline(text)
        if est_text:
            try:
                if hasattr(est_text, 'words') and est_text.words:
                    tokens = [word.text.lower() for word in est_text.words
                            if word.text.isalpha() and len(word.text) > 1 and word.text.lower() not in self.estonian_stopwords]
                else:
                    tokens = []

                if not tokens:
                    return 0.0

                return len(set(tokens)) / len(tokens)
            except Exception as e:
                logger.info(f"⚠️  EstNLTK tokenization failed: {e}, falling back to regex")

        # Fallback to regex tokenization using parent class method
        # Use min_token_length=2 to match EstNLTK's len(word.text) > 1 filter (exclude single chars)
        tokens = self.tokenize_with_stopwords_and_regex(text, self.estonian_stopwords, min_token_length=2)

        if not tokens:
            return 0.0

        return len(set(tokens)) / len(tokens)

    def estonian_moving_average_ttr(self, text: str, window_size: int = 100) -> float: #USED
        """
        Calculate Moving Average Type-Token Ratio (MATTR) for Estonian text.

        What it measures:
        MATTR addresses TTR's text-length dependency by calculating TTR across
        sliding windows of fixed size, then averaging. More stable than basic TTR.

        Range: 0.0 - 1.0
        Better: Higher is better (more consistent lexical diversity)

        Reference Values:
        - 0.4-0.6: Low lexical diversity, repetitive
        - 0.6-0.8: Moderate diversity, natural conversation
        - 0.8-0.95: High diversity, rich vocabulary
        - 0.95+: Exceptionally diverse, likely academic text

        Parameters:
        - window_size: Default 100 words (standard in research)

        Advantages over TTR:
        - Length-independent
        - More reliable for comparing texts of different lengths
        """
        if not text:
            return 0.0

        # Use EstNLTK pipeline if available
        est_text = self._build_estnltk_pipeline(text)
        if est_text:
            try:
                if hasattr(est_text, 'words') and est_text.words:
                    tokens = [word.text.lower() for word in est_text.words
                            if word.text.isalpha() and len(word.text) > 1 and word.text.lower() not in self.estonian_stopwords]
                else:
                    tokens = []
            except Exception as e:
                logger.info(f"⚠️  EstNLTK tokenization failed: {e}, falling back to regex")
                # Fallback to regex tokenization using parent class method
                # Note: min_token_length=1 means "length > 1" (i.e., minimum 2 chars) due to > operator in parent method
                tokens = self.tokenize_with_stopwords_and_regex(text, self.estonian_stopwords, min_token_length=1)
        else:
            # Fallback to regex tokenization using parent class method
            # Note: min_token_length=1 means "length > 1" (i.e., minimum 2 chars) due to > operator in parent method
            tokens = self.tokenize_with_stopwords_and_regex(text, self.estonian_stopwords, min_token_length=1)

        # If text is shorter than window size, calculate simple TTR directly (avoid inefficient recursion)
        if len(tokens) < window_size:
            return len(set(tokens)) / len(tokens) if tokens else 0.0

        ttrs = []
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i + window_size]
            window_ttr = len(set(window)) / len(window)
            ttrs.append(window_ttr)

        return np.mean(ttrs)

    def estonian_self_bleu(self, texts: List[str], n_gram: int = 4) -> Dict[str, float]: #USED
        """
        Calculate Self-BLEU score with Estonian-specific preprocessing.

        Complete override of parent's self_bleu() method with optimized Estonian processing:
        - EstNLTK morphological tokenization when available
        - Estonian stopword filtering (single-pass, no redundancy)
        - Direct BLEU calculation on pre-tokenized lists (no join-split inefficiency)

        What it measures:
        Self-BLEU measures how similar texts are to each other within the same dataset.
        For each text, calculates BLEU score against all other texts in the collection.
        Higher Self-BLEU indicates lower diversity (more formulaic/repetitive patterns).

        Range: 0.0 - 1.0 (self_bleu key in returned dict)
        Better: Lower values indicate more diversity

        Reference Values (dependent on domain, text length, n_gram, number of texts):
        - 0.0-0.2: High diversity, very different texts
        - 0.2-0.4: Moderate diversity, some variation
        - 0.4-0.6: Low diversity, similar patterns
        - 0.6-0.8: Very low diversity, formulaic
        - 0.8-1.0: Extremely formulaic, nearly identical

        Parameters:
        - texts: List of Estonian text strings to analyze
        - n_gram: Maximum n-gram order for BLEU (1-4, default 4)

        Returns:
        Dictionary with Self-BLEU statistics:
        - self_bleu: Average Self-BLEU score (main metric)
        - std: Standard deviation of scores
        - median: Median score
        - min: Minimum score (most diverse pair)
        - max: Maximum score (most similar pair)
        - num_texts: Total texts compared
        - num_empty: Number of empty/invalid texts skipped
        - num_duplicates: Number of duplicate texts detected

        Use case: Detecting formulaic patterns in Estonian customer service conversations
        
        Raises:
        - ValueError: If input validation fails or NLTK unavailable
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

        # Track empty texts and pre-tokenize with Estonian processing
        empty_indices = []
        tokenized_texts = []

        for idx, text in enumerate(texts):
            if not text or not text.strip():
                empty_indices.append(idx)
                continue

            # Estonian-specific tokenization
            if self.estnltk_available:
                try:
                    est_text = self._build_estnltk_pipeline(text)
                    if est_text and hasattr(est_text, 'words'):
                        # Filter: isalpha(), len > 1 (exclude single chars), exclude stopwords
                        tokens = [word.text.lower() for word in est_text.words
                                if word.text.isalpha() and len(word.text) > 1 and word.text.lower() not in self.estonian_stopwords]
                        if tokens:
                            tokenized_texts.append(tokens)
                        else:
                            empty_indices.append(idx)
                        continue
                except Exception as e:
                    logger.warning(f"EstNLTK preprocessing failed for text {idx}: {e}, using fallback")

            # Fallback: regex-based tokenization with Estonian stopwords
            tokens = [word.lower() for word in re.findall(r'\b\w+\b', text)
                    if word.lower() not in self.estonian_stopwords and len(word) > 1]
            
            if tokens:
                tokenized_texts.append(tokens)
            else:
                empty_indices.append(idx)

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
                # Calculate BLEU score directly on pre-tokenized lists
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


    def calculate_intra_model_conversation_similarity(self, conversations: List[Union[str, ConversationData]], model_name: str = None) -> Dict[str, float]:
        """
        Calculate pairwise conversation similarity within a single model without repeating combinations.

        What it measures:
        Computes semantic similarity between all unique pairs of conversations from the same model
        using multilingual sentence embeddings. Helps identify how diverse vs. formulaic a model's
        conversation generation is.

        Parameters:
        - conversations: List of conversations from the same model
        - model_name: Optional model name for logging

        Returns:
        Dictionary with similarity statistics:
        - avg_similarity: Mean pairwise similarity
        - max_similarity: Highest pairwise similarity
        - min_similarity: Lowest pairwise similarity
        - std_similarity: Standard deviation of similarities
        - median_similarity: Median pairwise similarity
        - num_comparisons: Total number of pairwise comparisons made
        - num_duplicates: Number of duplicate conversations detected
        - num_empty: Number of empty conversations skipped

        Range: -1.0 to 1.0 (cosine similarity)
        Lower average = more diverse conversations
        Higher average = more formulaic conversations

        Reference Values:
        - < 0.3: High diversity, very different conversations
        - 0.3-0.5: Moderate diversity, some variation
        - 0.5-0.7: Low diversity, similar patterns
        - > 0.7: Very formulaic, template-like conversations
        
        Raises:
        - ValueError: If semantic model not initialized, insufficient conversations,
                    or too many empty conversations (>20%)
        """
        sentence_model = self.semantic_models.get('sentence_transformer')
        if sentence_model is None:
            raise ValueError("Semantic model not initialized - sentence transformer required")
        
        if not conversations:
            raise ValueError("Empty conversations list provided")
        
        if len(conversations) < 2:
            raise ValueError(f"Insufficient conversations: {len(conversations)} provided, need ≥2")

        model_label = f" for {model_name}" if model_name else ""
        logger.info(f"🔄 Processing {len(conversations)} conversations{model_label}...")

        # Convert all conversations to text format and track empty ones
        conv_texts = []
        empty_indices = []
        for idx, conv in enumerate(conversations):
            # Use duck typing to avoid __main__ vs module import issues
            if hasattr(conv, 'formatted_string'):
                text = conv.formatted_string
            else:
                text = conv
            
            if not text or not text.strip():
                empty_indices.append(idx)
                conv_texts.append("")
            else:
                conv_texts.append(text)
        
        # Validate empty conversation ratio
        if empty_indices:
            empty_percentage = (len(empty_indices) / len(conversations)) * 100
            if empty_percentage > 20:
                raise ValueError(
                    f"Too many empty conversations: {len(empty_indices)}/{len(conversations)} "
                    f"({empty_percentage:.1f}%) are empty. Indices: {empty_indices[:10]}"
                    + ("..." if len(empty_indices) > 10 else "")
                )
            logger.warning(
                f"⚠️  {len(empty_indices)} empty conversations skipped "
                f"({empty_percentage:.1f}% of {len(conversations)})"
            )
        
        # Filter out empty conversations for processing
        valid_texts = [(idx, text) for idx, text in enumerate(conv_texts) if text.strip()]
        
        if len(valid_texts) < 2:
            raise ValueError(
                f"Insufficient non-empty conversations: {len(valid_texts)} valid, need ≥2. "
                f"Empty indices: {empty_indices}"
            )
        
        # Deduplicate conversations
        unique_texts = {}
        duplicate_count = 0
        for idx, text in valid_texts:
            if text not in unique_texts:
                unique_texts[text] = idx
            else:
                duplicate_count += 1
        
        if duplicate_count > 0:
            logger.warning(
                f"⚠️  {duplicate_count} duplicate conversations detected and removed. "
                f"Unique conversations: {len(unique_texts)}"
            )
        
        if len(unique_texts) < 2:
            raise ValueError(
                f"Insufficient unique conversations: {len(unique_texts)} unique after deduplication, need ≥2. "
                f"Duplicates: {duplicate_count}"
            )
        
        # Pre-compute embeddings once (fixes catastrophic 99x redundancy)
        unique_texts_list = list(unique_texts.keys())
        n = len(unique_texts_list)
        
        logger.info(f"🧮 Pre-computing embeddings for {n} unique conversations...")
        try:
            embeddings = sentence_model.encode(unique_texts_list, show_progress_bar=False)
        except Exception as e:
            raise ValueError(f"Embedding computation failed: {e}")
        
        # Calculate all unique pairwise similarities using pre-computed embeddings
        similarities = []
        num_comparisons = n * (n - 1) // 2
        
        logger.info(f"📊 Calculating {num_comparisons} pairwise similarities{model_label}...")
        
        for i in range(n):
            for j in range(i + 1, n):
                # Cosine similarity using pre-computed embeddings
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(float(sim))
        
        if not similarities:
            raise ValueError(
                f"No similarities calculated despite {n} unique conversations. "
                "This should not happen - possible embedding computation issue."
            )
        
        similarities = np.array(similarities)
        
        logger.info(
            f"✅ Analysis complete{model_label}: "
            f"avg={np.mean(similarities):.3f}, "
            f"median={np.median(similarities):.3f}, "
            f"std={np.std(similarities):.3f}"
        )

        return {
            'avg_similarity': float(np.mean(similarities)),
            'max_similarity': float(np.max(similarities)),
            'min_similarity': float(np.min(similarities)),
            'std_similarity': float(np.std(similarities)),
            'median_similarity': float(np.median(similarities)),
            'num_comparisons': len(similarities),
            'num_duplicates': duplicate_count,
            'num_empty': len(empty_indices)
        }

    def estonian_conversation_self_bleu(self, conversation_inputs: List[Union[str, ConversationData]], analysis_level: str = 'full') -> Dict[str, float]:
        """
        Calculate Self-BLEU scores for Estonian conversations at different granularity levels.

        What it measures:
        Analyzes formulaic patterns in Estonian conversations by comparing:
        1. Full conversations (entire dialogue as single text)
        2. Agent responses only (consistency of agent language)
        3. Client responses only (diversity of client language)
        4. Individual turns (turn-level formulaic patterns)

        Parameters:
        - conversation_inputs: List of conversations (strings or ConversationData objects)
        - analysis_level: 'full', 'agent', 'client', 'turns', or 'all'

        Returns:
        Dictionary with Self-BLEU scores for requested analysis levels

        Lower scores = more diverse, higher scores = more formulaic

        Estonian conversation patterns:
        - Agent responses often formulaic (higher Self-BLEU expected)
        - Client responses should be diverse (lower Self-BLEU expected)
        - Overall conversations should balance structure and diversity
        """
        results = {}

        if not conversation_inputs:
            return results

        # Extract different text types from conversations
        full_conversations = []
        agent_responses = []
        client_responses = []
        all_turns = []

        for conv_input in conversation_inputs:
            # Extract turns using existing method
            turns = self.extract_estonian_turns(conv_input)

            if not turns:
                continue

            # Full conversation text
            # Use duck typing to avoid __main__ vs module import issues
            if hasattr(conv_input, 'formatted_string'):
                full_text = conv_input.formatted_string
            else:
                full_text = conv_input
            full_conversations.append(full_text)

            # Separate agent and client responses
            for turn in turns:
                speaker = turn['speaker'].lower()
                text = turn['text']
                # Get is_agent field from the original conversation data if available
                is_agent = turn.get('is_agent', None)

                if text.strip():
                    all_turns.append(text)

                    # Use is_agent field if available (for gpt-4.1-mini and similar data)
                    if is_agent is True:
                        agent_responses.append(text)
                    elif is_agent is False:
                        client_responses.append(text)
                    # Fallback: classify based on speaker name keywords
                    elif any(agent_word in speaker for agent_word in ['agent', 'abistaja', 'klienditeenindaja', 'teenindaja']):
                        agent_responses.append(text)
                    elif any(client_word in speaker for client_word in ['klient', 'kasutaja', 'client', 'customer']):
                        client_responses.append(text)

        # Calculate Self-BLEU for requested analysis levels
        if analysis_level == 'all' or analysis_level == 'full':
            if len(full_conversations) >= 2:
                results['full_conversation_self_bleu'] = self.estonian_self_bleu(full_conversations)

        if analysis_level == 'all' or analysis_level == 'agent':
            if len(agent_responses) >= 2:
                results['agent_response_self_bleu'] = self.estonian_self_bleu(agent_responses)

        if analysis_level == 'all' or analysis_level == 'client':
            if len(client_responses) >= 2:
                results['client_response_self_bleu'] = self.estonian_self_bleu(client_responses)

        if analysis_level == 'all' or analysis_level == 'turns':
            if len(all_turns) >= 2:
                results['all_turns_self_bleu'] = self.estonian_self_bleu(all_turns)

        return results

    def batch_analyze_estonian_conversations(
        self,
        conversations: List[Union[str, ConversationData]],
        progress_callback: Optional[callable] = None,
        include_self_bleu: bool = True
    ) -> pd.DataFrame:
        """
        Analyze multiple Estonian conversations.
        
        Calculates:
        - TTR (Type-Token Ratio)
        - MATTR (Moving Average TTR)
        - Self-BLEU (Full, Agent, Client)
        """
        results = []

        logger.info(f"🔍 Analyzing {len(conversations)} Estonian conversations for paper metrics...")

        # Calculate Self-BLEU scores for the entire dataset first
        self_bleu_results = {}
        if include_self_bleu and len(conversations) >= 2:
            logger.info("📊 Calculating Self-BLEU scores for dataset diversity...")
            try:
                self_bleu_results = self.estonian_conversation_self_bleu(conversations, analysis_level='all')
                logger.info(f"✅ Self-BLEU analysis complete: {len(self_bleu_results)} metrics calculated")
            except Exception as e:
                logger.info(f"⚠️  Self-BLEU calculation failed: {e}")

        # Calculate per-conversation metrics
        for i, conversation in enumerate(conversations):
            if progress_callback:
                progress_callback(i, len(conversations))
            elif i % max(1, len(conversations) // 10) == 0:
                logger.info(f"Progress: {i}/{len(conversations)} ({100*i/len(conversations):.1f}%)")

            # Convert ConversationData to text
            if hasattr(conversation, 'formatted_string'):
                conversation_text = conversation.formatted_string
            else:
                conversation_text = conversation

            # Calculate paper metrics only: TTR and MATTR
            flat_result = {
                'conversation_id': i,
                'ttr': self.estonian_type_token_ratio(conversation_text),
                'mattr': self.estonian_moving_average_ttr(conversation_text)
            }

            # Add dataset-level Self-BLEU scores to each row
            for analysis_level, bleu_stats in self_bleu_results.items():
                if isinstance(bleu_stats, dict):
                    for stat_name, stat_value in bleu_stats.items():
                        flat_result[f"dataset_{analysis_level}_{stat_name}"] = stat_value
                else:
                    flat_result[f"dataset_{analysis_level}"] = bleu_stats

            results.append(flat_result)

        logger.info("✅ Analysis complete!")
        return pd.DataFrame(results)

    def create_model_metrics_table(
        self,
        model_results: Dict[str, pd.DataFrame],
        console: Optional[Console] = None
    ) -> None:
        """
        Create and display a rich table comparing paper metrics across models.
        Delegates to parent class's generic implementation.

        Parameters:
        - model_results: Dictionary mapping model names to DataFrames of results
        - console: Optional Rich Console object for custom rendering
        """
        # Use parent class's generic implementation with Estonian-specific parameters
        super().create_model_metrics_table(
            model_results=model_results,
            language_name="Estonian",
            language_flag="🇪🇪",
            console=console
        )