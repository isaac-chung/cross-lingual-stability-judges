"""
Finnish Conversation Metrics Module

This module provides Finnish-specific conversation diversity and quality metrics.
Uses Stanza for proper Finnish language processing when available.

Requirements:
    pip install numpy pandas nltk transformers sentence-transformers torch stanza rich
    python -c "import stanza; stanza.download('fi')"
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
import re
import nltk
import logging
from rich.console import Console

# Import parent class and data structures from conversation_metrics module
from conversation_metrics import ConversationMetrics, ConversationData, ConversationTurn

# Configure logging
logger = logging.getLogger(__name__)

class FinnishConversationMetrics(ConversationMetrics):
    """Main class for analyzing Finnish conversation diversity and quality."""

    def __init__(self):
        """Initialize Finnish-specific conversation metrics."""
        super().__init__()
        self.finnish_stopwords = set()
        self.stanza_available = False
        self.nlp = None
        
        # Set Finnish-specific speaker pattern
        self.speaker_pattern = r'(Agentti|Asikas|Asiakas|Palvelunedustaja|Agent|Klient):'

        logger.info("🇫🇮 Initializing Finnish Conversation Diversity Analysis Tool")
        logger.info("=" * 50)

        self._setup_stanza()
        self._setup_dependencies()
        self._initialize_semantic_models()

    def _setup_stanza(self):
        """Setup Stanza for proper Finnish language processing."""
        logger.info("🇫🇮 Setting up Stanza for Finnish processing...")

        try:
            import stanza
            try:
                # Try to load the Finnish pipeline
                self.nlp = stanza.Pipeline(
                    'fi',
                    processors='tokenize,pos,lemma,depparse',
                    verbose=False,
                    download_method=None  # Don't auto-download
                )
                self.stanza_available = True
                logger.info("✅ Stanza loaded successfully for Finnish")
            except Exception as e:
                logger.info(f"⚠️  Stanza model not found: {e}")
                logger.info("💡 Install with: python -c \"import stanza; stanza.download('fi')\"")
                self.stanza_available = False
                self.nlp = None
        except ImportError as e:
            logger.info(f"⚠️  Stanza not available: {e}")
            logger.info("💡 Install with: pip install stanza")
            self.stanza_available = False
            self.nlp = None

    def _setup_dependencies(self):
        """Setup basic language dependencies."""
        logger.info("📚 Setting up basic language dependencies...")

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("📥 Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
            logger.info("✅ NLTK data downloaded")

        # Load comprehensive Finnish stopwords
        self.finnish_stopwords = {
            # Pronouns
            'minä', 'sinä', 'hän', 'me', 'te', 'he',
            'minun', 'sinun', 'hänen', 'meidän', 'teidän', 'heidän',
            'minua', 'sinua', 'häntä', 'meitä', 'teitä', 'heitä',
            'minussa', 'sinussa', 'hänessä', 'meissä', 'teissä', 'heissä',
            'minulla', 'sinulla', 'hänellä', 'meillä', 'teillä', 'heillä',
            'minulta', 'sinulta', 'häneltä', 'meiltä', 'teiltä', 'heiltä',
            'minulle', 'sinulle', 'hänelle', 'meille', 'teille', 'heille',
            'minusta', 'sinusta', 'hänestä', 'meistä', 'teistä', 'heistä',
            'itse', 'itseni', 'itsesi', 'itsensä', 'itsemme', 'itsenne',
            'kuka', 'mikä', 'joka', 'kumpi', 'mitä', 'ketä', 'kenen',
            'tämä', 'tuo', 'se', 'nämä', 'nuo', 'ne',
            'tämän', 'tuon', 'sen', 'näiden', 'noiden', 'niiden',
            'tätä', 'tuota', 'sitä', 'näitä', 'noita', 'niitä',
            'joku', 'jokin', 'jotain', 'jotakin', 'kukaan', 'mikään', 'mitään',
            
            # Conjunctions
            'ja', 'tai', 'mutta', 'vaan', 'että', 'jos', 'kun', 'koska', 'sillä',
            'sekä', 'eli', 'vai', 'kuin', 'jotta', 'vaikka', 'kunnes',
            
            # Prepositions
            'kanssa', 'mukaan', 'ilman', 'luona', 'luokse', 'luota',
            'alla', 'päällä', 'yli', 'alle', 'päälle', 'alta', 'päältä',
            'edessä', 'takana', 'vieressä', 'välissä', 'kesken',
            'ennen', 'jälkeen', 'aikana', 'ajan', 'läpi', 'kautta',
            'ympäri', 'kohti', 'vastaan', 'saakka', 'lähtien', 'alkaen',
            
            # Articles and determiners (Finnish has no articles, but includes demonstratives)
            'se', 'tämä', 'tuo', 'nämä', 'nuo', 'ne',
            'jokin', 'joku', 'kaikki', 'molemmat', 'kumpikin',
            
            # Question words
            'kuka', 'mikä', 'mitä', 'milloin', 'missä', 'mihin', 'mistä',
            'miten', 'miksi', 'kuinka', 'paljonko', 'kumpi', 'millainen',
            
            # Adverbs
            'ei', 'en', 'et', 'emme', 'ette', 'eivät',
            'on', 'olla', 'ole', 'olen', 'olet', 'olemme', 'olette', 'ovat',
            'oli', 'olin', 'olit', 'olimme', 'olitte', 'olivat',
            'hyvin', 'erittäin', 'aika', 'melko', 'varsin', 'todella', 'oikein',
            'nyt', 'sitten', 'niin', 'vielä', 'jo', 'aina', 'usein', 'joskus',
            'koskaan', 'tänään', 'eilen', 'huomenna',
            'täällä', 'siellä', 'tuolla', 'täältä', 'sieltä', 'tuolta',
            'tänne', 'sinne', 'tuonne',
            'myös', 'vain', 'ainoastaan', 'vasta', 'enää',
            'liian', 'melko', 'aivan', 'juuri', 'ehkä', 'varmaanki',
            'tietysti', 'toki', 'kyllä',
            
            # Quantifiers
            'kaikki', 'jokainen', 'joka', 'mikä', 'molemmat',
            'moni', 'muutama', 'useimmat', 'harva',
            'paljon', 'vähän', 'enemmän', 'vähemmän',
            'yksi', 'kaksi', 'kolme', 'ensimmäinen', 'toinen',
            
            # Common verbs
            'on', 'olla', 'oli', 'ollut', 'olisi',
            'voida', 'voi', 'voin', 'voit', 'voivat', 'voisi',
            'pitää', 'pitäisi', 'täytyä', 'täytyy',
            'saada', 'saa', 'sain', 'sait', 'saavat',
            'tehdä', 'tee', 'teen', 'teet', 'tekevät',
            'mennä', 'mene', 'menen', 'menet', 'menevät',
            'tulla', 'tule', 'tulen', 'tulet', 'tulevat',
            
            # Others
            'joo', 'kyllä', 'ei', 'ehkä', 'no', 'noh', 'hyvä', 'okei', 'selvä'
        }
        
        # Try to load additional Finnish stopwords from file if available
        try:
            try: #Gotten from: https://github.com/stopwords-iso/stopwords-fi/blob/master/stopwords-fi.txt
                with open('stopwords/fi/finnish_stopwords.txt', 'r', encoding='utf-8') as fi_stopwords:
                    words = fi_stopwords.read().splitlines()
                self.finnish_stopwords.update(set(words))
                logger.info("✅ Finnish stopwords enhanced with Finnish stopword list")
            except OSError:
                logger.info("⚠️  Using comprehensive built-in Finnish stopwords list")
        except Exception as e:
            logger.info(f"⚠️  Using built-in Finnish stopwords: {e}")
        logger.info(f"📝 Loaded {len(self.finnish_stopwords)} Finnish stopwords")

    def clean_text(self, text: str) -> str:
        """Clean and normalize Finnish conversation text (overrides parent method)."""
        # Call parent clean_text first
        text = super().clean_text(text)
        
        if not text:
            return ""

        # Finnish-specific cleaning - remove chat artifacts
        text = re.sub(r'\b(hmmm|mhm|aha|okei|ok|joo)\b', '', text, flags=re.IGNORECASE)

        return text

    def _process_with_stanza(self, text: str):
        """Process text with Stanza pipeline."""
        if not self.stanza_available or not text:
            return None

        try:
            doc = self.nlp(text)
            return doc
        except Exception as e:
            logger.warning(f"⚠️  Stanza processing failed: {e}")
            return None

    def _tokenize_finnish_text(self, text: str, min_token_length: int = 2) -> List[str]:
        """
        Tokenize Finnish text using Stanza with lemmatization and stopword filtering.
        
        Parameters:
        - text: Input text to tokenize
        - min_token_length: Minimum token length to include (default: 2)
        
        Returns:
        List of lemmatized, filtered tokens
        
        Note: This method ensures consistent tokenization across all lexical metrics.
        """
        if not text:
            return []
        
        # Use Stanza pipeline if available
        doc = self._process_with_stanza(text)
        if doc:
            # Use lemmas for better Finnish analysis (agglutinative language)
            tokens = []
            for sent in doc.sentences:
                for word in sent.words:
                    lemma_lower = word.lemma.lower()
                    # Filter: stopwords, length check on lemma
                    if lemma_lower not in self.finnish_stopwords and len(lemma_lower) >= min_token_length:
                        tokens.append(lemma_lower)
        else:
            # Fallback to regex tokenization using parent class method
            tokens = self.tokenize_with_stopwords_and_regex(
                text, 
                self.finnish_stopwords, 
                min_token_length=min_token_length
            )
        
        return tokens

    def extract_finnish_turns(self, conversation_input: Union[str, ConversationData], speaker_pattern: str = None) -> List[Dict[str, str]]:
        """
        Extract conversation turns with Finnish speaker labels.
        This is an alias for extract_turns() for backward compatibility.
        """
        return self.extract_turns(conversation_input, speaker_pattern)

    def finnish_type_token_ratio(self, text: str) -> float:
        """
        Calculate Type-Token Ratio (TTR) for Finnish text - measures lexical diversity.

        What it measures:
        TTR = unique words / total words. Higher values indicate greater vocabulary diversity.
        Uses Stanza for proper Finnish morphological analysis when available.

        Range: 0.0 - 1.0
        Better: Higher is better (more diverse vocabulary)

        Reference Values:
        - 0.3-0.5: Low diversity, repetitive language
        - 0.5-0.7: Moderate diversity, typical conversation
        - 0.7-0.9: High diversity, rich vocabulary
        - 0.9+: Very high diversity, academic/literary text

        Finnish-specific: Accounts for Finnish stopwords and agglutinative morphology.
        """
        # Input validation
        if not isinstance(text, str) or not text.strip():
            logger.debug("TTR: Empty or invalid text input")
            return 0.0

        # Clean text before processing (remove chat artifacts)
        cleaned_text = self.clean_text(text)
        if not cleaned_text.strip():
            logger.debug("TTR: Text became empty after cleaning")
            return 0.0

        # Tokenize using shared method (ensures consistency)
        tokens = self._tokenize_finnish_text(cleaned_text, min_token_length=2)

        if not tokens:
            logger.debug("TTR: No valid tokens after filtering")
            return 0.0

        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        ttr = unique_tokens / total_tokens
        
        logger.debug(f"TTR: {unique_tokens} unique / {total_tokens} total = {ttr:.3f}")
        return ttr

    def finnish_moving_average_ttr(self, text: str, window_size: int = 100) -> float:
        """
        Calculate Moving Average Type-Token Ratio (MATTR) for Finnish text.

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
        - text: Finnish text to analyze
        - window_size: Number of filtered tokens per window (default: 100)
                    Note: This is the count AFTER stopword removal and filtering.
        """
        if not isinstance(text, str) or not text.strip():
            logger.debug("MATTR: Empty or invalid text input")
            return 0.0

        # Clean text before processing
        cleaned_text = self.clean_text(text)
        if not cleaned_text.strip():
            logger.debug("MATTR: Text became empty after cleaning")
            return 0.0

        # Tokenize using shared method (ensures consistency with TTR)
        tokens = self._tokenize_finnish_text(cleaned_text, min_token_length=2)

        if not tokens:
            logger.debug("MATTR: No valid tokens after filtering")
            return 0.0

        if len(tokens) < window_size:
            # If text is shorter than window, return basic TTR as fallback
            ttr = len(set(tokens)) / len(tokens)
            logger.debug(f"MATTR: Text too short ({len(tokens)} < {window_size}), returning basic TTR: {ttr:.3f}")
            return ttr

        # Calculate TTR for each sliding window
        ttr_scores = []
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i + window_size]
            ttr = len(set(window)) / len(window)
            ttr_scores.append(ttr)

        mattr = np.mean(ttr_scores) if ttr_scores else 0.0
        logger.debug(f"MATTR: {len(ttr_scores)} windows, mean TTR = {mattr:.3f}")
        return mattr

    def finnish_conversation_self_bleu(self, conversation_inputs: List[Union[str, ConversationData]], analysis_level: str = 'full') -> Dict[str, float]:
        """
        Calculate Self-BLEU for Finnish conversations with multiple analysis levels.

        Complete override of parent's self_bleu() method with optimized Finnish processing:
        - Stanza morphological lemmatization when available
        - Finnish stopword filtering (single-pass, no redundancy)
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
        - conversation_inputs: List of conversation texts or ConversationData objects
        - analysis_level: 'full' (entire conversations), 'agent' (agent responses only),
                        'client' (client responses only)

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

        Use case: Detecting formulaic patterns in Finnish customer service conversations
        
        Raises:
        - ValueError: If input validation fails or NLTK unavailable
        """
        # Input validation
        if not conversation_inputs:
            raise ValueError("Empty conversation list provided")

        if analysis_level not in ['full', 'agent', 'client']:
            raise ValueError(f"Invalid analysis_level: {analysis_level}. Must be 'full', 'agent', or 'client'")

        if len(conversation_inputs) > 10000:
            logger.warning(f"Large dataset ({len(conversation_inputs)} conversations) - Self-BLEU calculation may be slow")

        # Validate NLTK BLEU scorer availability
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        except ImportError:
            raise ValueError("NLTK BLEU scorer required for Self-BLEU calculation. Install: pip install nltk")
        
        smoothie = SmoothingFunction().method4

        # Extract texts based on analysis level
        texts_to_analyze = []

        for conv_input in conversation_inputs:
            turns = self.extract_finnish_turns(conv_input)
            if not turns:
                continue

            # Full conversation text
            if hasattr(conv_input, 'formatted_string'):
                full_text = conv_input.formatted_string
            else:
                full_text = conv_input

            if analysis_level == 'full':
                texts_to_analyze.append(full_text)
            elif analysis_level == 'agent':
                # Use is_agent field if available, fallback to keyword matching
                agent_texts = []
                for turn in turns:
                    is_agent = turn.get('is_agent', None)
                    if is_agent is True:
                        agent_texts.append(turn['text'])
                    elif is_agent is None:
                        # Fallback: keyword-based detection only if is_agent not available
                        if 'agentti' in turn['speaker'].lower() or 'agent' in turn['speaker'].lower():
                            agent_texts.append(turn['text'])
                if agent_texts:
                    texts_to_analyze.append(' '.join(agent_texts))
            elif analysis_level == 'client':
                # Use is_agent field if available, fallback to keyword matching
                client_texts = []
                for turn in turns:
                    is_agent = turn.get('is_agent', None)
                    if is_agent is False:
                        client_texts.append(turn['text'])
                    elif is_agent is None:
                        # Fallback: keyword-based detection only if is_agent not available
                        if 'asiakas' in turn['speaker'].lower() or 'klient' in turn['speaker'].lower():
                            client_texts.append(turn['text'])
                if client_texts:
                    texts_to_analyze.append(' '.join(client_texts))

        if not texts_to_analyze:
            raise ValueError(f"No texts found for analysis level: {analysis_level}")

        logger.debug(f"Self-BLEU: Collected {len(texts_to_analyze)} texts for {analysis_level} analysis")

        # Track empty texts and pre-tokenize with Finnish processing
        empty_indices = []
        tokenized_texts = []

        for idx, text in enumerate(texts_to_analyze):
            if not text or not text.strip():
                empty_indices.append(idx)
                continue

            # Clean text before processing (remove chat artifacts)
            cleaned_text = self.clean_text(text)
            if not cleaned_text.strip():
                empty_indices.append(idx)
                continue

            # Finnish-specific tokenization with Stanza lemmatization
            tokens = self._tokenize_finnish_text(cleaned_text, min_token_length=2)

            if tokens:
                tokenized_texts.append(tokens)
            else:
                empty_indices.append(idx)

        # Validate empty ratio
        if empty_indices:
            empty_percentage = (len(empty_indices) / len(texts_to_analyze)) * 100
            if empty_percentage > 20:
                raise ValueError(
                    f"Too many empty texts: {len(empty_indices)}/{len(texts_to_analyze)} ({empty_percentage:.1f}%) are empty. "
                    f"First indices: {empty_indices[:10]}"
                    + ("..." if len(empty_indices) > 10 else "")
                )
            logger.warning(
                f"⚠️  {len(empty_indices)} empty/invalid texts skipped ({empty_percentage:.1f}% of {len(texts_to_analyze)})"
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

        # Set n-gram weights (fixed to 4-gram)
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
        Calculate pairwise conversation similarity within a single model.

        What it measures:
        Computes semantic similarity between all unique pairs of conversations from the same model
        using multilingual sentence embeddings.

        Parameters:
        - conversations: List of conversations from the same model
        - model_name: Optional model name for logging

        Returns:
        Dictionary with similarity statistics
        """
        sentence_model = self.semantic_models.get('sentence_transformer')
        if sentence_model is None:
            raise ValueError("Semantic model not initialized")
        
        if not conversations:
            raise ValueError("Empty conversations list")
        
        if len(conversations) < 2:
            raise ValueError(f"Insufficient conversations: {len(conversations)} provided, need ≥2")

        model_label = f" for {model_name}" if model_name else ""
        logger.info(f"🔄 Processing {len(conversations)} conversations{model_label}...")

        # Convert all conversations to text format and track empty ones
        conv_texts = []
        empty_indices = []
        for idx, conv in enumerate(conversations):
            if hasattr(conv, 'formatted_string'):
                text = conv.formatted_string
            else:
                text = str(conv)
            
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

        # Filter and validate
        valid_texts = [(idx, text) for idx, text in enumerate(conv_texts) if text.strip()]
        
        if len(valid_texts) < 2:
            raise ValueError(f"Insufficient valid conversations: {len(valid_texts)}/2 required")
        
        # Deduplicate
        unique_texts = {}
        duplicate_count = 0
        for idx, text in valid_texts:
            if text not in unique_texts:
                unique_texts[text] = idx
            else:
                duplicate_count += 1
        
        if duplicate_count > 0:
            logger.warning(
                f"⚠️  {duplicate_count} duplicate conversations detected. "
                f"Unique: {len(unique_texts)}/{len(valid_texts)}"
            )
        
        # Pre-compute embeddings for unique conversations
        unique_texts_list = list(unique_texts.keys())
        n = len(unique_texts_list)
        
        logger.info(f"🧮 Pre-computing embeddings for {n} unique conversations...")

        try:
            # Generate embeddings for unique conversations
            embeddings = sentence_model.encode(unique_texts_list, convert_to_numpy=True, show_progress_bar=False)

            # Calculate pairwise similarities
            similarities = []
            for i in range(n):
                for j in range(i + 1, n):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(float(sim))

            if not similarities:
                return {
                    'avg_similarity': 0.0,
                    'max_similarity': 0.0,
                    'min_similarity': 0.0,
                    'std_similarity': 0.0,
                    'median_similarity': 0.0,
                    'num_comparisons': 0,
                    'num_duplicates': duplicate_count,
                    'num_empty': len(empty_indices)
                }

            similarities = np.array(similarities)
            
            logger.info(
                f"✅ Analysis complete: avg={np.mean(similarities):.3f}, "
                f"median={np.median(similarities):.3f}, std={np.std(similarities):.3f}"
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

        except Exception as e:
            logger.error(f"❌ Similarity calculation failed: {e}")
            raise

    def batch_analyze_finnish_conversations(
        self,
        conversations: List[Union[str, ConversationData]],
        progress_callback: Optional[callable] = None,
        include_self_bleu: bool = True
    ) -> pd.DataFrame:
        """
        Batch analyze multiple Finnish conversations and return DataFrame (simplified for paper metrics).

        Parameters:
        - conversations: List of conversation texts or ConversationData objects
        - progress_callback: Optional callback for progress updates
        - include_self_bleu: Whether to calculate Self-BLEU scores (can be slow)

        Returns:
        pandas DataFrame with paper metrics (TTR, MATTR, Self-BLEU)
        """
        logger.info(f"🔍 Analyzing {len(conversations)} Finnish conversations...")

        # Calculate Self-BLEU scores
        if include_self_bleu:
            logger.info("📊 Calculating Self-BLEU scores for dataset diversity...")
            try:
                full_bleu = self.finnish_conversation_self_bleu(conversations, analysis_level='full')
                agent_bleu = self.finnish_conversation_self_bleu(conversations, analysis_level='agent')
                client_bleu = self.finnish_conversation_self_bleu(conversations, analysis_level='client')
            except Exception as e:
                logger.warning(f"⚠️  Self-BLEU calculation failed: {e}")
                full_bleu = agent_bleu = client_bleu = {'self_bleu': 0.0}
        else:
            full_bleu = agent_bleu = client_bleu = {'self_bleu': 0.0}

        results = []
        for idx, conv in enumerate(conversations):
            if progress_callback:
                progress_callback(idx, len(conversations))
            
            logger.info(f"Progress: {idx}/{len(conversations)} ({idx/len(conversations)*100:.1f}%)")

            try:
                # Get full text
                if hasattr(conv, 'formatted_string'):
                    full_text = conv.formatted_string
                else:
                    full_text = str(conv)
                
                # Calculate paper metrics only (match Estonian column names)
                flat_result = {
                    'conversation_id': idx,
                    'ttr': self.finnish_type_token_ratio(full_text),
                    'mattr': self.finnish_moving_average_ttr(full_text),
                    'dataset_full_conversation_self_bleu_self_bleu': full_bleu['self_bleu'],
                    'dataset_agent_response_self_bleu_self_bleu': agent_bleu['self_bleu'],
                    'dataset_client_response_self_bleu_self_bleu': client_bleu['self_bleu']
                }
                
                results.append(flat_result)
            
            except Exception as e:
                logger.warning(f"⚠️  Failed to analyze conversation {idx}: {e}")
                continue

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
        # Use parent class's generic implementation with Finnish-specific parameters
        super().create_model_metrics_table(
            model_results=model_results,
            language_name="Finnish",
            language_flag="🇫🇮",
            console=console
        )
