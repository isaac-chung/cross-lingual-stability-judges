"""
Estonian Conversation Diversity Metrics Script

This script implements automatic metrics including COMET alternatives, BARTScore,
and Type-Token Ratio to assess dataset diversity and detect formulaic patterns
across different models. Optimized for Estonian language processing.

Usage:
    python metrics.py -l et
    python metrics.py --language et
    python metrics.py -l et -m gpt-4.1-mini --limit 100
Languages supported: Estonian (et), Hungarian (hu), Finnish (fi)

Requirements:
    pip install numpy pandas nltk transformers sentence-transformers torch matplotlib seaborn estnltk
    # EstNLTK provides proper Estonian language processing
"""

import argparse
import pandas as pd
from typing import Optional
import warnings
import logging
from rich import print as rich_print
import json

# Import base classes and helpers from conversation_metrics module
from conversation_metrics import ConversationData, save_model_results_to_csv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set other loggers to WARNING to reduce noise
logging.getLogger('transformers').setLevel(logging.INFO)
logging.getLogger('sentence_transformers').setLevel(logging.INFO)
logging.getLogger('torch').setLevel(logging.INFO)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

__version__ = "1.0.0"

##---------------------------------------------------- ESTONIAN WORKFLOW ----------------------------------------------------##
def get_example_estonian_conversations(model: Optional[str] = None, json_file_path: Optional[str] = None):

    """Return Estonian conversations as structured ConversationData objects.
    
    Parameters:
    - model: Optional model identifier to filter conversations. If None, returns all conversations.

    - json_file_path: Optional path to the JSON file (default: '../../paper_data/combined_et.json')

    
    Available models:
    1. anthropic.claude-sonnet-4-20250514-v1:0
    2. cohere.command-r-v1:0
    3. gpt-4.1-mini
    4. meta.llama3-70b-instruct-v1:0
    5. meta.llama3-8b-instruct-v1:0
    6. mistral.mixtral-8x7b-instruct-v0:1
    """
    # Set default path to paper_data folder
    if json_file_path is None:
        json_file_path = '../../paper_data/combined_et.json'
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        conversations = []
        # Filter by model if specified
        if model is not None:
            if model not in data:
                logger.warning(f"Model '{model}' not found in data. Available models: {list(data.keys())}")
                return []
            model_dict = {model: data[model]}
        else:
            model_dict = data
        
        for model_name, model_conversations in model_dict.items():
            for conversation_data in model_conversations:
                messages = conversation_data.get('messages', [])

                if not messages:
                    continue

                # Convert messages to the format expected by ConversationData.from_comments
                # Map from_name -> authorName, message -> comment
                comments = []
                for msg in messages:
                    # Extract is_agent from from_type field
                    from_type = msg.get('from_type', None)
                    is_agent = None
                    if from_type == 'agent':
                        is_agent = True
                    elif from_type == 'customer':
                        is_agent = False
                    
                    comments.append({
                        'authorName': msg.get('from_name', 'Unknown'),
                        'comment': msg.get('message', ''),
                        'is_agent': is_agent
                    })

                # Create ConversationData directly from structured comments
                conversation = ConversationData.from_comments(comments)

                if conversation.turns:  # Only add conversations with turns
                    conversations.append(conversation)

        logger.info(f"✅ Loaded {len(conversations)} Estonian conversations from JSON file")
        if len(conversations) == 0:
            raise FileNotFoundError("No conversations found in JSON file")
        return conversations

    except FileNotFoundError:
        logger.info(f"❌ JSON file not found: {json_file_path}")
        logger.info("🔄 Falling back to example conversations...")
        # Fallback to original example conversations as ConversationData objects
        example_conversations_str = [
            """
        Agent: Tere! Kuidas saan teid aidata?
        Klient: Tere! Mul on probleeme oma konto sisselogimisega.
        Agent: Saan teile kindlasti aidata. Millist veateadet te näete?
        Klient: See ütleb, et minu parool on vale, aga ma olen kindel, et see on õige.
        Agent: Saadan teile turvalise lingi uue parooli loomiseks. Kas see sobib?
        Klient: Jah, tänan väga! See oleks suurepärane.
        """,
            """
        Agent: Tere hommikust! Millega saan aidata?
        Klient: Tahan oma arveldusandmeid uuendada.
        Agent: Loomulikult! Saan teid aidata arveldusandmete uuendamisega. Turvalisuse huvides, kas saate kinnitada oma konto e-maili?
        Klient: Muidugi, see on klient@näidis.ee.
        Agent: Suurepärane! Leidsin teie konto. Milliseid muudatusi soovite arveldusandmetesse teha?
        Klient: Soovin muuta oma aadressi ja telefoninumbrit.
        """,
            """
        Agent: Tere päevast! Kuidas saan teid teenindada?
        Klient: Minu hiljutine tellimus pole veel kohale jõudnud.
        Agent: Kahju kuulda viivitusest. Laske mul teie tellimust jälitada. Kas saate anda tellimusnumbri?
        Klient: See on TELL-12345.
        Agent: Aitäh! Näen, et teie tellimus saadeti eile välja ja peaks jõudma 2-3 tööpäeva jooksul.
        Klient: Tänan teavitamast! Olen rahul.
        """
        ]
        # Convert string conversations to ConversationData objects
        return [ConversationData.from_string(conv_str) for conv_str in example_conversations_str]
    except Exception as e:
        logger.info(f"❌ Error loading JSON file: {e}")
        logger.info("🔄 Falling back to example conversations...")
        # Fallback to original example conversations as ConversationData objects
        example_conversations_str = [
            """
        Agent: Tere! Kuidas saan teid aidata?
        Klient: Tere! Mul on probleeme oma konto sisselogimisega.
        Agent: Saan teile kindlasti aidata. Millist veateadet te näete?
        Klient: See ütleb, et minu parool on vale, aga ma olen kindel, et see on õige.
        Agent: Saadan teile turvalise lingi uue parooli loomiseks. Kas see sobib?
        Klient: Jah, tänan väga! See oleks suurepärane.
        """,
            """
        Agent: Tere hommikust! Millega saan aidata?
        Klient: Tahan oma arveldusandmeid uuendada.
        Agent: Loomulikult! Saan teid aidata arveldusandmete uuendamisega. Turvalisuse huvides, kas saate kinnitada oma konto e-maili?
        Klient: Muidugi, see on klient@näidis.ee.
        Agent: Suurepärane! Leidsin teie konto. Milliseid muudatusi soovite arveldusandmetesse teha?
        Klient: Soovin muuta oma aadressi ja telefoninumbrit.
        """,
            """
        Agent: Tere päevast! Kuidas saan teid teenindada?
        Klient: Minu hiljutine tellimus pole veel kohale jõudnud.
        Agent: Kahju kuulda viivitusest. Laske mul teie tellimust jälitada. Kas saate anda tellimusnumbri?
        Klient: See on TELL-12345.
        Agent: Aitäh! Näen, et teie tellimus saadeti eile välja ja peaks jõudma 2-3 tööpäeva jooksul.
        Klient: Tänan teavitamast! Olen rahul.
        """
        ]
        # Convert string conversations to ConversationData objects
        return [ConversationData.from_string(conv_str) for conv_str in example_conversations_str]

def et_metrics_workflow(model: Optional[str] = None, limit: Optional[int] = None, json_file_path: Optional[str] = None):

    """Estonian metrics workflow using EstNLTK.
    
    Args:
        model: Model identifier or 'all' to analyze all models
        limit: Optional limit on number of conversations to analyze per model
        json_file_path: Optional path to the JSON file

    """
    # Import here to avoid circular import
    from et_metrics import EstonianConversationMetrics
    
    # Initialize the metrics analyzer
    try:
        analyzer = EstonianConversationMetrics()
    except Exception as e:
        rich_print(f"[red]❌ Failed to initialize analyzer: {e}[/red]")
        rich_print("[yellow]💡 Try installing missing dependencies:[/yellow]")
        rich_print("pip install numpy pandas nltk estnltk transformers sentence-transformers torch matplotlib seaborn rich")
        return

    # Get example conversations grouped by model
    logger.info("📝 Loading example Estonian conversations...")
    
    if model == 'all':
        # Load all models separately
        all_models = [
            'anthropic.claude-sonnet-4-20250514-v1:0',
            'cohere.command-r-v1:0',
            'gpt-4.1-mini',
            'meta.llama3-70b-instruct-v1:0',
            'meta.llama3-8b-instruct-v1:0',
            'mistral.mixtral-8x7b-instruct-v0:1'
        ]
        model_conversations = {}
        total_conversations = 0
        for model_name in all_models:
            convos = get_example_estonian_conversations(model=model_name, json_file_path=json_file_path)

            if convos:
                if limit:
                    convos = convos[:limit]
                    logger.info(f"🔍 Limited to {len(convos)} conversations for {model_name}")
                model_conversations[model_name] = convos
                total_conversations += len(convos)
        rich_print(f"[green]✅ Loaded {total_conversations} conversations across {len(model_conversations)} models[/green]")
    else:
        # Load single model
        conversations = get_example_estonian_conversations(model=model, json_file_path=json_file_path)

        if limit:
            conversations = conversations[:limit]
            logger.info(f"🔍 Limited to {len(conversations)} conversations for {model}")
        model_conversations = {
            model: conversations
        }
        rich_print(f"[green]✅ Loaded {len(conversations)} conversations for {model}[/green]")

    # Analyze conversations by model using the existing batch analysis method
    model_results = {}
    for model_name, convos in model_conversations.items():
        if convos:
            # Filter out conversations with fewer than 2 turns (needed for coherence calculation)
            valid_convos = [conv for conv in convos if len(conv.turns) >= 2]
            if len(valid_convos) < len(convos):
                logger.warning(f"⚠️  Filtered out {len(convos) - len(valid_convos)} single-turn conversations from {model_name}")
            
            if valid_convos:
                logger.info(f"Analyzing {model_name}...")
                df_results = analyzer.batch_analyze_estonian_conversations(valid_convos)
                
                # Calculate intra-model similarity for this model
                if len(valid_convos) >= 2:
                    logger.info(f"📊 Calculating intra-model similarity for {model_name}...")
                    try:
                        similarity_stats = analyzer.calculate_intra_model_conversation_similarity(valid_convos, model_name)
                        # Add similarity metrics to the dataframe
                        for key, value in similarity_stats.items():
                            df_results[key] = value
                    except Exception as e:
                        logger.warning(f"⚠️  Intra-model similarity calculation failed for {model_name}: {e}")
                
                model_results[model_name] = df_results
            else:
                logger.warning(f"⚠️  No valid conversations to analyze for {model_name}")

    # Display rich table with metrics
    if model_results:
        analyzer.create_model_metrics_table(model_results)
        # Save results to CSV
        save_model_results_to_csv(model_results, 'et')
    else:
        rich_print("[red]❌ No model results to display[/red]")

    # Display processing info
    rich_print(f"\n[cyan]Processing method:[/cyan] {'EstNLTK' if analyzer.estnltk_available else 'Basic fallback'}")
    rich_print(f"[cyan]Syntax analysis:[/cyan] {'Available' if analyzer.estnltk_syntax else 'Not available'}")
    rich_print(f"[cyan]Estonian stopwords:[/cyan] {len(analyzer.estonian_stopwords)}")

    if not analyzer.estnltk_available:
        rich_print("\n[yellow]⚠️  For best results, install EstNLTK: pip install estnltk[/yellow]")

    rich_print("\n[bold green]🎉 Analysis complete! Check the generated files for detailed results.[/bold green]")

##---------------------------------------------------- HUNGARIAN WORKFLOW ----------------------------------------------------##
def get_example_hungarian_conversations(model: Optional[str] = None, json_file_path: Optional[str] = None):
    """Return Hungarian conversations as structured ConversationData objects.
    
    Parameters:
    - model: Optional model identifier to filter conversations. If None, returns all conversations.
    - json_file_path: Optional path to the JSON file (default: '../../paper_data/combined_hu.json')
    
    Available models:
    1. anthropic.claude-sonnet-4-20250514-v1:0
    2. cohere.command-r-v1:0
    3. gpt-4.1-mini
    4. meta.llama3-70b-instruct-v1:0
    5. meta.llama3-8b-instruct-v1:0
    6. mistral.mixtral-8x7b-instruct-v0:1
    """

    # Set default path to paper_data folder
    if json_file_path is None:
        json_file_path = '../../paper_data/combined_hu.json'
    
    # Define Hungarian speaker pattern
    hungarian_speaker_pattern = r'(Ügynök|Ügyfél|Ügyfélszolgálat|Munkatárs|Agent|Kliens):'

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        conversations = []

        # New structure: model names as top-level keys, each containing array of conversations
        # Filter by model if specified
        if model is not None:
            if model not in data:
                logger.warning(f"Model '{model}' not found in data. Available models: {list(data.keys())}")
                return []
            model_dict = {model: data[model]}
        else:
            model_dict = data
        
        for model_name, model_conversations in model_dict.items():
            for conversation_data in model_conversations:
                messages = conversation_data.get('messages', [])

                if not messages:
                    continue

                # Convert messages to the format expected by ConversationData.from_comments
                # Map from_name -> authorName, message -> comment
                comments = []
                for msg in messages:
                    # Extract is_agent from from_type field
                    from_type = msg.get('from_type', None)
                    is_agent = None
                    if from_type == 'agent':
                        is_agent = True
                    elif from_type == 'customer':
                        is_agent = False
                    
                    comments.append({
                        'authorName': msg.get('from_name', 'Unknown'),
                        'comment': msg.get('message', ''),
                        'is_agent': is_agent
                    })

                # Create ConversationData directly from structured comments
                conversation = ConversationData.from_comments(comments)

                if conversation.turns:  # Only add conversations with turns
                    conversations.append(conversation)

        logger.info(f"✅ Loaded {len(conversations)} Hungarian conversations from JSON file")
        if len(conversations) == 0:
            raise FileNotFoundError("No conversations found in JSON file")
        else:
            return conversations

    except FileNotFoundError:
        logger.info(f"❌ JSON file not found: {json_file_path}")
        logger.info("🔄 Falling back to example conversations...")
        # Fallback to original example conversations as ConversationData objects
        example_conversations_str =  [
        """
    Ügynök: Üdvözlöm! Hogyan segíthetek?
    Ügyfél: Szia! Problémám van a belépéssel.
    Ügynök: Természetesen segítek. Milyen hibaüzenetet lát?
    Ügyfél: Azt írja, hogy rossz a jelszavam, de biztos vagyok benne, hogy helyes.
    Ügynök: Küldök egy biztonságos linket új jelszó létrehozásához. Jó lesz így?
    Ügyfél: Igen, köszönöm szépen! Az nagyszerű lenne.
    """,
        """
    Ügynök: Jó reggelt! Miben segíthetek?
    Ügyfél: Szeretném frissíteni a számlázási adataimat.
    Ügynök: Természetesen! Segítek a számlázási adatok frissítésében. Biztonsági okokból megerősítené a fiók e-mail címét?
    Ügyfél: Persze, az ügyfél@példa.hu.
    Ügynök: Nagyszerű! Megtaláltam a fiókját. Milyen változtatásokat szeretne eszközölni?
    Ügyfél: Szeretném megváltoztatni a címemet és telefonszámomat.
    """,
        """
    Ügynök: Jó napot! Hogyan szolgálhatom ki?
    Ügyfél: A legutóbbi rendelésem még nem érkezett meg.
    Ügynök: Sajnálom a késedelmet. Hadd kövessem nyomon a rendelését. Meg tudná adni a rendelési számot?
    Ügyfél: Ez a REND-12345.
    Ügynök: Köszönöm! Látom, hogy a rendelését tegnap küldték el, és 2-3 munkanapon belül meg kell érkeznie.
    Ügyfél: Köszönöm az információt! Elégedett vagyok.
    """
    ]
        # Convert string conversations to ConversationData objects with Hungarian speaker pattern
        return [ConversationData.from_string(conv_str, speaker_pattern=hungarian_speaker_pattern) 
                for conv_str in example_conversations_str]
    except Exception as e:
        logger.info(f"❌ Error loading JSON file: {e}")
        logger.info("🔄 Falling back to example conversations...")
        # Fallback to original example conversations as ConversationData objects
        example_conversations_str =  [
        """
    Ügynök: Üdvözlöm! Hogyan segíthetek?
    Ügyfél: Szia! Problémám van a belépéssel.
    Ügynök: Természetesen segítek. Milyen hibaüzenetet lát?
    Ügyfél: Azt írja, hogy rossz a jelszavam, de biztos vagyok benne, hogy helyes.
    Ügynök: Küldök egy biztonságos linket új jelszó létrehozásához. Jó lesz így?
    Ügyfél: Igen, köszönöm szépen! Az nagyszerű lenne.
    """,
        """
    Ügynök: Jó reggelt! Miben segíthetek?
    Ügyfél: Szeretném frissíteni a számlázási adataimat.
    Ügynök: Természetesen! Segítek a számlázási adatok frissítésében. Biztonsági okokból megerősítené a fiók e-mail címét?
    Ügyfél: Persze, az ügyfél@példa.hu.
    Ügynök: Nagyszerű! Megtaláltam a fiókját. Milyen változtatásokat szeretne eszközölni?
    Ügyfél: Szeretném megváltoztatni a címemet és telefonszámomat.
    """,
        """
    Ügynök: Jó napot! Hogyan szolgálhatom ki?
    Ügyfél: A legutóbbi rendelésem még nem érkezett meg.
    Ügynök: Sajnálom a késedelmet. Hadd kövessem nyomon a rendelését. Meg tudná adni a rendelési számot?
    Ügyfél: Ez a REND-12345.
    Ügynök: Köszönöm! Látom, hogy a rendelését tegnap küldték el, és 2-3 munkanapon belül meg kell érkeznie.
    Ügyfél: Köszönöm az információt! Elégedett vagyok.
    """
    ]
        # Convert string conversations to ConversationData objects with Hungarian speaker pattern
        return [ConversationData.from_string(conv_str, speaker_pattern=hungarian_speaker_pattern) 
                for conv_str in example_conversations_str]


def hu_metrics_workflow(model: Optional[str] = None, limit: Optional[int] = None, json_file_path: Optional[str] = None):
    """
    Hungarian metrics workflow using HuSpaCy.
    Args:
        model: Model identifier or 'all' to analyze all models
        limit: Optional limit on number of conversations to analyze per model
        json_file_path: Optional path to the JSON file
    """
    # Import here to avoid circular import
    from hu_metrics import HungarianConversationMetrics
    
    # Initialize the metrics analyzer
    try:
        analyzer = HungarianConversationMetrics()
    except Exception as e:
        rich_print(f"[red]❌ Failed to initialize analyzer: {e}[/red]")
        rich_print("[yellow]💡 Try installing missing dependencies:[/yellow]")
        rich_print("pip install numpy pandas nltk huspacy transformers sentence-transformers torch rich")
        rich_print("python -m spacy download hu_core_news_lg")
        return

    # Load Hungarian conversations
    logger.info("📝 Loading example Hungarian conversations...")
    
    if model == 'all':
        # Load all models separately
        all_models = [
            'anthropic.claude-sonnet-4-20250514-v1:0',
            'cohere.command-r-v1:0',
            'gpt-4.1-mini',
            'meta.llama3-70b-instruct-v1:0',
            'meta.llama3-8b-instruct-v1:0',
            'mistral.mixtral-8x7b-instruct-v0:1'
        ]
        model_conversations = {}
        total_conversations = 0
        for model_name in all_models:
            convos = get_example_hungarian_conversations(model=model_name, json_file_path=json_file_path)
            if convos:
                if limit:
                    convos = convos[:limit]
                    logger.info(f"🔍 Limited to {len(convos)} conversations for {model_name}")
                model_conversations[model_name] = convos
                total_conversations += len(convos)
        rich_print(f"[green]✅ Loaded {total_conversations} conversations across {len(model_conversations)} models[/green]")
    else:
        # Load single model
        conversations = get_example_hungarian_conversations(model=model, json_file_path=json_file_path)
        if limit:
            conversations = conversations[:limit]
            logger.info(f"🔍 Limited to {len(conversations)} conversations for {model}")
        model_conversations = {
            model: conversations
        }
        rich_print(f"[green]✅ Loaded {len(conversations)} conversations for {model}[/green]")

    # Analyze conversations by model
    model_results = {}
    for model_name, convos in model_conversations.items():
        if convos:
            # Filter out conversations with fewer than 2 turns (needed for coherence calculation)
            valid_convos = [conv for conv in convos if len(conv.turns) >= 2]
            if len(valid_convos) < len(convos):
                logger.warning(f"⚠️  Filtered out {len(convos) - len(valid_convos)} single-turn conversations from {model_name}")
            
            if valid_convos:
                logger.info(f"Analyzing {model_name}...")
                df_results = analyzer.batch_analyze_hungarian_conversations(valid_convos)
                
                # Calculate intra-model similarity for this model
                if len(valid_convos) >= 2:
                    logger.info(f"📊 Calculating intra-model similarity for {model_name}...")
                    try:
                        similarity_stats = analyzer.calculate_intra_model_conversation_similarity(valid_convos, model_name)
                        # Add similarity metrics to the dataframe
                        for key, value in similarity_stats.items():
                            df_results[key] = value
                    except Exception as e:
                        logger.warning(f"⚠️  Intra-model similarity calculation failed for {model_name}: {e}")
                
                model_results[model_name] = df_results
            else:
                logger.warning(f"⚠️  No valid conversations to analyze for {model_name}")

    # Display rich table with metrics
    if model_results:
        analyzer.create_model_metrics_table(model_results)
        # Save results to CSV
        save_model_results_to_csv(model_results, 'hu')
    else:
        rich_print("[red]❌ No model results to display[/red]")

    

    # Display processing info
    rich_print(f"\n[cyan]Processing method:[/cyan] {'Stanza' if analyzer.stanza_available else 'Basic fallback'}")
    rich_print(f"[cyan]Hungarian stopwords:[/cyan] {len(analyzer.hungarian_stopwords)}")

    if not analyzer.stanza_available:
        rich_print("\n[yellow]⚠️  For best results, install Stanza: pip install stanza[/yellow]")
        rich_print("[yellow]   Then download model: python -c \"import stanza; stanza.download('hu')\"[/yellow]")

    rich_print("\n[bold green]🎉 Analysis complete! Check the generated files for detailed results.[/bold green]")

##---------------------------------------------------- FINNISH WORKFLOW ----------------------------------------------------##
def get_example_finnish_conversations(model: Optional[str] = None, json_file_path: Optional[str] = None):
    """Return Finnish conversations as structured ConversationData objects.
    
    Parameters:
    - model: Optional model identifier to filter conversations. If None, returns all conversations.
    - json_file_path: Optional path to the JSON file (default: '../../paper_data/combined_fi.json')
    
    Available models:
    1. anthropic.claude-sonnet-4-20250514-v1:0
    2. cohere.command-r-v1:0
    3. gpt-4.1-mini
    4. meta.llama3-70b-instruct-v1:0
    5. meta.llama3-8b-instruct-v1:0
    6. mistral.mixtral-8x7b-instruct-v0:1
    """

    # Set default path to paper_data folder
    if json_file_path is None:
        json_file_path = '../../paper_data/combined_fi.json'
    
    # Define Finnish speaker pattern
    finnish_speaker_pattern = r'(Agentti|Asikas|Asiakas|Palvelunedustaja|Agent|Klient):'

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        conversations = []

        # New structure: model names as top-level keys, each containing array of conversations
        # Filter by model if specified
        if model is not None:
            if model not in data:
                logger.warning(f"Model '{model}' not found in data. Available models: {list(data.keys())}")
                return []
            model_dict = {model: data[model]}
        else:
            model_dict = data
        
        for model_name, model_conversations in model_dict.items():
            for conversation_data in model_conversations:
                messages = conversation_data.get('messages', [])

                if not messages:
                    continue

                # Convert messages to the format expected by ConversationData.from_comments
                # Map from_name -> authorName, message -> comment
                comments = []
                for msg in messages:
                    # Extract is_agent from from_type field
                    from_type = msg.get('from_type', None)
                    is_agent = None
                    if from_type == 'agent':
                        is_agent = True
                    elif from_type == 'customer':
                        is_agent = False
                    
                    comments.append({
                        'authorName': msg.get('from_name', 'Unknown'),
                        'comment': msg.get('message', ''),
                        'is_agent': is_agent
                    })

                # Create ConversationData directly from structured comments
                conversation = ConversationData.from_comments(comments)

                if conversation.turns:  # Only add conversations with turns
                    conversations.append(conversation)

        logger.info(f"✅ Loaded {len(conversations)} Finnish conversations from JSON file")
        if len(conversations) == 0:
            raise FileNotFoundError("No conversations found in JSON file")
        else:
            return conversations

    except FileNotFoundError:
        logger.info(f"❌ JSON file not found: {json_file_path}")
        logger.info("🔄 Falling back to example conversations...")
        # Fallback to original example conversations as ConversationData objects
        example_conversations_str =  [
        """
    Agentti: Hei! Kuinka voin auttaa sinua?
    Asiakas: Hei! Minulla on ongelmia kirjautumisessa.
    Agentti: Voin varmasti auttaa. Mikä virheilmoitus näkyy?
    Asiakas: Se sanoo, että salasanani on väärä, mutta olen varma että se on oikein.
    Agentti: Lähetän sinulle turvallisen linkin uuden salasanan luomiseen. Sopiiko se?
    Asiakas: Kyllä, kiitos paljon! Se olisi mahtavaa.
    """,
        """
    Agentti: Hyvää huomenta! Millä voin palvella?
    Asiakas: Haluaisin päivittää laskutustietojani.
    Agentti: Tietysti! Voin auttaa laskutustietojen päivittämisessä. Turvallisuussyistä, voitko vahvistaa tilisi sähköpostiosoitteen?
    Asiakas: Totta kai, se on asiakas@esimerkki.fi.
    Agentti: Loistavaa! Löysin tilisi. Mitä muutoksia haluaisit tehdä?
    Asiakas: Haluaisin vaihtaa osoitteeni ja puhelinnumeroni.
    """,
        """
    Agentti: Hyvää päivää! Kuinka voin palvella sinua?
    Asiakas: Viimeisin tilaukseni ei ole vielä saapunut.
    Agentti: Pahoittelen viivästystä. Anna minun seurata tilauksiasi. Voitko antaa tilausnumeron?
    Asiakas: Se on TILAUS-12345.
    Agentti: Kiitos! Näen, että tilauksesi lähetettiin eilen ja sen pitäisi saapua 2-3 arkipäivässä.
    Asiakas: Kiitos tiedosta! Olen tyytyväinen.
    """
    ]
        # Convert string conversations to ConversationData objects with Finnish speaker pattern
        return [ConversationData.from_string(conv_str, speaker_pattern=finnish_speaker_pattern) 
                for conv_str in example_conversations_str]
    except Exception as e:
        logger.info(f"❌ Error loading JSON file: {e}")
        logger.info("🔄 Falling back to example conversations...")
        # Fallback to original example conversations as ConversationData objects
        example_conversations_str =  [
        """
    Agentti: Hei! Kuinka voin auttaa sinua?
    Asiakas: Hei! Minulla on ongelmia kirjautumisessa.
    Agentti: Voin varmasti auttaa. Mikä virheilmoitus näkyy?
    Asiakas: Se sanoo, että salasanani on väärä, mutta olen varma että se on oikein.
    Agentti: Lähetän sinulle turvallisen linkin uuden salasanan luomiseen. Sopiiko se?
    Asiakas: Kyllä, kiitos paljon! Se olisi mahtavaa.
    """,
        """
    Agentti: Hyvää huomenta! Millä voin palvella?
    Asiakas: Haluaisin päivittää laskutustietojani.
    Agentti: Tietysti! Voin auttaa laskutustietojen päivittämisessä. Turvallisuussyistä, voitko vahvistaa tilisi sähköpostiosoitteen?
    Asiakas: Totta kai, se on asiakas@esimerkki.fi.
    Agentti: Loistavaa! Löysin tilisi. Mitä muutoksia haluaisit tehdä?
    Asiakas: Haluaisin vaihtaa osoitteeni ja puhelinnumeroni.
    """,
        """
    Agentti: Hyvää päivää! Kuinka voin palvella sinua?
    Asiakas: Viimeisin tilaukseni ei ole vielä saapunut.
    Agentti: Pahoittelen viivästystä. Anna minun seurata tilauksiasi. Voitko antaa tilausnumeron?
    Asiakas: Se on TILAUS-12345.
    Agentti: Kiitos! Näen, että tilauksesi lähetettiin eilen ja sen pitäisi saapua 2-3 arkipäivässä.
    Asiakas: Kiitos tiedosta! Olen tyytyväinen.
    """
    ]
        # Convert string conversations to ConversationData objects with Finnish speaker pattern
        return [ConversationData.from_string(conv_str, speaker_pattern=finnish_speaker_pattern) 
                for conv_str in example_conversations_str]
        
def fi_metrics_workflow(model: Optional[str] = None, limit: Optional[int] = None, json_file_path: Optional[str] = None):
    """Finnish metrics workflow using Stanza.
    
    Args:
        model: Model identifier or 'all' to analyze all models
        limit: Optional limit on number of conversations to analyze per model
        json_file_path: Optional path to the JSON file
    """
    # Import here to avoid circular import
    from fi_metrics import FinnishConversationMetrics
    
    # Initialize the metrics analyzer
    try:
        analyzer = FinnishConversationMetrics()
    except Exception as e:
        rich_print(f"[red]❌ Failed to initialize analyzer: {e}[/red]")
        rich_print("[yellow]💡 Try installing missing dependencies:[/yellow]")
        rich_print("pip install numpy pandas nltk stanza transformers sentence-transformers torch rich")
        rich_print("python -c \"import stanza; stanza.download('fi')\"")
        return

    # Load Finnish conversations
    logger.info("📝 Loading example Finnish conversations...")
    
    if model == 'all':
        # Load all models separately
        all_models = [
            'anthropic.claude-sonnet-4-20250514-v1:0',
            'cohere.command-r-v1:0',
            'gpt-4.1-mini',
            'meta.llama3-70b-instruct-v1:0',
            'meta.llama3-8b-instruct-v1:0',
            'mistral.mixtral-8x7b-instruct-v0:1'
        ]
        model_conversations = {}
        total_conversations = 0
        for model_name in all_models:
            convos = get_example_finnish_conversations(model=model_name, json_file_path=json_file_path)
            if convos:
                if limit:
                    convos = convos[:limit]
                    logger.info(f"🔍 Limited to {len(convos)} conversations for {model_name}")
                model_conversations[model_name] = convos
                total_conversations += len(convos)
        rich_print(f"[green]✅ Loaded {total_conversations} conversations across {len(model_conversations)} models[/green]")
    else:
        # Load single model
        conversations = get_example_finnish_conversations(model=model, json_file_path=json_file_path)
        if limit:
            conversations = conversations[:limit]
            logger.info(f"🔍 Limited to {len(conversations)} conversations")
        model_conversations = {
            model: conversations
        }
        rich_print(f"[green]✅ Loaded {len(conversations)} conversations for {model}[/green]")

    # Analyze conversations by model
    model_results = {}
    for model_name, convos in model_conversations.items():
        if convos:
            # Filter out conversations with fewer than 2 turns (needed for coherence calculation)
            valid_convos = [conv for conv in convos if len(conv.turns) >= 2]
            if len(valid_convos) < len(convos):
                logger.warning(f"⚠️  Filtered out {len(convos) - len(valid_convos)} single-turn conversations from {model_name}")
            
            if valid_convos:
                logger.info(f"Analyzing {model_name}...")
                df_results = analyzer.batch_analyze_finnish_conversations(valid_convos)
                
                # Calculate intra-model similarity for this model
                if len(valid_convos) >= 2:
                    logger.info(f"📊 Calculating intra-model similarity for {model_name}...")
                    try:
                        similarity_stats = analyzer.calculate_intra_model_conversation_similarity(valid_convos, model_name)
                        # Add similarity metrics to the dataframe
                        for key, value in similarity_stats.items():
                            df_results[key] = value
                    except Exception as e:
                        logger.warning(f"⚠️  Intra-model similarity calculation failed for {model_name}: {e}")
                
                model_results[model_name] = df_results
            else:
                logger.warning(f"⚠️  No valid conversations to analyze for {model_name}")

    # Display rich table with metrics
    if model_results:
        analyzer.create_model_metrics_table(model_results)
        # Save results to CSV
        save_model_results_to_csv(model_results, 'fi')
    else:
        rich_print("[red]❌ No model results to display[/red]")

    # Display processing info
    rich_print(f"\n[cyan]Processing method:[/cyan] {'Stanza' if analyzer.stanza_available else 'Basic fallback'}")
    rich_print(f"[cyan]Finnish stopwords:[/cyan] {len(analyzer.finnish_stopwords)}")

    if not analyzer.stanza_available:
        rich_print("\n[yellow]⚠️  For best results, install Stanza: pip install stanza[/yellow]")
        rich_print("[yellow]   Then download model: python -c \"import stanza; stanza.download('fi')\"[/yellow]")

    rich_print("\n[bold green]🎉 Analysis complete! Check the generated files for detailed results.[/bold green]")

##---------------------------------------------------- MAIN FUNCTION ----------------------------------------------------##
def main():
    """š
    Main function to run the conversation analysis for the paper 
    Cross-Lingual Stability of LLM Judges Under Controlled Generation:
    Evidence from Finno-Ugric Languages
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generated Finno-Ugric Conversation Diversity Metrics Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-l', '--language',
        type=str,
        default='et',
        choices=['et', 'hu', 'fi'], #validates if the given arg is among the supported languages, otherwise throws an error
        help='Language code for analysis (default: et for Estonian, supported are hu for Hungarian and fi for Finnish)'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='all',
        choices=[
            'all',
            'anthropic.claude-sonnet-4-20250514-v1:0',
            'cohere.command-r-v1:0',
            'gpt-4.1-mini',
            'meta.llama3-70b-instruct-v1:0',
            'meta.llama3-8b-instruct-v1:0',
            'mistral.mixtral-8x7b-instruct-v0:1'
        ],
        help='Model identifier to filter conversations (default: all). Use "all" to analyze all models separately, or specify a single model.'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit the number of conversations to analyze (useful for testing on smaller datasets)'
    )
    parser.add_argument(
        '--json-file',
        type=str,
        default=None,
        help='Path to the JSON file containing conversations. Defaults: Estonian (../../paper_data/combined_et.json), Hungarian (../../paper_data/combined_hu.json), Finnish (../../paper_data/combined_fi.json)'
    )
    args = parser.parse_args()
    
    if args.language == 'et':
        logger.info("Starting Estonian Conversation Diversity Analysis")
        logger.info("=" * 50)
        if args.limit:
            logger.info(f"🔍 Testing mode: Limited to {args.limit} conversations per model")
        et_metrics_workflow(model=args.model, limit=args.limit, json_file_path=args.json_file)
    elif args.language == 'hu':
        logger.info("Starting Hungarian Conversation Diversity Analysis")
        logger.info("=" * 50)
        if args.limit:
            logger.info(f"🔍 Testing mode: Limited to {args.limit} conversations per model")
        hu_metrics_workflow(model=args.model, limit=args.limit, json_file_path=args.json_file)
    elif args.language == 'fi':
        logger.info("Starting Finnish Conversation Diversity Analysis")
        logger.info("=" * 50)
        if args.limit:
            logger.info(f"🔍 Testing mode: Limited to {args.limit} conversations per model")
        fi_metrics_workflow(model=args.model, limit=args.limit, json_file_path=args.json_file)
    else: #For sanity, should not reach here due to argparse choices
        logger.error(f"Unsupported language code: {args.language}")
    

if __name__ == "__main__":
    main()