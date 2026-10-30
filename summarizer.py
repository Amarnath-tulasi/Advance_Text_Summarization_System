import re
import math
import torch

from collections import Counter

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("=" * 70)
print("ADVANCED TEXT SUMMARIZATION SYSTEM")
print("=" * 70)

print(f"Device : {DEVICE}")
print(f"Model  : {MODEL_NAME}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# LOAD TRANSFORMER
# ============================================================

print("Loading Transformer model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)

model.eval()

print("Transformer model loaded successfully.")

print("=" * 70)


# ============================================================
# BASIC TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    # Remove control characters
    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F]",
        " ",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# WORD COUNT
# ============================================================

def count_words(text):

    if not text:
        return 0

    return len(text.split())


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):

    text = clean_text(text)

    if not text:
        return []

    # Basic but reliable sentence segmentation.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


# ============================================================
# NORMALIZE WORDS
# ============================================================

def normalize_word(word):

    return re.sub(
        r"[^a-zA-Z0-9]",
        "",
        word.lower()
    )


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "the", "a", "an", "and", "or",
    "but", "if", "then", "than",
    "is", "are", "was", "were",
    "be", "been", "being",
    "to", "of", "in", "on",
    "for", "with", "by", "from",
    "at", "as", "it", "this",
    "that", "these", "those",
    "they", "their", "there",
    "he", "she", "his", "her",
    "we", "our", "you", "your",
    "i", "me", "my",
    "which", "who", "whom",
    "what", "when", "where",
    "why", "how",
    "can", "could", "would",
    "should", "may", "might",
    "will", "shall",
    "do", "does", "did",
    "have", "has", "had",
    "not", "no",
    "very", "also", "more",
    "most", "some", "such"
}


# ============================================================
# CREATE IMPORTANT WORD FREQUENCY
# ============================================================

def calculate_word_frequency(text):

    words = [
        normalize_word(word)
        for word in text.split()
    ]

    words = [
        word
        for word in words
        if word
        and word not in STOP_WORDS
        and len(word) > 2
    ]

    frequency = Counter(words)

    if not frequency:
        return {}

    maximum = max(
        frequency.values()
    )

    normalized_frequency = {}

    for word, value in frequency.items():

        normalized_frequency[word] = (
            value / maximum
        )

    return normalized_frequency


# ============================================================
# SENTENCE IMPORTANCE SCORING
# ============================================================

def score_sentences(
    text
):

    sentences = split_sentences(
        text
    )

    if not sentences:
        return []


    word_frequency = (
        calculate_word_frequency(text)
    )


    scored_sentences = []

    total_sentences = len(
        sentences
    )


    for index, sentence in enumerate(
        sentences
    ):

        words = [
            normalize_word(word)
            for word in sentence.split()
        ]

        meaningful_words = [
            word
            for word in words
            if word
            and word not in STOP_WORDS
        ]


        # ----------------------------------------------------
        # Frequency score
        # ----------------------------------------------------

        if meaningful_words:

            frequency_score = sum(

                word_frequency.get(
                    word,
                    0
                )

                for word in meaningful_words

            ) / len(
                meaningful_words
            )

        else:

            frequency_score = 0


        # ----------------------------------------------------
        # Position score
        # ----------------------------------------------------

        if total_sentences == 1:

            position_score = 1.0

        else:

            position = (
                index /
                (total_sentences - 1)
            )

            # Beginning and ending sentences
            # often contain important information.

            position_score = max(

                1.0 - position,

                position

            )


        # ----------------------------------------------------
        # Length score
        # ----------------------------------------------------

        sentence_word_count = (
            len(sentence.split())
        )


        if 8 <= sentence_word_count <= 45:

            length_score = 1.0

        elif sentence_word_count < 8:

            length_score = 0.5

        else:

            length_score = 0.7


        # ----------------------------------------------------
        # Number / fact score
        # ----------------------------------------------------

        number_count = len(
            re.findall(
                r"\b\d+(?:\.\d+)?\b",
                sentence
            )
        )


        fact_score = min(
            1.0,
            number_count * 0.15
        )


        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        final_score = (

            0.55 * frequency_score

            + 0.20 * position_score

            + 0.15 * length_score

            + 0.10 * fact_score

        )


        scored_sentences.append({

            "index": index,

            "sentence": sentence,

            "score": final_score,

            "words": sentence_word_count

        })


    return scored_sentences


# ============================================================
# TOKEN CHUNKING
# ============================================================

def split_text_into_chunks(
    text,
    max_tokens=850
):

    text = clean_text(text)

    if not text:
        return []


    sentences = split_sentences(
        text
    )

    if not sentences:

        return [text]


    chunks = []

    current_chunk = []

    current_tokens = 0


    for sentence in sentences:

        sentence_tokens = len(
            tokenizer.encode(
                sentence,
                add_special_tokens=False
            )
        )


        # If one sentence is larger than
        # the chunk limit, split it directly.

        if sentence_tokens > max_tokens:

            if current_chunk:

                chunks.append(
                    " ".join(current_chunk)
                )

                current_chunk = []

                current_tokens = 0


            token_ids = tokenizer.encode(
                sentence,
                add_special_tokens=False
            )


            for start in range(
                0,
                len(token_ids),
                max_tokens
            ):

                part = token_ids[
                    start:start + max_tokens
                ]

                part_text = tokenizer.decode(
                    part,
                    skip_special_tokens=True
                )

                if part_text.strip():

                    chunks.append(
                        part_text.strip()
                    )

            continue


        # Add sentence to current chunk

        if (
            current_tokens +
            sentence_tokens
            <= max_tokens
        ):

            current_chunk.append(
                sentence
            )

            current_tokens += (
                sentence_tokens
            )

        else:

            if current_chunk:

                chunks.append(
                    " ".join(current_chunk)
                )


            current_chunk = [
                sentence
            ]

            current_tokens = (
                sentence_tokens
            )


    # Add final chunk

    if current_chunk:

        chunks.append(
            " ".join(current_chunk)
        )


    return chunks


# ============================================================
# TRANSFORMER SUMMARY FOR ONE CHUNK
# ============================================================

def transformer_summary(
    text,
    target_words
):

    text = clean_text(text)

    if not text:
        return ""


    input_words = count_words(
        text
    )


    # --------------------------------------------------------
    # Approximate token target
    # --------------------------------------------------------

    desired_tokens = max(
        30,
        int(
            target_words * 1.35
        )
    )


    min_tokens = max(
        20,
        int(
            desired_tokens * 0.45
        )
    )


    max_tokens = max(
        min_tokens + 10,
        desired_tokens
    )


    # DistilBART generation limit

    max_tokens = min(
        max_tokens,
        500
    )


    # --------------------------------------------------------
    # Tokenize input
    # --------------------------------------------------------

    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        max_length=1024

    )


    inputs = {

        key: value.to(DEVICE)

        for key, value
        in inputs.items()

    }


    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.no_grad():

        output = model.generate(

            **inputs,

            max_length=max_tokens,

            min_length=min_tokens,

            num_beams=4,

            length_penalty=1.0,

            no_repeat_ngram_size=3,

            early_stopping=True

        )


    summary = tokenizer.decode(

        output[0],

        skip_special_tokens=True

    )


    return clean_text(
        summary
    )


# ============================================================
# EXTRACT IMPORTANT SENTENCES
# ============================================================

def extract_important_sentences(
    text,
    target_words,
    existing_summary=""
):

    scored = score_sentences(
        text
    )


    if not scored:

        return ""


    # Sort by importance

    ranked = sorted(

        scored,

        key=lambda item:
            item["score"],

        reverse=True

    )


    existing_words = set(

        normalize_word(word)

        for word in
        existing_summary.split()

        if normalize_word(word)

    )


    selected = []

    selected_word_count = 0


    # --------------------------------------------------------
    # First pass:
    # select high-value sentences
    # --------------------------------------------------------

    for item in ranked:

        sentence = item[
            "sentence"
        ]

        sentence_words = item[
            "words"
        ]


        # Don't duplicate sentences already
        # present in the Transformer summary.

        sentence_normalized = set(

            normalize_word(word)

            for word in sentence.split()

            if normalize_word(word)

        )


        overlap = (

            len(
                sentence_normalized
                & existing_words
            )

            /

            max(
                1,
                len(sentence_normalized)
            )

        )


        if overlap > 0.75:

            continue


        # Don't exceed target too much.

        if (

            selected_word_count
            + sentence_words

            <= target_words

        ):

            selected.append(
                item
            )

            selected_word_count += (
                sentence_words
            )


        if (
            selected_word_count
            >= target_words
        ):

            break


    # Restore original document order

    selected.sort(
        key=lambda item:
            item["index"]
    )


    return " ".join(

        item["sentence"]

        for item in selected

    )


# ============================================================
# BUILD TARGET LENGTH SUMMARY
# ============================================================

def build_target_summary(
    original_text,
    transformer_text,
    target_words
):

    transformer_text = clean_text(
        transformer_text
    )


    transformer_words = count_words(
        transformer_text
    )


    # ========================================================
    # CASE 1
    # Transformer is already close
    # ========================================================

    tolerance = max(
        20,
        int(
            target_words * 0.10
        )
    )


    if (

        target_words - tolerance

        <= transformer_words

        <=

        target_words + tolerance

    ):

        return transformer_text


    # ========================================================
    # CASE 2
    # Transformer is TOO SHORT
    # ========================================================

    if transformer_words < (
        target_words - tolerance
    ):

        print(
            "Transformer summary is "
            "shorter than requested."
        )

        print(
            "Adding important original sentences..."
        )


        additional = (
            extract_important_sentences(

                original_text,

                target_words,

                transformer_text

            )
        )


        if additional:

            combined = (

                transformer_text
                + " "
                + additional

            )

        else:

            combined = (
                transformer_text
            )


        # ----------------------------------------------------
        # If combined result is still too long,
        # trim intelligently.
        # ----------------------------------------------------

        return trim_to_target(

            combined,

            target_words

        )


    # ========================================================
    # CASE 3
    # Transformer is TOO LONG
    # ========================================================

    print(
        "Transformer summary is "
        "longer than requested."
    )

    print(
        "Selecting the most important "
        "sentences..."
    )


    return trim_to_target(

        transformer_text,

        target_words

    )


# ============================================================
# TRIM TEXT TO TARGET
# ============================================================

def trim_to_target(
    text,
    target_words
):

    sentences = split_sentences(
        text
    )


    if not sentences:

        words = text.split()

        return " ".join(
            words[:target_words]
        )


    selected = []

    current_words = 0


    # --------------------------------------------------------
    # Prefer complete sentences
    # --------------------------------------------------------

    for sentence in sentences:

        sentence_words = count_words(
            sentence
        )


        if (

            current_words +
            sentence_words

            <= target_words

        ):

            selected.append(
                sentence
            )

            current_words += (
                sentence_words
            )

        else:

            # Don't break if we're already
            # reasonably close to target.

            if current_words >= (
                target_words * 0.90
            ):

                break


    result = " ".join(
        selected
    )


    # --------------------------------------------------------
    # If sentence selection is too short,
    # fill remaining words.
    # --------------------------------------------------------

    result_words = result.split()


    if len(result_words) < (
        target_words * 0.85
    ):

        all_words = text.split()

        existing = len(
            result_words
        )


        remaining = (
            target_words - existing
        )


        if remaining > 0:

            extra = all_words[
                existing:
                existing + remaining
            ]


            if extra:

                result += " " + " ".join(
                    extra
                )


    return clean_text(
        result
    )


# ============================================================
# FINAL TARGET LENGTH CONTROL
# ============================================================

def enforce_target_length(
    original_text,
    summary,
    target_words
):

    summary = clean_text(
        summary
    )


    current_words = count_words(
        summary
    )


    print()
    print(
        f"Before length control: "
        f"{current_words} words"
    )


    # --------------------------------------------------------
    # If very close to target
    # --------------------------------------------------------

    tolerance = max(
        15,
        int(
            target_words * 0.05
        )
    )


    if abs(
        current_words -
        target_words
    ) <= tolerance:

        return summary


    # --------------------------------------------------------
    # If short
    # --------------------------------------------------------

    if current_words < target_words:

        print(
            "Applying sentence expansion..."
        )


        additional = (
            extract_important_sentences(

                original_text,

                target_words,

                summary

            )
        )


        combined = clean_text(

            summary
            + " "
            + additional

        )


        return trim_to_target(

            combined,

            target_words

        )


    # --------------------------------------------------------
    # If long
    # --------------------------------------------------------

    print(
        "Applying intelligent trimming..."
    )


    return trim_to_target(

        summary,

        target_words

    )


# ============================================================
# MAIN SUMMARIZATION FUNCTION
# ============================================================

def summarize_document(
    text,
    reduction_percentage=70
):

    # --------------------------------------------------------
    # Clean input
    # --------------------------------------------------------

    text = clean_text(
        text
    )


    if not text:

        return {

            "summary": "",

            "original_words": 0,

            "summary_words": 0,

            "chunks": 0

        }


    # --------------------------------------------------------
    # Validate reduction
    # --------------------------------------------------------

    try:

        reduction_percentage = int(
            reduction_percentage
        )

    except ValueError:

        reduction_percentage = 70


    reduction_percentage = max(
        10,
        min(
            90,
            reduction_percentage
        )
    )


    # --------------------------------------------------------
    # Original words
    # --------------------------------------------------------

    original_words = count_words(
        text
    )


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    target_words = max(

        5,

        round(

            original_words *

            (
                1 -
                reduction_percentage / 100
            )

        )

    )


    print()
    print("=" * 70)

    print("SUMMARIZATION STARTED")

    print("=" * 70)

    print(
        f"Original words : {original_words}"
    )

    print(
        f"Requested reduction : "
        f"{reduction_percentage}%"
    )

    print(
        f"Target summary words : "
        f"{target_words}"
    )


    # ========================================================
    # SMALL DOCUMENT
    # ========================================================

    if original_words <= 900:

        print(
            "\nProcessing as one document..."
        )


        transformer_result = (
            transformer_summary(

                text,

                target_words

            )
        )


        final_summary = (
            build_target_summary(

                text,

                transformer_result,

                target_words

            )
        )


        final_summary = (
            enforce_target_length(

                text,

                final_summary,

                target_words

            )
        )


        chunks_count = 1


    # ========================================================
    # LARGE DOCUMENT
    # ========================================================

    else:

        print(
            "\nLarge document detected."
        )


        chunks = (
            split_text_into_chunks(

                text,

                max_tokens=850

            )
        )


        chunks_count = len(
            chunks
        )


        print(
            f"Total chunks : "
            f"{chunks_count}"
        )


        chunk_summaries = []


        # ----------------------------------------------------
        # Process each chunk
        # ----------------------------------------------------

        for index, chunk in enumerate(

            chunks,

            start=1

        ):

            chunk_words = count_words(
                chunk
            )


            # Proportional target

            chunk_target = max(

                30,

                round(

                    target_words *

                    (
                        chunk_words /
                        original_words
                    )

                )

            )


            print(
                f"\nProcessing chunk "
                f"{index}/{chunks_count}"
            )


            print(
                f"Chunk words : "
                f"{chunk_words}"
            )


            print(
                f"Chunk target : "
                f"{chunk_target}"
            )


            chunk_summary = (
                transformer_summary(

                    chunk,

                    chunk_target

                )
            )


            if chunk_summary:

                chunk_summaries.append(
                    chunk_summary
                )


        # ----------------------------------------------------
        # Combine chunk summaries
        # ----------------------------------------------------

        combined_summary = clean_text(

            " ".join(
                chunk_summaries
            )

        )


        print()
        print(
            f"Combined summary : "
            f"{count_words(combined_summary)} "
            f"words"
        )


        # ----------------------------------------------------
        # Final target controller
        # ----------------------------------------------------

        final_summary = (
            build_target_summary(

                text,

                combined_summary,

                target_words

            )
        )


        final_summary = (
            enforce_target_length(

                text,

                final_summary,

                target_words

            )
        )


    # ========================================================
    # FINAL CLEANING
    # ========================================================

    final_summary = clean_text(
        final_summary
    )


    final_words = count_words(
        final_summary
    )


    # ========================================================
    # ACTUAL COMPRESSION
    # ========================================================

    if original_words > 0:

        actual_compression = (

            1 -

            (
                final_words /
                original_words
            )

        ) * 100

    else:

        actual_compression = 0


    actual_compression = round(
        actual_compression,
        1
    )


    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("=" * 70)

    print(
        "SUMMARIZATION COMPLETE"
    )

    print("=" * 70)

    print(
        f"Original words : "
        f"{original_words}"
    )

    print(
        f"Target words   : "
        f"{target_words}"
    )

    print(
        f"Final words    : "
        f"{final_words}"
    )

    print(
        f"Requested compression : "
        f"{reduction_percentage}%"
    )

    print(
        f"Actual compression : "
        f"{actual_compression}%"
    )

    print(
        f"Chunks : "
        f"{chunks_count}"
    )

    print("=" * 70)


    return {

        "summary":
            final_summary,

        "original_words":
            original_words,

        "summary_words":
            final_words,

        "chunks":
            chunks_count,

        "requested_reduction":
            reduction_percentage,

        "target_words":
            target_words,

        "compression":
            actual_compression

    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    test_text = """

    Natural Language Processing is a field
    of artificial intelligence that focuses
    on enabling computers to understand,
    interpret and generate human language.

    NLP is used in machine translation,
    sentiment analysis, chatbots,
    information retrieval, question answering,
    text classification and text summarization.

    Modern NLP systems use transformer
    architectures because attention mechanisms
    allow models to understand relationships
    between words and sentences.

    Transformer models have significantly
    improved language understanding and
    generation.

    Text summarization is one of the important
    applications of NLP because it allows
    users to obtain important information
    from a long document in a shorter form.

    Summarization systems can be extractive,
    abstractive or hybrid.

    Extractive systems select important
    sentences from the original document.

    Abstractive systems generate new sentences
    that represent the meaning of the original
    document.

    Hybrid systems combine both approaches
    to improve information coverage and
    control the length of the generated summary.

    """

    result = summarize_document(

        test_text,

        reduction_percentage=70

    )


    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        result["summary"]
    )

    print()
    print(
        f"Original words : "
        f"{result['original_words']}"
    )

    print(
        f"Summary words : "
        f"{result['summary_words']}"
    )

    print(
        f"Target words : "
        f"{result['target_words']}"
    )

    print(
        f"Compression : "
        f"{result['compression']}%"
    )

    print("=" * 70)