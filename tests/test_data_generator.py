"""
Tests for data_generator.py - Synthetic data generation module.

Tests cover:
- Filler text generation (English and Hebrew)
- Fact embedding at different positions
- Needle in haystack document generation
- Context size document generation
- Hebrew medical document generation
- Dataset generation and file I/O
"""

import pytest
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generator import DataGenerator


class TestDataGeneratorInit:
    """Test DataGenerator initialization."""

    def test_init_default_seed(self):
        """Test initialization with default seed."""
        generator = DataGenerator()
        assert generator.seed == 42
        assert generator.faker_en is not None
        assert generator.faker_he is not None

    def test_init_custom_seed(self):
        """Test initialization with custom seed."""
        generator = DataGenerator(seed=123)
        assert generator.seed == 123

    def test_reproducibility_with_same_seed(self):
        """Test that same seed produces same output."""
        gen1 = DataGenerator(seed=42)
        gen2 = DataGenerator(seed=42)

        text1 = gen1.generate_filler_text(100)
        text2 = gen2.generate_filler_text(100)

        assert text1 == text2


class TestGenerateFillerText:
    """Test filler text generation."""

    def test_generate_english_text(self):
        """Test English text generation."""
        generator = DataGenerator(seed=42)
        text = generator.generate_filler_text(num_words=50, language='en')

        assert isinstance(text, str)
        assert len(text) > 0

        # Check approximate word count
        word_count = len(text.split())
        assert 40 <= word_count <= 60  # Allow some variance

    def test_generate_hebrew_text(self):
        """Test Hebrew text generation."""
        generator = DataGenerator(seed=42)
        text = generator.generate_filler_text(num_words=50, language='he')

        assert isinstance(text, str)
        assert len(text) > 0

        # Hebrew text should contain Hebrew characters
        # Check for at least some non-ASCII characters
        has_hebrew = any(ord(char) > 127 for char in text)
        assert has_hebrew

    def test_word_count_approximation(self):
        """Test that generated text approximates requested word count."""
        generator = DataGenerator(seed=42)

        for target_words in [10, 50, 100, 200]:
            text = generator.generate_filler_text(num_words=target_words)
            actual_words = len(text.split())

            # Allow 20% variance
            lower_bound = target_words * 0.8
            upper_bound = target_words * 1.2

            assert lower_bound <= actual_words <= upper_bound, \
                f"Word count {actual_words} not in range [{lower_bound}, {upper_bound}]"

    def test_different_texts_with_different_calls(self):
        """Test that multiple calls produce different text (within same generator)."""
        generator = DataGenerator(seed=42)

        text1 = generator.generate_filler_text(100)
        text2 = generator.generate_filler_text(100)

        # Texts should be different
        assert text1 != text2


class TestEmbedFactInText:
    """Test fact embedding in text."""

    def test_embed_at_start(self):
        """Test embedding fact at start of text."""
        generator = DataGenerator(seed=42)
        text = "Sentence one. Sentence two. Sentence three."
        fact = "This is the important fact."

        result = generator.embed_fact_in_text(text, fact, position='start')

        assert fact.rstrip('.') in result
        # Fact should appear near the beginning
        assert result.index(fact.rstrip('.')) < 50

    def test_embed_at_middle(self):
        """Test embedding fact in middle of text."""
        generator = DataGenerator(seed=42)
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        fact = "This is the important fact."

        result = generator.embed_fact_in_text(text, fact, position='middle')

        assert fact.rstrip('.') in result

        # Fact should be roughly in the middle
        fact_position = result.index(fact.rstrip('.'))
        text_length = len(result)
        assert 0.3 * text_length < fact_position < 0.7 * text_length

    def test_embed_at_end(self):
        """Test embedding fact at end of text."""
        generator = DataGenerator(seed=42)
        text = "Sentence one. Sentence two. Sentence three."
        fact = "This is the important fact."

        result = generator.embed_fact_in_text(text, fact, position='end')

        assert fact.rstrip('.') in result

        # Fact should appear near the end
        fact_position = result.index(fact.rstrip('.'))
        text_length = len(result)
        assert fact_position > 0.6 * text_length

    def test_embed_preserves_text_structure(self):
        """Test that embedding preserves sentence structure."""
        generator = DataGenerator(seed=42)
        text = "First sentence. Second sentence. Third sentence."
        fact = "Important fact"

        result = generator.embed_fact_in_text(text, fact, position='middle')

        # Result should still contain original sentences
        assert "First sentence" in result
        assert "Second sentence" in result or "Third sentence" in result

    def test_embed_fact_with_trailing_period(self):
        """Test that trailing periods are handled correctly."""
        generator = DataGenerator(seed=42)
        text = "Sentence one. Sentence two."
        fact = "Fact with period."

        result = generator.embed_fact_in_text(text, fact, position='middle')

        # Should not have double periods
        assert ".." not in result


class TestGenerateNeedleHaystackDocument:
    """Test needle in haystack document generation."""

    def test_generate_with_default_secret(self):
        """Test document generation with auto-generated secret."""
        generator = DataGenerator(seed=42)
        doc_data = generator.generate_needle_haystack_document()

        assert "document" in doc_data
        assert "fact" in doc_data
        assert "position" in doc_data
        assert "secret_value" in doc_data

        # Secret should be in the document
        assert doc_data["secret_value"] in doc_data["document"]

    def test_generate_with_custom_secret(self):
        """Test document generation with custom secret."""
        generator = DataGenerator(seed=42)
        custom_secret = "CUSTOM123SECRET"

        doc_data = generator.generate_needle_haystack_document(
            secret_value=custom_secret
        )

        assert doc_data["secret_value"] == custom_secret
        assert custom_secret in doc_data["document"]
        assert custom_secret in doc_data["fact"]

    def test_generate_different_positions(self):
        """Test document generation at different positions."""
        generator = DataGenerator(seed=42)

        for position in ['start', 'middle', 'end']:
            doc_data = generator.generate_needle_haystack_document(
                position=position,
                secret_value="TEST123"
            )

            assert doc_data["position"] == position
            assert "TEST123" in doc_data["document"]

    def test_document_word_count(self):
        """Test that document has approximately correct word count."""
        generator = DataGenerator(seed=42)

        for word_count in [100, 200, 300]:
            doc_data = generator.generate_needle_haystack_document(words=word_count)

            actual_words = len(doc_data["document"].split())

            # Allow 20% variance
            lower_bound = word_count * 0.8
            upper_bound = word_count * 1.2

            assert lower_bound <= actual_words <= upper_bound

    def test_fact_format(self):
        """Test that fact follows the expected template."""
        generator = DataGenerator(seed=42)
        secret = "ABC123"

        doc_data = generator.generate_needle_haystack_document(secret_value=secret)

        # Fact should follow template
        assert "password" in doc_data["fact"].lower()
        assert secret in doc_data["fact"]


class TestGenerateNeedleHaystackDataset:
    """Test needle in haystack dataset generation."""

    def test_generate_dataset(self):
        """Test complete dataset generation."""
        generator = DataGenerator(seed=42)

        dataset = generator.generate_needle_haystack_dataset(
            num_docs=2,
            words_per_doc=100
        )

        # Should have 2 docs per position * 3 positions = 6 total
        assert len(dataset) == 6

        # Check all positions are represented
        positions = [doc["position"] for doc in dataset]
        assert positions.count("start") == 2
        assert positions.count("middle") == 2
        assert positions.count("end") == 2

    def test_dataset_has_doc_ids(self):
        """Test that all documents have unique IDs."""
        generator = DataGenerator(seed=42)

        dataset = generator.generate_needle_haystack_dataset(num_docs=2)

        doc_ids = [doc["doc_id"] for doc in dataset]

        # All IDs should be unique
        assert len(doc_ids) == len(set(doc_ids))

        # IDs should follow pattern
        assert any("start" in doc_id for doc_id in doc_ids)
        assert any("middle" in doc_id for doc_id in doc_ids)
        assert any("end" in doc_id for doc_id in doc_ids)

    def test_save_dataset_to_file(self, temp_directory):
        """Test saving dataset to JSON file."""
        generator = DataGenerator(seed=42)
        save_path = temp_directory / "test_dataset.json"

        dataset = generator.generate_needle_haystack_dataset(
            num_docs=1,
            save_path=save_path
        )

        # File should exist
        assert save_path.exists()

        # Load and verify
        with open(save_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        assert len(loaded_data) == 3  # 1 doc per position
        assert loaded_data == dataset


class TestGenerateContextSizeDocument:
    """Test context size document generation."""

    def test_generate_with_default_revenue(self):
        """Test document generation with auto-generated revenue."""
        generator = DataGenerator(seed=42)
        doc_data = generator.generate_context_size_document()

        assert "document" in doc_data
        assert "revenue_value" in doc_data
        assert "company_name" in doc_data
        assert "year" in doc_data

        # Revenue should be in document
        assert doc_data["revenue_value"] in doc_data["document"]

    def test_generate_with_custom_revenue(self):
        """Test document generation with custom revenue."""
        generator = DataGenerator(seed=42)
        custom_revenue = "$999 million"

        doc_data = generator.generate_context_size_document(
            revenue_value=custom_revenue
        )

        assert doc_data["revenue_value"] == custom_revenue
        assert custom_revenue in doc_data["document"]

    def test_company_name_in_document(self):
        """Test that company name appears in document."""
        generator = DataGenerator(seed=42)
        doc_data = generator.generate_context_size_document()

        assert doc_data["company_name"] in doc_data["document"]

    def test_year_range(self):
        """Test that year is in expected range."""
        generator = DataGenerator(seed=42)

        # Generate multiple documents
        for _ in range(10):
            doc_data = generator.generate_context_size_document()
            year = doc_data["year"]

            assert 2020 <= year <= 2024


class TestGenerateContextSizeDataset:
    """Test context size dataset generation."""

    def test_generate_dataset_multiple_sizes(self):
        """Test dataset generation for multiple context sizes."""
        generator = DataGenerator(seed=42)

        doc_counts = [2, 5, 10]
        datasets = generator.generate_context_size_dataset(doc_counts=doc_counts)

        # Should have datasets for each count
        assert len(datasets) == 3
        assert 2 in datasets
        assert 5 in datasets
        assert 10 in datasets

        # Each dataset should have correct number of docs
        assert len(datasets[2]) == 2
        assert len(datasets[5]) == 5
        assert len(datasets[10]) == 10

    def test_one_document_has_fact(self):
        """Test that exactly one document per set has the revenue fact."""
        generator = DataGenerator(seed=42)

        datasets = generator.generate_context_size_dataset(doc_counts=[5])
        docs = datasets[5]

        # Count documents with revenue_value
        docs_with_revenue = sum(1 for doc in docs if doc.get("revenue_value") is not None)

        assert docs_with_revenue == 1

    def test_save_datasets_to_files(self, temp_directory):
        """Test saving datasets to JSON files."""
        generator = DataGenerator(seed=42)

        datasets = generator.generate_context_size_dataset(
            doc_counts=[2, 5],
            save_dir=temp_directory
        )

        # Files should exist
        assert (temp_directory / "context_size_2.json").exists()
        assert (temp_directory / "context_size_5.json").exists()

        # Load and verify
        with open(temp_directory / "context_size_5.json", 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        assert len(loaded_data) == 5


class TestGenerateHebrewMedicalDocument:
    """Test Hebrew medical document generation."""

    def test_generate_hebrew_document(self):
        """Test Hebrew medical document generation."""
        generator = DataGenerator(seed=42)

        drug_name = "פנדול"
        side_effects = ["כאב ראש", "בחילה", "סחרחורת"]

        document = generator.generate_hebrew_medical_document(drug_name, side_effects)

        assert isinstance(document, str)
        assert len(document) > 0

        # Drug name should be in document
        assert drug_name in document

        # Side effects should be mentioned
        for effect in side_effects:
            assert effect in document

    def test_document_has_hebrew_text(self):
        """Test that document contains Hebrew characters."""
        generator = DataGenerator(seed=42)

        document = generator.generate_hebrew_medical_document(
            "אספירין",
            ["עייפות", "יובש בפה"]
        )

        # Should have Hebrew characters
        has_hebrew = any(ord(char) > 127 for char in document)
        assert has_hebrew


class TestGenerateHebrewCorpus:
    """Test Hebrew corpus generation."""

    def test_generate_corpus(self):
        """Test Hebrew corpus generation."""
        generator = DataGenerator(seed=42)

        corpus = generator.generate_hebrew_corpus(num_docs=10)

        assert len(corpus) == 10

        # All should have required fields
        for doc in corpus:
            assert "doc_id" in doc
            assert "topic" in doc
            assert "content" in doc

    def test_corpus_has_medical_documents(self):
        """Test that corpus includes medical documents."""
        generator = DataGenerator(seed=42)

        corpus = generator.generate_hebrew_corpus(num_docs=20)

        # At least some should be medical
        medical_docs = [doc for doc in corpus if doc["topic"] == "medicine"]

        assert len(medical_docs) > 0

        # Medical docs should have drug info
        for doc in medical_docs:
            assert "drug_name" in doc
            assert "side_effects" in doc

    def test_save_corpus_to_file(self, temp_directory):
        """Test saving corpus to JSON file."""
        generator = DataGenerator(seed=42)

        corpus = generator.generate_hebrew_corpus(
            num_docs=5,
            save_dir=temp_directory
        )

        save_path = temp_directory / "hebrew_corpus.json"

        # File should exist
        assert save_path.exists()

        # Load and verify
        with open(save_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        assert len(loaded_data) == 5

    def test_corpus_reproducibility(self):
        """Test that corpus generation is reproducible."""
        gen1 = DataGenerator(seed=42)
        gen2 = DataGenerator(seed=42)

        corpus1 = gen1.generate_hebrew_corpus(num_docs=5)
        corpus2 = gen2.generate_hebrew_corpus(num_docs=5)

        # Should be identical
        assert corpus1 == corpus2


class TestDataGeneratorIntegration:
    """Integration tests for DataGenerator."""

    def test_end_to_end_experiment1_data(self, temp_directory):
        """Test end-to-end Experiment 1 data generation."""
        generator = DataGenerator(seed=42)

        # Generate complete dataset
        dataset = generator.generate_needle_haystack_dataset(
            num_docs=3,
            words_per_doc=150,
            save_path=temp_directory / "exp1_data.json"
        )

        # Validate dataset
        assert len(dataset) == 9  # 3 docs * 3 positions

        # All documents should be usable
        for doc in dataset:
            assert len(doc["document"]) > 100
            assert doc["secret_value"] in doc["document"]
            assert doc["position"] in ["start", "middle", "end"]

    def test_end_to_end_experiment2_data(self, temp_directory):
        """Test end-to-end Experiment 2 data generation."""
        generator = DataGenerator(seed=42)

        # Generate datasets for different sizes
        datasets = generator.generate_context_size_dataset(
            doc_counts=[2, 5, 10],
            save_dir=temp_directory
        )

        # Validate all datasets
        for count, docs in datasets.items():
            assert len(docs) == count

            # Exactly one should have the target revenue
            revenue_count = sum(1 for d in docs if d.get("revenue_value") is not None)
            assert revenue_count == 1

    def test_end_to_end_experiment3_data(self, temp_directory):
        """Test end-to-end Experiment 3 data generation."""
        generator = DataGenerator(seed=42)

        # Generate Hebrew corpus
        corpus = generator.generate_hebrew_corpus(
            num_docs=15,
            save_dir=temp_directory
        )

        # Validate corpus
        assert len(corpus) == 15

        # Should have mix of topics
        topics = [doc["topic"] for doc in corpus]
        unique_topics = set(topics)
        assert len(unique_topics) > 1  # Multiple topics
