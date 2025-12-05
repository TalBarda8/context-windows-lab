"""
Data generator for creating synthetic documents and corpora.

This module generates synthetic text data for all experiments,
including filler text, embedded facts, and domain-specific content.
"""

import random
import string
from typing import List, Dict, Tuple, Optional
from faker import Faker
import json
from pathlib import Path

# Import configuration
from config import (
    NEEDLE_HAYSTACK_DIR,
    CONTEXT_SIZE_DIR,
    HEBREW_CORPUS_DIR,
    EXP1_CONFIG,
    EXP2_CONFIG,
    EXP3_CONFIG,
)


class DataGenerator:
    """Generate synthetic data for context window experiments."""

    def __init__(self, seed: int = 42):
        """
        Initialize data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        self.faker_en = Faker('en_US')
        self.faker_he = Faker('he_IL')
        Faker.seed(seed)

    def generate_filler_text(self, num_words: int,
                            language: str = 'en') -> str:
        """
        Generate realistic filler text.

        Args:
            num_words: Target number of words
            language: Language ('en' or 'he')

        Returns:
            Generated filler text
        """
        faker = self.faker_he if language == 'he' else self.faker_en

        sentences = []
        current_words = 0

        while current_words < num_words:
            # Generate a mix of sentences and paragraphs
            if random.random() < 0.7:
                sentence = faker.sentence(nb_words=random.randint(8, 20))
            else:
                sentence = faker.paragraph(nb_sentences=random.randint(2, 4))

            sentences.append(sentence)
            current_words += len(sentence.split())

        text = " ".join(sentences)

        # Trim to approximate word count
        words = text.split()
        if len(words) > num_words:
            words = words[:num_words]

        return " ".join(words)

    def embed_fact_in_text(self, text: str, fact: str,
                          position: str = 'middle') -> str:
        """
        Embed a critical fact at specified position in text.

        Args:
            text: Base text
            fact: Fact to embed
            position: Where to embed ('start', 'middle', 'end')

        Returns:
            Text with embedded fact
        """
        sentences = text.split('. ')

        if position == 'start':
            # Insert at beginning
            sentences.insert(0, fact.rstrip('.'))
        elif position == 'end':
            # Append at end
            sentences.append(fact.rstrip('.'))
        else:  # middle
            # Insert in the middle
            mid_point = len(sentences) // 2
            sentences.insert(mid_point, fact.rstrip('.'))

        return '. '.join(sentences) + '.'

    def generate_needle_haystack_document(self,
                                         words: int = 200,
                                         position: str = 'middle',
                                         secret_value: Optional[str] = None
                                         ) -> Dict[str, str]:
        """
        Generate a document with embedded secret fact (Experiment 1).

        Args:
            words: Number of words in document
            position: Position of fact ('start', 'middle', 'end')
            secret_value: Secret value to embed (random if None)

        Returns:
            Dictionary with document, fact, position, and secret value
        """
        if secret_value is None:
            # Generate random secret (password-like string)
            secret_value = ''.join(random.choices(
                string.ascii_letters + string.digits, k=12
            ))

        fact = EXP1_CONFIG["critical_fact_template"].format(password=secret_value)

        # Generate filler text
        filler = self.generate_filler_text(words - 10)  # Leave room for fact

        # Embed fact at specified position
        document = self.embed_fact_in_text(filler, fact, position)

        return {
            "document": document,
            "fact": fact,
            "position": position,
            "secret_value": secret_value,
        }

    def generate_needle_haystack_dataset(self,
                                         num_docs: int = 5,
                                         words_per_doc: int = 200,
                                         save_path: Optional[Path] = None
                                         ) -> List[Dict[str, str]]:
        """
        Generate complete dataset for Experiment 1.

        Args:
            num_docs: Number of documents per position
            words_per_doc: Words per document
            save_path: Path to save dataset (JSON)

        Returns:
            List of generated documents
        """
        positions = EXP1_CONFIG["positions"]
        dataset = []

        for position in positions:
            for i in range(num_docs):
                doc_data = self.generate_needle_haystack_document(
                    words=words_per_doc,
                    position=position
                )
                doc_data["doc_id"] = f"{position}_{i}"
                dataset.append(doc_data)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"Dataset saved to {save_path}")

        return dataset

    def generate_context_size_document(self,
                                       words: int = 200,
                                       revenue_value: Optional[str] = None
                                       ) -> Dict[str, str]:
        """
        Generate a business document with revenue fact (Experiment 2).

        Args:
            words: Number of words
            revenue_value: Revenue value to embed

        Returns:
            Dictionary with document and revenue value
        """
        if revenue_value is None:
            # Generate random revenue (in millions)
            revenue_value = f"${random.randint(10, 999)} million"

        # Create a business-themed document
        company_name = self.faker_en.company()
        year = random.randint(2020, 2024)

        fact = f"The company {company_name} reported annual revenue of {revenue_value} for the fiscal year {year}."

        # Generate business-themed filler
        filler = self.generate_filler_text(words - 20)

        # Embed fact in random position
        document = self.embed_fact_in_text(
            filler, fact, random.choice(['start', 'middle', 'end'])
        )

        return {
            "document": document,
            "revenue_value": revenue_value,
            "company_name": company_name,
            "year": year,
        }

    def generate_context_size_dataset(self,
                                      doc_counts: List[int] = None,
                                      words_per_doc: int = 200,
                                      save_dir: Optional[Path] = None
                                      ) -> Dict[int, List[Dict[str, str]]]:
        """
        Generate datasets for different context sizes (Experiment 2).

        Args:
            doc_counts: List of document counts to generate
            words_per_doc: Words per document
            save_dir: Directory to save datasets

        Returns:
            Dictionary mapping doc count to list of documents
        """
        if doc_counts is None:
            doc_counts = EXP2_CONFIG["document_counts"]

        datasets = {}

        for count in doc_counts:
            # Use same revenue value across all documents in this set
            revenue_value = f"${random.randint(100, 999)} million"

            docs = []
            for i in range(count):
                # Only embed fact in one random document
                if i == random.randint(0, count - 1):
                    doc_data = self.generate_context_size_document(
                        words=words_per_doc,
                        revenue_value=revenue_value
                    )
                else:
                    # Generate document without the target fact
                    doc_data = {
                        "document": self.generate_filler_text(words_per_doc),
                        "revenue_value": None,
                    }

                doc_data["doc_id"] = f"size_{count}_doc_{i}"
                docs.append(doc_data)

            datasets[count] = docs

            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"context_size_{count}.json"
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(docs, f, indent=2, ensure_ascii=False)
                print(f"Dataset for {count} docs saved to {save_path}")

        return datasets

    def generate_hebrew_medical_document(self, drug_name: str,
                                        side_effects: List[str]) -> str:
        """
        Generate Hebrew medical document about a drug.

        Args:
            drug_name: Name of the drug
            side_effects: List of side effects

        Returns:
            Hebrew medical document
        """
        # Create Hebrew medical content
        intro = f"{drug_name} היא תרופה המשמשת לטיפול במחלות שונות."

        side_effects_text = f"תופעות הלוואי הידועות של {drug_name} כוללות: " + \
                           ", ".join(side_effects) + "."

        # Add filler content
        filler = self.generate_filler_text(150, language='he')

        # Combine with fact in middle
        sentences = filler.split('. ')
        mid = len(sentences) // 2
        sentences.insert(mid, side_effects_text)

        return intro + " " + ". ".join(sentences) + "."

    def generate_hebrew_corpus(self,
                               num_docs: int = 20,
                               save_dir: Optional[Path] = None
                               ) -> List[Dict[str, str]]:
        """
        Generate Hebrew corpus for Experiment 3.

        Args:
            num_docs: Number of documents to generate
            save_dir: Directory to save documents

        Returns:
            List of document dictionaries
        """
        topics = EXP3_CONFIG["topics"]
        corpus = []

        drugs = ["פנדול", "אקמול", "נורופן", "אספירין", "אנטיביוטיקה"]
        side_effects_list = [
            ["כאב ראש", "בחילה", "סחרחורת"],
            ["עייפות", "יובש בפה", "נדודי שינה"],
            ["גירוי קיבה", "אדמומיות", "פריחה"],
        ]

        for i in range(num_docs):
            topic = random.choice(topics)

            if topic == "medicine":
                drug = random.choice(drugs)
                side_effects = random.choice(side_effects_list)
                content = self.generate_hebrew_medical_document(drug, side_effects)

                doc = {
                    "doc_id": f"he_med_{i}",
                    "topic": topic,
                    "content": content,
                    "drug_name": drug,
                    "side_effects": side_effects,
                }
            else:
                # Generate generic Hebrew content for other topics
                content = self.generate_filler_text(200, language='he')
                doc = {
                    "doc_id": f"he_{topic}_{i}",
                    "topic": topic,
                    "content": content,
                }

            corpus.append(doc)

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "hebrew_corpus.json"
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(corpus, f, indent=2, ensure_ascii=False)
            print(f"Hebrew corpus saved to {save_path}")

        return corpus


def main():
    """Generate all datasets for experiments."""
    generator = DataGenerator(seed=42)

    print("Generating Experiment 1 data (Needle in Haystack)...")
    exp1_data = generator.generate_needle_haystack_dataset(
        num_docs=EXP1_CONFIG["num_documents"],
        words_per_doc=EXP1_CONFIG["words_per_document"],
        save_path=NEEDLE_HAYSTACK_DIR / "dataset.json"
    )
    print(f"Generated {len(exp1_data)} documents for Experiment 1\n")

    print("Generating Experiment 2 data (Context Size)...")
    exp2_data = generator.generate_context_size_dataset(
        save_dir=CONTEXT_SIZE_DIR
    )
    print(f"Generated datasets for {len(exp2_data)} different sizes\n")

    print("Generating Experiment 3 data (Hebrew Corpus)...")
    exp3_data = generator.generate_hebrew_corpus(
        num_docs=EXP3_CONFIG["num_documents"],
        save_dir=HEBREW_CORPUS_DIR
    )
    print(f"Generated {len(exp3_data)} Hebrew documents\n")

    print("All datasets generated successfully!")


if __name__ == "__main__":
    main()
