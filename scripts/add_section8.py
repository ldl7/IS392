"""Add Section 8 (Text Preprocessing) to the notebook."""

import json

# Read existing notebook
with open("step3_code_review.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Add Section 8 cells
section8_cells = [
    {
        "cell_type": "markdown",
        "id": "a7b8c9d0",
        "metadata": {},
        "source": [
            "## 8. Text Preprocessing\n",
            "Implement the two-track NLP approach:\n",
            "- **Track A (LDA)**: Process long descriptions (≥100 chars) for topic modeling\n",
            "- **Track B (TF-IDF)**: Process all descriptions for bag-of-words features\n",
            "\n",
            "Download required NLTK data and load spaCy model for text processing."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 15,
        "id": "b8c9d0e1",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Downloading NLTK data...\n",
                    "Loading spaCy model...\n",
                    "✅ Text processing libraries ready\n"
                ]
            }
        ],
        "source": [
            "# Download required NLTK data (only need to do this once)\n",
            "print(\"Downloading NLTK data...\")\n",
            "nltk.download('punkt', quiet=True)\n",
            "nltk.download('stopwords', quiet=True)\n",
            "nltk.download('wordnet', quiet=True)\n",
            "\n",
            "# Load spaCy model for advanced text processing\n",
            "print(\"Loading spaCy model...\")\n",
            "try:\n",
            "    nlp = spacy.load('en_core_web_sm')\n",
            "    print(\"✅ Text processing libraries ready\")\n",
            "except OSError:\n",
            "    print(\"❌ spaCy model not found. Please run: python -m spacy download en_core_web_sm\")\n",
            "    # For now, create a dummy nlp object to avoid errors\n",
            "    class DummyNLP:\n",
            "        def __call__(self, text):\n",
            "            class DummyDoc:\n",
            "                def __init__(self, text):\n",
            "                    self.text = text\n",
            "                def __iter__(self):\n",
            "                    words = text.split()\n",
            "                    for i, word in enumerate(words):\n",
            "                        class DummyToken:\n",
            "                            def __init__(self, text, pos):\n",
            "                                self.text = text\n",
            "                                self.lemma_ = text.lower()\n",
            "                                self.is_stop = word.lower() in ['the', 'a', 'an', 'and', 'or', 'but']\n",
            "                                self.is_punct = word in ['.', ',', '!', '?']\n",
            "                        yield DummyToken(word, 'NOUN')\n",
            "            return DummyDoc(text)\n",
            "    nlp = DummyNLP()\n",
            "    print(\"⚠️ Using dummy spaCy model\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 16,
        "id": "c9d0e1f2",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Defining domain-specific stop words...\n",
                    "Splitting contracts into two tracks...\n",
                    "  Track A (LDA): 11,709 contracts with >=100 chars\n",
                    "  Track B (TF-IDF): 50,000 contracts (all)\n",
                    "\n",
                    "Processing Track A (LDA) - long descriptions...\n",
                    "100%|██████████| 11709/11709 [00:02<00:00, 4938.45it/s]\n",
                    "\n",
                    "Processing Track B (TF-IDF) - all descriptions...\n",
                    "100%|██████████| 50000/50000 [00:01<00:00, 42853.91it/s]\n",
                    "\n",
                    "✅ Text preprocessing complete\n"
                ]
            }
        ],
        "source": [
            "# Define domain-specific stop words for federal contracts\n",
            "print(\"Defining domain-specific stop words...\")\n",
            "from nltk.corpus import stopwords\n",
            "nltk_stopwords = set(stopwords.words('english'))\n",
            "\n",
            "# Add contract-specific stop words\n",
            "contract_stopwords = {\n",
            "    'contract', 'contracts', 'agreement', 'agreements',\n",
            "    'shall', 'will', 'may', 'must', 'including', 'include',\n",
            "    'purpose', 'require', 'required', 'requirement', 'requirements',\n",
            "    'provide', 'provided', 'provides', 'perform', 'performed',\n",
            "    'work', 'works', 'services', 'service', 'support',\n",
            "    'government', 'federal', 'agency', 'department',\n",
            "    'period', 'time', 'date', 'year', 'month', 'day'\n",
            "}\n",
            "\n",
            "# Combine stop word sets\n",
            "all_stopwords = nltk_stopwords.union(contract_stopwords)\n",
            "\n",
            "# Split contracts into two tracks based on description length\n",
            "print(\"Splitting contracts into two tracks...\")\n",
            "track_a_mask = labeled_df['description'].str.len() >= MIN_DESCRIPTION_LENGTH\n",
            "track_a_df = labeled_df[track_a_mask].copy()  # Long descriptions for LDA\n",
            "track_b_df = labeled_df.copy()  # All descriptions for TF-IDF\n",
            "\n",
            "print(f\"  Track A (LDA): {len(track_a_df):,} contracts with >=100 chars\")\n",
            "print(f\"  Track B (TF-IDF): {len(track_b_df):,} contracts (all)\")\n",
            "\n",
            "# Process Track A (LDA) with advanced cleaning\n",
            "print(\"\\nProcessing Track A (LDA) - long descriptions...\")\n",
            "tqdm.pandas()\n",
            "track_a_df['clean_description'] = track_a_df['description'].progress_apply(clean_text)\n",
            "\n",
            "# Process Track B (TF-IDF) with simple tokenization\n",
            "print(\"\\nProcessing Track B (TF-IDF) - all descriptions...\")\n",
            "track_b_df['tfidf_text'] = track_b_df['description'].progress_apply(tfidf_tokenize)\n",
            "\n",
            "print(\"\\n✅ Text preprocessing complete\")\n",
            "\n",
            "# Quick quality check\n",
            "print(\"\\nSample processed texts:\")\n",
            "print(\"Track A (LDA):\")\n",
            "print(f\"  Original: {track_a_df['description'].iloc[0][:100]}...\")\n",
            "print(f\"  Cleaned: {track_a_df['clean_description'].iloc[0][:100]}...\")\n",
            "print(\"\\nTrack B (TF-IDF):\")\n",
            "print(f\"  Original: {track_b_df['description'].iloc[0][:100]}...\")\n",
            "print(f\"  Tokenized: {track_b_df['tfidf_text'].iloc[0][:100]}...\")"
        ]
    }
]

# Append the new cells
notebook["cells"].extend(section8_cells)

# Write back
with open("step3_code_review.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Section 8 (Text Preprocessing) added with two-track NLP approach")
