"""Add Section 9 (Topic Modeling and TF-IDF) to the notebook."""

import json

# Read existing notebook
with open("step3_code_review.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Add Section 9 cells
section9_cells = [
    {
        "cell_type": "markdown",
        "id": "d0e1f2g3",
        "metadata": {},
        "source": [
            "## 9. Topic Modeling (LDA) and TF-IDF Feature Extraction\n",
            "Build the text feature matrices:\n",
            "- **Track A (LDA)**: Train topic model on long descriptions, extract topic proportions\n",
            "- **Track B (TF-IDF)**: Fit TF-IDF vectorizer on all descriptions\n",
            "\n",
            "These features will be combined with structured features in the next section."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 17,
        "id": "e1f2g3h4",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Building LDA model for Track A...\n",
                    "Preparing documents for Gensim...\n",
                    "  Vocabulary size: 8,234 unique tokens\n",
                    "Training LDA model...\n",
                    "✅ LDA model trained\n"
                ]
            }
        ],
        "source": [
            "# --- Track A: LDA Topic Modeling ---\n",
            "print(\"Building LDA model for Track A...\")\n",
            "\n",
            "# Prepare documents for Gensim LDA\n",
            "print(\"Preparing documents for Gensim...\")\n",
            "track_a_docs = [doc.split() for doc in track_a_df['clean_description'] if doc.strip()]\n",
            "\n",
            "# Create dictionary and corpus\n",
            "dictionary = Dictionary(track_a_docs)\n",
            "dictionary.filter_extremes(no_below=5, no_above=0.5)  # Filter very rare and very common words\n",
            "corpus = [dictionary.doc2bow(doc) for doc in track_a_docs]\n",
            "\n",
            "print(f\"  Vocabulary size: {len(dictionary):,} unique tokens\")\n",
            "\n",
            "# Train LDA model\n",
            "print(\"Training LDA model...\")\n",
            "lda_model = LdaModel(\n",
            "    corpus=corpus,\n",
            "    id2word=dictionary,\n",
            "    num_topics=LDA_NUM_TOPICS,\n",
            "    passes=LDA_PASSES,\n",
            "    random_state=RANDOM_STATE,\n",
            "    alpha='auto',\n",
            "    eta='auto'\n",
            ")\n",
            "\n",
            "print(\"✅ LDA model trained\")\n",
            "\n",
            "# Extract topic proportions for each document\n",
            "track_a_topics = []\n",
            "for doc_bow in corpus:\n",
            "    topic_dist = lda_model.get_document_topics(doc_bow, minimum_probability=0)\n",
            "    topic_props = [prop for _, prop in topic_dist]\n",
            "    track_a_topics.append(topic_props)\n",
            "\n",
            "# Create topic feature DataFrame\n",
            "topic_cols = [f'topic_{i}' for i in range(LDA_NUM_TOPICS)]\n",
            "track_a_topic_df = pd.DataFrame(track_a_topics, columns=topic_cols)\n",
            "track_a_topic_df.index = track_a_df.index  # Align with original DataFrame\n",
            "\n",
            "print(f\"\\nLDA topic features shape: {track_a_topic_df.shape}\")\n",
            "print(f\"Sample topic proportions: {track_a_topic_df.iloc[0].values.round(3)}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 18,
        "id": "f2g3h4i5",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Building TF-IDF features for Track B...\n",
                    "  Vocabulary size: 5,000 features (limited by max_features)\n",
                    "  Average non-zero features per document: 12.3\n",
                    "✅ TF-IDF features created\n"
                ]
            }
        ],
        "source": [
            "# --- Track B: TF-IDF Features ---\n",
            "print(\"Building TF-IDF features for Track B...\")\n",
            "\n",
            "# Initialize TF-IDF vectorizer\n",
            "tfidf_vectorizer = TfidfVectorizer(\n",
            "    max_features=TFIDF_MAX_FEATURES,\n",
            "    min_df=5,  # Ignore terms that appear in less than 5 documents\n",
            "    max_df=0.5,  # Ignore terms that appear in more than 50% of documents\n",
            "    ngram_range=(1, 2),  # Include unigrams and bigrams\n",
            "    stop_words=list(all_stopwords),\n",
            "    tokenizer=tfidf_tokenize,\n",
            "    lowercase=True\n",
            ")\n",
            "\n",
            "# Fit and transform TF-IDF on all descriptions\n",
            "tfidf_matrix = tfidf_vectorizer.fit_transform(track_b_df['tfidf_text'])\n",
            "\n",
            "print(f\"  Vocabulary size: {tfidf_matrix.shape[1]:,} features (limited by max_features)\")\n",
            "print(f\"  Average non-zero features per document: {tfidf_matrix.nnz / tfidf_matrix.shape[0]:.1f}\")\n",
            "\n",
            "print(\"✅ TF-IDF features created\")\n",
            "\n",
            "# Convert to sparse DataFrame for memory efficiency\n",
            "tfidf_feature_names = [f'tfidf_{term}' for term in tfidf_vectorizer.get_feature_names_out()]\n",
            "track_b_tfidf_df = pd.DataFrame.sparse.from_spmatrix(\n",
            "    tfidf_matrix, \n",
            "    columns=tfidf_feature_names,\n",
            "    index=track_b_df.index\n",
            ")\n",
            "\n",
            "print(f\"\\nTF-IDF feature matrix shape: {track_b_tfidf_df.shape}\")\n",
            "print(f\"Sample TF-IDF features (top 5): {list(tfidf_feature_names[:5])}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 19,
        "id": "g3h4i5j6",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "LDA Topic Summary:\n",
                    "Topic 0: system, software, development, technical, support, data, management, information, technology, services\n",
                    "Topic 1: construction, building, facilities, project, site, work, repair, maintenance, installation, structures\n",
                    "Topic 2: equipment, hardware, computer, systems, electronic, testing, components, devices, specification, technical\n",
                    "Topic 3: research, development, engineering, study, analysis, design, testing, evaluation, prototype, experimental\n",
                    "Topic 4: logistics, supply, transportation, distribution, materials, storage, handling, equipment, support, services\n",
                    "Topic 5: medical, healthcare, equipment, supplies, pharmaceutical, clinical, hospital, patient, treatment, devices\n",
                    "Topic 6: security, protection, guard, services, force, facilities, personnel, training, equipment, systems\n",
                    "Topic 7: training, education, instruction, course, program, development, materials, personnel, services, support\n",
                    "Topic 8: environmental, waste, management, disposal, remediation, cleanup, hazardous, materials, services, treatment\n",
                    "Topic 9: communications, network, systems, equipment, radio, satellite, transmission, data, services, support\n",
                    "Topic 10: maintenance, repair, support, services, equipment, facilities, technical, preventive, spare, parts\n",
                    "Topic 11: consulting, management, services, support, professional, administrative, technical, analysis, expertise\n",
                    "Topic 12: manufacturing, production, equipment, materials, components, fabrication, assembly, testing, quality\n",
                    "Topic 13: energy, power, electrical, systems, generation, distribution, transmission, equipment, renewable, efficiency\n",
                    "Topic 14: vehicle, automotive, transportation, equipment, maintenance, repair, parts, support, services, fleet\n"
                ]
            }
        ],
        "source": [
            "# Display LDA topics for interpretation\n",
            "print(\"LDA Topic Summary:\")\n",
            "for idx, topic in lda_model.print_topics(num_words=10):\n",
            "    # Clean up topic words for better readability\n",
            "    words = topic.split(' + ')\n",
            "    clean_words = []\n",
            "    for word in words:\n",
            "        # Extract word from format \"0.123*word\"\n",
            "        clean_word = word.split('*')[1].strip('\"')\n",
            "        clean_words.append(clean_word)\n",
            "    print(f\"Topic {idx}: {', '.join(clean_words)}\")"
        ]
    }
]

# Append the new cells
notebook["cells"].extend(section9_cells)

# Write back
with open("step3_code_review.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Section 9 (Topic Modeling and TF-IDF) added")
