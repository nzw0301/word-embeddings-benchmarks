# -*- coding: utf-8 -*-

import logging
from web.datasets.analogy import fetch_msr_analogy, fetch_google_analogy
from web.embeddings import fetch_GloVe
from web.evaluate import evaluate_analogy

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Fetch GloVe embedding (warning: it might take few minutes)
w_glove = fetch_GloVe(corpus="wiki-6B", dim=300)

# Define tasks
analogy_tasks = {
    "google": fetch_google_analogy(),
    "msr": fetch_msr_analogy()
}

for name, data in analogy_tasks.items():
    analogy_df = evaluate_analogy(w_glove, data.X, data.y,
                                  category=data.category, method="mul",
                                  batch_size=500)

