Named Entity Recognition (NER) with BiLSTM and BERT
This project utilizes the CoNLL-2003 dataset for Named Entity Recognition (NER). The dataset consists of three subsets:

train (eng.train): The training dataset used to train the NER model.
testa (eng.testa): The validation dataset used for hyperparameter tuning and model selection.
testb (eng.testb): The test dataset for evaluating final model performance.
The dataset follows the BIO (Begin, Inside, Outside) format, where each token is labeled with an entity type, including:

PER (Person)
ORG (Organization)
LOC (Location)
MISC (Miscellaneous)
Data Preprocessing
Tokenization: The dataset is tokenized, and words are mapped to unique indices.
Padding and Sequence Conversion: Sentences are padded to a fixed length to handle variable-sized sequences efficiently.
Word and Tag Indexing: Unknown words (<UNK>) are handled gracefully, and tag indices are mapped for model compatibility.
Batch Processing: The data is converted into PyTorch datasets and processed using DataLoader with custom collation to handle different sequence lengths.
 
This repository contains an implementation of Named Entity Recognition (NER) using two different approaches:

BiLSTM-based model: A traditional deep learning approach using Bidirectional LSTM (BiLSTM).
BERT-based model: A transformer-based method leveraging the pre-trained BERT model.
Features:
Data Preprocessing: Reads and processes NER datasets in BIO format.
BiLSTM Model: Implements a BiLSTM-based NER model with optimized hyperparameters.
BERT-based Model: Utilizes BertForTokenClassification for advanced NER tasks.
Training and Evaluation: Both models are trained and evaluated using PyTorch and the transformers library.
Hyperparameter Tuning: A grid search is performed to find the best model parameters.
NER Inference: The trained BERT model can predict named entities from input text.


Dataset
The dataset used follows the BIO format and is preprocessed into token-label pairs. The model supports multiple entity types, including:

PER (Person)
ORG (Organization)
LOC (Location)
MISC (Miscellaneous)
Results
After training, the best hyperparameters for the BiLSTM model are selected through validation set(testa) on an average of 95% is received through bilstm, and BERT is fine-tuned for optimal performance. The final accuracy on the test sets is displayed at the end of training. by BERT we have recieved accuracy of 98 %

Model Saving
The trained BERT model and tokenizer are saved for future inference:


bert_ner_model/
Future Improvements
Support for additional languages.
Experimentation with other transformer models like RoBERTa or DeBERTa.
Integration with an API for real-time NER.
