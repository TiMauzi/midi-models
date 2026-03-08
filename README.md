# 🎵 MIDI-to-Year Prediction
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML: Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

This project features a machine learning pipeline that predicts the **composition year** of a musical work based solely on features extracted from MIDI files.

## 👀 Overview
The notebook demonstrates how to transform raw MIDI data into meaningful features to train a regression model. The workflow includes:

* **Data Acquisition**: Loading the [TiMauzi/imslp-midi-by-sa](https://huggingface.co/datasets/TiMauzi/imslp-midi-by-sa/) dataset from the Hugging Face Hub.
* **Feature Engineering**: Extracting the top 15 most common MIDI message types (e.g., `note_on`, `control_change`, `set_tempo`) and calculating their frequency per piece.
* **Model Training**: Implementing a `GradientBoostingRegressor` to handle the non-linear relationships in musical structure across eras.
* **Evaluation**: Testing the model's precision using Mean Absolute Error (MAE) and accuracy windows (e.g., how often the model estimates within the decade or century of the actual date).

## 📚 MIDI Dataset Foundations

Machine learning models are tested on several tasks nowadays. Especially in the field of NLP, there are numerous benchmarks for measuring the quality of language models. In the area of computer vision, there are multiple challenges which are applied to compare systems concerning multiple tasks.

In the area of sound-related studies, only little work has been done. Indeed, the area of speech recognition is of high interest. Moreover, music generation models have also been developed, reviewed and refined. However, most NLP models are trained and compared on plain text corpora.

When it comes to music and audio data in general, usually formats such as MP3 are considered. While these often contain high-quality data, they are also noisy by nature. Even high-quality recordings of music contain all different kinds of additional information—this includes sounds from the audience (coughing, whispering), but also specific sounds from the artists (slightly mistuned instruments, different pitches of singers etc.).

These issues may prove troublesome if one wants to compare musical data independently from their temporal or spatial attributes, i.e. the place and situation a specific piece has been recorded in. Moreover, an artist’s unique interpretation also imbues the piece with a distinctive cultural and contemporary essence.

Musical Instrument Digital Interface (MIDI) is a type of musical representation that has been first introduced in 1982. While it is still used for synthesizers and keyboards, for recordings other digital formats such as MP3 are applied, since they provide a low-loss storage. MIDI data on the other hand only contains signals and raw information on which notes are to be played sequentially.

In a way, MIDI embodies a digitalized, audibly presentable variant of sheet music, which is independent of any recording’s spatiotemporal context. Therefore, a collection of MIDI data provides a clean, standardized and valuable resource for evaluating the ability of machine learning models to analyze musical data.

This project approaches that problem by having constructed datasets based on MIDI files retrieved from IMSLP and pairing them with structured metadata.

### 🎼 IMSLP

One of the largest collections of MIDI data is the **[International Music Score Library Project (IMSLP)](https://imslp.org/)**, also known as the Petrucci Music Library.

This wiki-based digital library contains hundreds of thousands of musical pieces, providing both sheet music and recordings from the public domain or open source. At the time of writing, it also comprises more than **25,000 MIDI files**, which can often be mapped to:

- composer
- year of composition
- historical era
- stylistic era
- musical key

For the datasets constructed for this project, IMSLP is particularly well-suited for analyzing historical musical pieces.

To retrieve the data, a crawling script collected the following information for each MIDI file:

- title  
- composer  
- year of composition  
- era of time  
- era of style  
- key  
- license  

### ℹ️ Metadata

This work proposes an exemplary use of the metadata fields **composer, era, style, year, and key**. However, additional metadata fields such as titles may be useful for future research tasks (e.g., title generation based on musical information).

All in all, the final dataset consists of the following fields:

- `midi_source`: URL to the original MIDI file on IMSLP (incl. original uploader),
- `metadata_source`: URL to the original metadata on IMSLP,
- `file_name`, `title`, `composer`, `year`, `era`, `style`, `key`, `license`: Metadata fields as described above,
- `midi`: Raw MIDI bytes,
- `midi_mido`: JSON-serialized mido object.

Most attributes can be retrieved from the *work information* section of a musical piece’s IMSLP page. However, the metadata is not always consistently formatted.

#### 📆 Year Normalization

The year of composition, for example, while mostly written as a single year, is sometimes written as an interval and, in some cases, supplemented with additional notes. For the piece ‘Te Deum, H.146’ by Marc-Antoine Charpentier, the information ‘1688-98 ca.?’ is specified. Since elaborate rule-based approaches to solve similar issues are infeasible, only the first number found in the corresponding row is considered. In the given case, only ‘1688’ is added to the dataset.

#### #️⃣ Key Normalization

Similarly, the key of a musical piece is oftentimes expressed in varying ways. This issue arises not least because there are multiple notations applied in different regions of the world. ‘C major’ may be called simply ‘C’, as opposed to ‘c’ for ‘C minor’. Furthermore, keys featuring a flat ($\flat$) or a sharp ($\sharp$) are normalized from, e.g., ‘G$\sharp$ major’ and ‘G\# major’ or ‘G$\flat$ minor’ and ‘Gb minor’—note the different characters for the sharp and flat symbols—to ‘G-sharp major’ or ‘G-flat minor’, respectively.

### 🧐 Limitations and Ethical Considerations

The dataset itself has a strong bias towards western music. Only very few pieces have been labeled as ‘Non-Western Classical’ so that models trained on any of the dataset variants will very likely be inadequate to predict or analyze any information on non-western music. Furthermore, the distribution of musical pieces per era is unequal even for western music: while works from the Romantic and Baroque eras are plentiful, other eras are scarcer. Future developments of MIDI dataset should care for a more equally distributed set of files for each class.

The models proposed within this project apply very basic architectures. Future work should consider more elaborate design choices in order to yield better results. Moreover, other data fields may be predicted, e.g., years, by using regression techniques.

The crawling of the datasets has been approved and supported by the CEO of IMSLP in 2024, [Edward Guo](https://www.linkedin.com/in/edward-guo-696bb1207/). However, due to the licensing restrictions, not all usages are permitted. This has been considered by splitting the datasets into three variants to value each MIDI files respective originally designated use, i.e., license.

### 🎩 Acknowledgements
This repository was created as part of a private project by the author and had no further funding or third-party support. Many thanks to [Edward Guo](https://www.linkedin.com/in/edward-guo-696bb1207/) for supporting the compilation of the IMSLP data as the administrator of the IMSLP website.

Dataset variants are available at HuggingFace: 
- [TiMauzi/imslp-midi-cc0-1.0](https://huggingface.co/datasets/TiMauzi/imslp-midi-cc0-1.0), 
- [TiMauzi/imslp-midi-by-nc-sa](https://huggingface.co/datasets/TiMauzi/imslp-midi-by-nc-sa), 
- [TiMauzi/imslp-midi-by-sa](https://huggingface.co/datasets/TiMauzi/imslp-midi-by-sa). 

## 📊 Results Summary
The simple regression model already shows a strong ability to identify the general era of a piece, even without access to the actual audio or sheet music.

| Metric | Result |
| :--- | :--- |
| **Mean Absolute Error (MAE)** | ~42.17 years |
| **Root Mean Squared Error (RMSE)** | ~58.74 years |
| **±25-Year Accuracy** | 34.4% |
| **±50-Year Accuracy** | 59.1% |

While predicting a specific year is challenging, the model correctly places nearly **60% of pieces within a one-century window**, proving that MIDI message distributions can be crucial indicators for musical analysis.
