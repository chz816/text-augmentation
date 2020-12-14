# Data Augmentation in Natural Language Processing (NLP)

Data augmentation is a useful approach to enhance the performance of the deep learning model. It generates new data instances from the existing training data, with the objective of improving the performance of the downstream model. This approach has achieved much success in the computer vision area. Recently text data augmentation has been extensively studied in natural language processing. In this repo, I summarize the common data augmentation approaches used in Natural Language Processing (NLP).

## Why Perform Data Augmentation?

- Improve the model performance by generating more data
- Reduce the degree of class imbalance

## Text Data Augmentation

In this section, I list the text data augmentation collected from the literature. For each approach, I provide a brief definition of methodology, and the literature described this approach. All these methods can be categorized into **unsupervised** and **supervised**, based on if they need the label associated with the data instance.

### Unsupervised Approach

#### Word-Level Transformation

Produce new sentences while preserving the semantic features of the original texts to a certain extent

##### Synonym Replacement

- **Definition:** randomly choose *n* words from the sentence and replace each of these words with its synonyms.

- **Literature**: 

- - Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks." *arXiv preprint arXiv:1901.11196* (2019).
  - Zhang, Xiang, Junbo Zhao, and Yann LeCun. "Character-level convolutional networks for text classification." *Advances in neural information processing systems*. 2015.
  - Wang, William Yang, and Diyi Yang. "That’s so annoying!!!: A lexical and frame-semantic embedding based data augmentation approach to automatic categorization of annoying behaviors using# petpeeve tweets." *Proceedings of the 2015 conference on empirical methods in natural language processing*. 2015.
  - Shim, Heereen, et al. "Data augmentation and semi-supervised learning for deep neural networks-based text classifier." *Proceedings of the 35th Annual ACM Symposium on Applied Computing*. 2020.
  - Kolomiyets, Oleksandr, Steven Bethard, and Marie-Francine Moens. "Model-portability experiments for textual temporal analysis." *Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies*. Vol. 2. ACL; East Stroudsburg, PA, 2011.
  - Rizos, Georgios, Konstantin Hemker, and Björn Schuller. "Augment to prevent: short-text data augmentation in deep learning for hate-speech classification." *Proceedings of the 28th ACM International Conference on Information and Knowledge Management*. 2019.
  - Mueller, Jonas, and Aditya Thyagarajan. "Siamese recurrent architectures for learning sentence similarity." *thirtieth AAAI conference on artificial intelligence*. 2016.

##### Uninformative Word Replacement

- **Definition:** generate an augmented instance that replaces uninformative words with low TF-IDF scores while keeping those with high TF-IDF values. This augmentation is designed to retain keywords and replace uninformative words with other uninformative words.

- **Literature:**

- - Xie, Qizhe, et al. "Unsupervised data augmentation for consistency training." *arXiv preprint arXiv:1904.12848* (2019).

##### Random Insertion

- **Definition:** find a random synonym of a random word in the sentence that is not a stop word. Insert that synonym into a random position in the sentence. Do this *n* times.

- **Literatures**: 

- - Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks." *arXiv preprint arXiv:1901.11196* (2019).
  - Shim, Heereen, et al. "Data augmentation and semi-supervised learning for deep neural networks-based text classifier." *Proceedings of the 35th Annual ACM Symposium on Applied Computing*. 2020.

##### Random Swap

- **Definition:** randomly choose two words in the sentence and swap their positions. Do this *n* times.

- **Literature**: 

- - Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks." *arXiv preprint arXiv:1901.11196* (2019).
  - Shim, Heereen, et al. "Data augmentation and semi-supervised learning for deep neural networks-based text classifier." *Proceedings of the 35th Annual ACM Symposium on Applied Computing*. 2020.

##### Random Deletion

- **Definition:** randomly remove each word in the sentence with probability *p*.

- **Literatures**: 

- - Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks." *arXiv preprint arXiv:1901.11196* (2019).
  - Shim, Heereen, et al. "Data augmentation and semi-supervised learning for deep neural networks-based text classifier." *Proceedings of the 35th Annual ACM Symposium on Applied Computing*. 2020.

##### Spelling Errors Injection

- **Definition:** generate texts containing common misspellings in order to train the models which are expected to buy more robust to this type of textual noise

- **Literature:**

- - Coulombe, Claude. "Text data augmentation made simple by leveraging NLP cloud APIs." *arXiv preprint arXiv:1812.04718*(2018).

##### Word Position Augmentation

- **Definition:** by shifting and warping the sentence within the confines of the padded sequence, we can receive meaningfully perturbed versions of the original samples with the same semantics.

- **Literature:**

- - Rizos, Georgios, Konstantin Hemker, and Björn Schuller. "Augment to prevent: short-text data augmentation in deep learning for hate-speech classification." *Proceedings of the 28th ACM International Conference on Information and Knowledge Management*. 2019.

##### Context-Aware Augmentation

- **Definition:** instead of the pre-defined synonyms, use words that are predicted by a language model given the context surrounding the original words to be augmented

- **Literature:**

- - Kobayashi, Sosuke. "Contextual augmentation: Data augmentation by words with paradigmatic relations." *arXiv preprint arXiv:1805.06201* (2018).

##### Translation Data Augmentation

- **Definition:** the intuition is to focus on a subset of the vocabulary that we know to be poorly modeled by the baseline system, namely words that occur rarely in the parallel corpus. The goal is to provide novel contexts for rare words. To achieve this, this method searches for contexts where a common word can be replaced by rare words. The targeted rare word is defined by looking for the *v* most common words observed in the training corpus that has fewer than *R* occurrences. If the language model suggests a rare substitution in a particular context, we replace the word and add the new sentence to the training data. The replaced word is learned and predicted by the language model.

- **Literature:**

- - Fadaee, Marzieh, Arianna Bisazza, and Christof Monz. "Data augmentation for low-resource neural machine translation." *arXiv preprint arXiv:1705.00440* (2017).

#### Neural Text Generation

##### Two-Way Translation

- **Definition:** translate the input document into other "pivot" languages and then back to the target language. This method helps to introduce lexical and syntactical variation to input documents.

- **Literature:**

- - Luque, Franco M. "Atalaya at tass 2019: Data augmentation and robust embeddings for sentiment analysis." *arXiv preprint arXiv:1909.11241* (2019).
  - Yu, Adams Wei, et al. "Qanet: Combining local convolution with global self-attention for reading comprehension." *arXiv preprint arXiv:1804.09541* (2018).
  - Sennrich, Rico, Barry Haddow, and Alexandra Birch. "Improving neural machine translation models with monolingual data." *arXiv preprint arXiv:1511.06709* (2015).

##### SentenceVAE

- **Definition:** Unconditional VAE + prior sampling

- **Literature**:

- - Xu, Binxia, et al. "Data Augmentation for Multiclass Utterance Classification–A Systematic Study." *Proceedings of the 28th International Conference on Computational Linguistics*. 2020.

### Supervised Approach 

#### Random Resampling

- **Definition:**

- - *Random Undersampling* refers to all classes have the same number of instances as the smallest class
  - *Random Oversampling* refers to all classes have the same number of instances as the largest class.
  - *Random Duplication* is a combination of undersampling and oversampling, which drops some data samples in the majority classes while duplicates data samples in the minority classes.

- **Literature:**

- - Qiu, Siyuan, et al. "EasyAug: An automatic textual data augmentation platform for classification tasks." *Companion Proceedings of the Web Conference 2020*. 2020.

#### Word-Level Transformation

Produce new sentences while preserving the semantic features of the original texts to a certain extent

##### Instance Crossover

- **Definition:** split tokenized input documents into two halves, and then randomly sample and combine first halves with second halves. (*Note: the label for the generated instance needs to be classified again.*) This method is expected to introduce new points in the "spaces" between the original ones.

- **Literature**:

- - Luque, Franco M. "Atalaya at tass 2019: Data augmentation and robust embeddings for sentiment analysis." *arXiv preprint arXiv:1909.11241* (2019).

##### wordMixup and senMixup

- **Definition:** It applies Mixup, which is a data augmentation method, on Natural Language Processing. Targeting the sentence classification task, one performs interpolation on word embeddings (**wordMixup**) and another on sentence embeddings (**senMixup**).

- - **wordMixup:** given a piece of text, such as a sentence with *N* words, it can be represented as a matrix *B*. Each row *t* of the matrix corresponds to one word. Given a pair of sentences, we have (Bi, Yi) and (Bj, Yj), where B is the matrix for each sentence, and Y here is the label associated with its sentence. We can generate a new instance (Bnew, Ynew): find the t-th word in the sentence, we can have the t-th row of Bnew is the weighted sum of the t-th word embedding for sentence *i* and sentence *j*. Similarly, Ynew is also the weighted sum of the labels for two sentences.
  - **senMixup:** here instead of using *B* as the representation for the sentence, we introduce *f* as the sentence encoder. So *f(B)* is the representation for the sentence. So a new instance (Bnew, Ynew) is constructed as: each k-th dimension of Bnew is the weighted sum of the sentence representation *f(Bi)* and *f(Bj)*.

- **Literatures:** 

- - Guo, Hongyu, Yongyi Mao, and Richong Zhang. "Augmenting data with mixup for sentence classification: An empirical study." *arXiv preprint arXiv:1905.08941* (2019).

##### Context-Aware Augmentation - Conditional Constraint

- **Definition:** context-aware augmentation is not always compatible with annotated labels, which may generate sentences that are implausible with respect to their original labels. A conditional constraint is introduced to control the replacement of words to prevent the generated words from reversing the information related to the labels of the sentences.

- **Literature:**

- - Kobayashi, Sosuke. "Contextual augmentation: Data augmentation by words with paradigmatic relations." *arXiv preprint arXiv:1805.06201* (2018).

#### Neural Text Generation

##### conditional BERT Contextual Augmentation

- **Definition:** apply BERT to contextual augmentation for labeled sentences, by offering a wider range of substitute words predicted by the masked language model task. Introduce a new fine-tuning objective that randomly masks some tokens from the input and the objective is to predict a label-compatible word based on both its context and sentence label. Given a trained conditional BERT and a labeled sentence, we randomly mask a few words in the sentence which are replaced by the predicted words from conditional BERT.

- **Literature:** 

- - Wu, Xing, et al. "Conditional BERT contextual augmentation." *International Conference on Computational Science*. Springer, Cham, 2019.
  - Jiao, Xiaoqi, et al. "Tinybert: Distilling bert for natural language understanding." *arXiv preprint arXiv:1909.10351* (2019).

##### Data Augmentation as Seq2Seq Generation

- **Definition:** use the standard seq2seq model to generate the alternative delexicalised utterance. The basic assumption is that if *d* and *d'* contain the same semantic frames, they can be generated from each other.

- **Literature:**

- - Hou, Yutai, et al. "Sequence-to-sequence data augmentation for dialogue language understanding." *arXiv preprint arXiv:1807.01554* (2018).

##### Language-Model-Based Data Augmentation

- **Definition:** fine-tune the language model (LM) using the provided dataset and synthesize labeled data. For different types of language models, including an auto-regressive LM like GPT-2, an autoencoder LM like BERT, and a pre-trained seq2seq LM like BART, a similar idea can be applied to these LMs for data augmentation.

- **Literature:**

- - Anaby-Tavor, Ateret, et al. "Do Not Have Enough Data? Deep Learning to the Rescue!." *AAAI*. 2020.
  - Kumar, Varun, Ashutosh Choudhary, and Eunah Cho. "Data augmentation using pre-trained transformer models." *arXiv preprint arXiv:2003.02245* (2020).

##### CVAE and CVAE-posterior (CVAE-p)

- **Definition:** conditional VAE + prior sampling/posterior sampling

- **Literature:**

- - Xu, Binxia, et al. "Data Augmentation for Multiclass Utterance Classification–A Systematic Study." *Proceedings of the 28th International Conference on Computational Linguistics*. 2020.

## Adversarial Examples in Natural Language Processing

Researchers have made several contributions in applying Generative Adversarial Networks (GANs) in natural language processing. In this type of research, generating adversarial examples is a key step. It is motivated by two goals: attack and defense. The attack aims to examine the robustness of the target model and the defense takes a step further utilizing generated adversarial examples to robustify the target model. [This paper](https://dl.acm.org/doi/pdf/10.1145/3374217) provides a very comprehensive survey for related articles.

The key difference between text data augmentation and generating adversarial examples is: data augmentation aims to generate a new instance without changing its label. But an adversarial example should have different labels. 

### Supervised Approach

#### BAE: BERT-based Adversarial Examples

- **Definition:** given a dataset (*S*,*Y*) and a trained classification model *C*, where *S* is the text and *Y* is the label associated with the instance, we want to generate an adversarial example *S(adv)* such that *C(S(adv))* doesn't equal to *Y*. In addition, we expect *S(adv)* is grammatically correct and semantically similar to *S*. BAE generates new data instances by **replacing** or **inserting** tokens in the sentence, which is similar to the unsupervised approach. But it calculates the **token importance** for each token by deleting this word from *S* and compute the decrease in the probability of predicting the correct label. It also uses the label information to choose the generated sample.

- - **Replace:** replace the selected token with a mask token and use pre-trained BERT to predict it
  - **Insert:** insert a mask token and use BERT to predict the word

- **Literature:**

- - Garg, Siddhant, and Goutham Ramakrishnan. "BAE: BERT-based Adversarial Examples for Text Classification." *arXiv preprint arXiv:2004.01970* (2020).