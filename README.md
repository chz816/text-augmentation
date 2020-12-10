# Data Augmentation for Natural Language Processing

## Approach

### Synonym Replacement

- **Definition:** randomly choose _n_ words from the sentence that are not stop words. Replace each of these words with one of its synonyms chosen at random
- **From**: Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks." *arXiv preprint arXiv:1901.11196* (2019).

### Random Insertion

- **Definition:** Find a random synonym of a random word in the sentence that is not a stop word. Insert that synonym into a random position in the sentence. Do this _n_ times.
- **From**: Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks." *arXiv preprint arXiv:1901.11196* (2019).

### Random Swap

- **Definition:** randomly choose two words in the sentence and swap their positions. Do this _n_ times.

- **From**: Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks." *arXiv preprint arXiv:1901.11196* (2019).

### Random Deletion

- **Definition:** randomly remove each word in the sentence with probability _p_.

- **From**: Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks." *arXiv preprint arXiv:1901.11196* (2019).

### wordMixup and senMixup

- It applies Mixup, which is a data augmentation method, on Natural Language Processing. Targeting the sentence classification task, one performs interpolation on word embeddings (**wordMixup**) and another on sentence embeddings (**senMixup**).
  - **wordMixup:** given a piece of text, such as a sentence with _N_ words, it can be represented as a matrix _B_. Each row _t_ of the matrix correspond to one word. Given a pair of sentences, we have (Bi, Yi) and (Bj, Yj), where B is the matrix for each sentence, and Y here is the label associated with its sentence. We can generated a new instance (Bnew, Ynew): find the t-th word in the sentence, we can have the t-th row of Bnew is the weighted sum of the t-th word embedding for sentence _i_ and sentence _j_. Similarly Ynew is also the weighted sum of the labels for two sentences.
  - **senMixup:** here instead of using _B_ as the representation for the sentence, we introduce _f_ as the sentence encoder. So _f(B)_ is the representation for the sentence. So a new instance (Bnew, Ynew) is constructed as: each k-th dimension of Bnew is the weighted sum of the sentence representation _f(Bi)_ and _f(Bj)_.
- **From:** Guo, Hongyu, Yongyi Mao, and Richong Zhang. "Augmenting data with mixup for sentence classification: An empirical study." *arXiv preprint arXiv:1905.08941* (2019).