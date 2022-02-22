# Code Intelligence Papers

Code intelligence is a cross-research field combined with machine learning and software engineering. Since the large-scale pre-trained language models(PLMs) have achieved amazing improvement in the NLP area, researchers were trying to migrate PLMs from natural language to programming language. [GPT-3](https://openai.com/) shown the ability to automatic programming followed by human instructions([CodeX](https://openai.com/blog/openai-codex/)) and Github proposed the [Copilot](https://copilot.github.com/) tools to help developers, the Code intelligence area gradually emerging and reflecting the commercial value.

In this repository, I collect a series of papers on code intelligence, including surveys, sub-area papers, pre-trained models, metrics, datasets, etc. Most of those papers are published on the top conference of AI or SE and attached with opensource code or data. Hope this list can help you to do further research.

## Survey

- Allamanis M, Barr E T, Devanbu P, et al. [A Survey of Machine Learning for Big Code and Naturalness](https://dl.acm.org/doi/pdf/10.1145/3212695). ACM Computing Surveys (CSUR), 2018.

- Gros D, Sezhiyan H, Devanbu P, et al. [Code to Comment “Translation”: Data, Metrics, Baselining & Evaluation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9286030), ASE 2020.

- Liu C, Xia X, Lo D, et al. [Opportunities and Challenges in Code Search Tools](https://arxiv.org/pdf/2011.02297.pdf). ACM Computing Surveys (CSUR), 2021.

## Text to Code

### Code Search

- Gu X, Zhang H, Kim S. [Deep Code Search](https://www.researchgate.net/profile/Hongyu-Zhang-46/publication/325732005_Deep_code_search/links/5b29dcfb4585150c633faa57/Deep-code-search.pdf), ICSE 2018.

- Sachdev S, Li H, Luan S, et al. [Retrieval on Source Code: A Neural Code Search](https://research.fb.com/wp-content/uploads/2021/04/Retrieval-on-Source-Code-A-Neural-Code-Search.pdf), MAPL 2018.

- Luan S, Yang D, Barnaby C, et al. [Aroma: Code recommendation via structural code search](https://dl.acm.org/doi/pdf/10.1145/3360578). Proceedings of the ACM on Programming Languages, 2019.

- Cambronero J, Li H, Kim S, et al. [When Deep Learning Met Code Search](https://dl.acm.org/doi/pdf/10.1145/3338906.3340458), ESEC/FSE 2019.


### Natural Language to Code

#### Text to Code
- Yin P, Neubig G. [**TRANX**: A transition-based neural abstract syntax parser for semantic parsing and code generation](https://aclanthology.org/D18-2002.pdf), EMNLP 2018.

- Sun Z, Zhu Q, Xiong Y, et al. [TreeGen: A Tree-Based Transformer Architecture for Code Generation](https://ojs.aaai.org/index.php/AAAI/article/view/6430/6286), AAAI 2020.

#### Text to SQL

- Zhong V, Xiong C, Socher R. [Seq2SQL: Generating Structured Queries From Natural Language Using Reinforcement Learning](https://arxiv.org/pdf/1709.00103), 2017.

- Xu X, Liu C, Song D. [SQLNet: Generating Structured Queries From Natural Language Without Reinforcement Learning](https://openreview.net/pdf?id=SkYibHlRb). ICLR 2018.

- Yu T, Li Z, Zhang Z, et al. [TypeSQL: Knowledge-Based Type-Aware Neural Text-to-SQL Generation](https://www.aclweb.org/anthology/N18-2093.pdf), NAACL 2018.

## Code to code

### Code Translation(Migration)

- Nguyen A T, Nguyen T T, Nguyen T N. Lexical Statistical Machine Translation for Language Migration. ESEC/FSE 2013.

- Karaivanov S, Raychev V, Vechev M. [Phrase-Based Statistical Translation of Programming Languages](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.706.9697&rep=rep1&type=pdf). Onward! 2014.

- Nguyen A T, Nguyen T T, Nguyen T N. Divide-and-Conquer Approach for Multi-phase Statistical Migration for Source Code. ASE 2015.

- Chen X, Liu C, Song D. [Tree-to-tree Neural Networks for Program Translation](https://proceedings.neurips.cc/paper/2018/file/d759175de8ea5b1d9a2660e45554894f-Paper.pdf). NIPS 2018.

- Fu C, Chen H, Liu H, et al. [Coda: An End-to-End Neural Program Decompiler](https://proceedings.neurips.cc/paper/2019/file/093b60fd0557804c8ba0cbf1453da22f-Paper.pdf). NIPS 2019.

- Shiv V, Quirk C. [Novel positional encodings to enable tree-based transformers](https://proceedings.neurips.cc/paper/2019/file/6e0917469214d8fbd8c517dcdc6b8dcf-Paper.pdf). NIPS 2019.

- Weisz J D, Muller M, Houde S, et al. [Perfection Not Required? Human-AI Partnerships in Code Translation](https://arxiv.org/pdf/2104.03820.pdf). IUI 2021.

#### API Mapping

- Nguyen T D, Nguyen A T, Nguyen T N. [Mapping API Elements for Code Migration with Vector Representations](https://dl.acm.org/doi/pdf/10.1145/2889160.2892661), ICSE-C 2016.

- Gu X, Zhang H, Zhang D, et al. [DeepAM: Migrate APIs with Multi-modal Sequence to Sequence Learning](https://www.ijcai.org/proceedings/2017/0514.pdf). IJCAI 2017.

- Nguyen T D, Nguyen A T, Phan H D, et al. [Exploring API embedding for API usages and applications](https://www.researchgate.net/profile/Trong-Nguyen/publication/318576091_Exploring_API_Embedding_for_API_Usages_and_Applications/links/5bd1d34192851c9b86b88294/Exploring-API-Embedding-for-API-Usages-and-Applications.pdf). ICSE 2017.

- [Bui N D Q, Yu Y, Jiang L. SAR: learning cross-language API mappings with little knowledge](https://arxiv.org/pdf/1906.03835), ESEC/FSE 2019.

- Collie B, Ginsbach P, Woodruff J, et al. [M3: Semantic api migrations](https://arxiv.org/pdf/2008.12118.pdf), ASE 2020.

### Code Completion

- Svyatkovskiy A, Deng S K, Fu S, et al. [IntelliCode Compose: Code Generation Using Transformer](https://arxiv.org/pdf/2005.08025.pdf). ESEC/FSE 2020.

### Code Repair

### Code Clone Dectection

## Code to Text

- Alon U, Brody S, Levy O, et al. [code2seq: Generating Sequences from Structured Representations of Code](https://openreview.net/pdf?id=H1gKYo09tX), ICLR 2018.

### Code Summarization/Documentation

- Li, Jia, et al. [EditSum: A Retrieve-and-Edit Framework for Source Code Summarization](https://xin-xia.github.io/publication/ase213.pdf), ASE 2021.

- Junyan Cheng, Iordanis Fostiropoulos, Barry Boehm. [GN-Transformer: Fusing Sequence and Graph Representation for Improved Code Summarization](https://arxiv.org/pdf/2111.08874), 2021.

- Shi E, Wang Y, Du L, et al. [CAST: Enhancing Code Summarization with Hierarchical Splitting and Reconstruction of Abstract Syntax Trees](https://aclanthology.org/2021.emnlp-main.332.pdf), EMNLP 2021.

- LeClair A, Haque S, Wu L, et al. [Improved Code Summarization via a Graph Neural Network](https://arxiv.org/pdf/2004.02843), ICPC 2020.

- Ahmad W, Chakraborty S, Ray B, et al. [A Transformer-based Approach for Source Code Summarization](https://aclanthology.org/2020.acl-main.449.pdf), ACL 2020.

- Wan Y, Zhao Z, Yang M, et al. [Improving automatic source code summarization via deep reinforcement learning](https://arxiv.org/pdf/1811.07234), ASE 2018.

- Iyer S, Konstas I, Cheung A, et al. [Summarizing Source Code using a Neural Attention Model](https://aclanthology.org/P16-1195.pdf), ACL 2016.

## Code Representation and Pretrained Models

- Nguyen T T, Nguyen A T, Nguyen H A, et al. A Statistical Semantic Language Model for Source Code, ESEC/FSE 2013.

- Karampatsis R M, Babii H, Robbes R, et al. [Big Code != Big Vocabulary: Open-Vocabulary Models for Source Code](https://dl.acm.org/doi/pdf/10.1145/3377811.3380342) ICSE 2020.

- Karampatsis R M, Sutton C. [**SCELMo**: Source Code Embeddings from Language Models](https://arxiv.org/pdf/2004.13214.pdf), 2020.

- Kanade A, Maniatis P, Balakrishnan G, et al. **CuBERT:**[Learning and Evaluating Contextual Embedding of Source Code](http://proceedings.mlr.press/v119/kanade20a/kanade20a.pdf). ICML 2020.

- Feng Z, Guo D, Tang D, et al. [**CodeBERT**: A Pre-Trained Model for Programming and Natural Languages](https://aclanthology.org/2020.findings-emnlp.139.pdf). Findings of EMNLP 2020.

- Guo D, Ren S, Lu S, et al. [**GraphCodeBERT**: Pre-training Code Representations with Data Flow](), ICLR 2021.

- Chen M, Tworek J, Jun H, et al. **CodeX**: [Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374.pdf), 2021.

- Lachaux M A, Roziere B, Chanussot L, et al. **TransCoder**: [Unsupervised Translation of Programming Languages](https://papers.nips.cc/paper/2020/file/ed23fbf18c2cd35f8c7f8de44f85c08d-Paper.pdf), NeurIPS 2020.

- Wang Y, Wang W, Joty S, et al. [**CodeT5**: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://aclanthology.org/2021.emnlp-main.685.pdf), EMNLP 2021.

- Ahmad W U, Chakraborty S, Ray B, et al. **PLBART**: [Unified Pre-training for Program Understanding and Generation](https://aclanthology.org/2021.naacl-main.211.pdf), NAACL 2021.

- Roziere B, Lachaux M A, Szafraniec M, et al. [**DOBF**: A Deobfuscation Pre-Training Objective for Programming Languages](https://proceedings.neurips.cc/paper/2021/file/7d6548bdc0082aacc950ed35e91fcccb-Paper.pdf), NeurIPS 2021.

- Clement C, Drain D, Timcheck J, et al. [**PyMT5**: Multi-mode Translation of Natural Language and Python Code with Transformers](https://www.aclweb.org/anthology/2020.emnlp-main.728.pdf), EMNLP 2020.

- Jung T H. [**CommitBERT**: Commit Message Generation Using Pre-Trained Programming Language Model](https://arxiv.org/pdf/2105.14242), 2021.

- Peng D, Zheng S, Li Y, et al. **OSCAR**:[How could Neural Networks understand Programs?](https://arxiv.org/pdf/2105.04297.pdf). 2021.

- Qi W, Gong Y, Yan Y, et al. [**ProphetNet-X**: Large-Scale Pre-training Models for English, Chinese, Multi-lingual, Dialog, and Code Generation](https://arxiv.org/pdf/2104.08006). 2021.

- Wang X, Wang Y, Mi F, et al. [**SynCoBERT**: Syntax-Guided Multi-Modal Contrastive Pre-Training for Code Representation](https://arxiv.org/pdf/2108.04556). 2021.

## Metrics and Estimation

- Papineni K, Roukos S, Ward T, et al. [**BLEU**: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf), ACL 2002.

- Ren S, Guo D, Lu S, et al.   [**CodeBLEU**: a Method for Automatic Evaluation of Code Synthesis](https://arxiv.org/pdf/2009.10297.pdf), 2020.

- Tran N, Tran H, Nguyen S, et al. [Does BLEU Score Work for Code Migration?](https://arxiv.org/pdf/1906.04903), ICPC 2019.

- Agarwal M, Talamadupula K, Houde S, et al. [Quality Estimation & Interpretability for Code Translation](https://openreview.net/pdf?id=U7-z8CD2nYg), NeurIPS 2020 Workshop on Computer-Assisted Programming.

## Datasets and Benchmark

- Lu S, Guo D, Ren S, et al.
[**CodeXGLUE**: A Machine Learning Benchmark Dataset for Code Understanding and Generation](https://arxiv.org/pdf/2102.04664.pdf), 2021. 

- Ahmad W U, Tushar M G R, Chakraborty S, et al. [**AVATAR**: A Parallel Corpus for Java-Python Program Translation](https://arxiv.org/pdf/2108.11590), 2021.
- Puri R, Kung D S, Janssen G, et al. [**CodeNet**: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding Tasks](https://arxiv.org/pdf/2105.12655.pdf), NeurIPS 2021.

- LeClair A, McMillan C. [Recommendations for Datasets for Source Code Summarization](https://www.aclweb.org/anthology/N19-1394.pdf), NAACL 2019.

- Husain H, Wu H H, Gazit T, et al. [CodeSearchNet Challenge: Evaluating the State of Semantic Code Search](https://arxiv.org/pdf/1909.09436.pdf). 2019.

- Hu X, Li G, Xia X, et al. [Summarizing source code with transferred API knowledge](https://www.ijcai.org/Proceedings/2018/0314.pdf), IJCAI 2018.

- Miceli-Barone A V, Sennrich R. [A Parallel Corpus of Python Functions and Documentation Strings for Automated Code Documentation and Code Generation](https://www.aclweb.org/anthology/I17-2053.pdf), IJCNLP 2017.

- Yu T, Zhang R, Yang K, et al. [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://www.aclweb.org/anthology/D18-1425.pdf), EMNLP 2018.

- Lin X V, Wang C, Zettlemoyer L, et al. [NL2Bash: A Corpus and Semantic Parser for Natural Language Interface to the Linux Operating System](https://www.aclweb.org/anthology/L18-1491.pdf), LREC 2018.