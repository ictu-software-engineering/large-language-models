# 🗣️ Mô hình ngôn ngữ lớn (Large Language Model - LLM)

Khóa học này được chia thành 02 phần:

1. 🧩 **Cơ bản về LLM** bao gồm các kiến thức về toán học , Python, và mạng nơ ron.
2. 🧑‍🔬 **Nhà nghiên cứu LLM** tập trung vào học cách làm thế nào để xây dựng kiến trúc LLMs tốt nhất bằng cách sử dụng kỹ thuật mới nhất. 

## 📝 Notebooks

Danh sách các code mẫu và bài viết liên quan đến LLM 

### Tinh chỉnh (Fine-tuning)

| Notebook | Mô tả | Bài viết | Notebook |
|---------------------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| Tinh chỉnh Llama 2 trong Google Colab | Hướng dẫn từng bước để tinh chỉnh mô hình Llama 2. | [Article](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html) | <a href="https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing"><img src="images/colab.svg" alt="Open In Colab"></a> |
| Tinh chỉnh LLMs với Axolotl | Hướng dẫn chi tiết về công cụ tinh chỉnh hiện đại. | [Article](https://mlabonne.github.io/blog/posts/A_Beginners_Guide_to_LLM_Finetuning.html) | W.I.P. |
| Tinh chỉnh mô hình Mistral-7b model với DPO | Tăng cường hiệu suất của các mô hình tinh chỉnh được giám sát với DPO. | [Tweet](https://twitter.com/maximelabonne/status/1729936514107290022) | <a href="https://colab.research.google.com/drive/15iFBr1xWgztXvhrj5I9fBv20c7CFOPBE?usp=sharing"><img src="images/colab.svg" alt="Open In Colab"></a> |

### Lượng tử hóa (Quantization)

| Notebook | Mô tả | Bài viết | Notebook |
|---------------------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. Giới thiệu về Lượng tử hóa trọng số | Tối ưu hóa mô hình ngôn ngữ lớn bằng cách sử dụng lượng tử hóa 8 bit. | [Article](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html) | <a href="https://colab.research.google.com/drive/1DPr4mUQ92Cc-xf4GgAaB6dFcFnWIvqYi?usp=sharing"><img src="images/colab.svg" alt="Open In Colab"></a> |
| 2. Lượng tử hóa LLM 4 bit bằng GPTQ | Lượng tử hóa mô hình LLM của bạn và chạy trên máy người dùng. | [Article](https://mlabonne.github.io/blog/4bit_quantization/) | <a href="https://colab.research.google.com/drive/1lSvVDaRgqQp_mWK_jC9gydz6_-y6Aq4A?usp=sharing"><img src="images/colab.svg" alt="Open In Colab"></a> |
| 3. Lượng tử mô hình Llama 2 với GGUF và llama.cpp | Lượng tử mô hình Llama 2 với llama.cpp và tải GGUF lên HF Hub. | [Article](https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html) | <a href="https://colab.research.google.com/drive/1pL8k7m04mgE5jo2NrjGi8atB0j_37aDD?usp=sharing"><img src="images/colab.svg" alt="Open In Colab"></a> |
| 4. ExLlamaV2: Thư viện nhanh nhất để chạy LLM | Lượng tử hóa và chạy mô hình EXL2 và tải chúng lên HF Hub. | [Article](https://mlabonne.github.io/blog/posts/ExLlamaV2_The_Fastest_Library_to_Run%C2%A0LLMs.html) | <a href="https://colab.research.google.com/drive/1yrq4XBlxiA0fALtMoT2dwiACVc77PHou?usp=sharing"><img src="images/colab.svg" alt="Open In Colab"></a> |

### Khác

| Notebook | Mô tả | Bài viết | Notebook |
|---------------------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| Hợp nhất LLM với Mergekit | Kết hợp nhiều LLM và tạo mô hình Frankenstein của riêng bạn | [Tweet](https://twitter.com/maximelabonne/status/1740732104554807676) | <a href="https://colab.research.google.com/drive/1_JS7JKJAQozD48-LhYdegcuuZ2ddgXfr?usp=sharing"><img src="images/colab.svg" alt="Open In Colab"></a> |
| Chiến lược giải mã trong các mô hình ngôn ngữ lớn | Hướng dẫn tạo văn bản từ tìm kiếm chùm tia đến lấy mẫu hạt nhân | [Article](https://mlabonne.github.io/blog/posts/2022-06-07-Decoding_strategies.html) | <a href="https://colab.research.google.com/drive/19CJlOS5lI29g-B3dziNn93Enez1yiHk2?usp=sharing"><img src="images/colab.svg" alt="Open In Colab"></a> |
| Trực quan hóa hàm mất mát của GPT-2 | Sơ đồ 3D trực quan hàm mất mát dựa trên sự nhiễu loạn trọng số. | [Tweet](https://twitter.com/maximelabonne/status/1667618081844219904) | <a href="https://colab.research.google.com/drive/1Fu1jikJzFxnSPzR_V2JJyDVWWJNXssaL?usp=sharing"><img src="images/colab.svg" alt="Open In Colab"></a> |
| Cải thiện ChatGPT bằng Sơ đồ tri thức | Tăng cường câu trả lời của ChatGPT bằng biểu đồ tri thức. | [Article](https://mlabonne.github.io/blog/posts/Article_Improve_ChatGPT_with_Knowledge_Graphs.html) | <a href="https://colab.research.google.com/drive/1mwhOSw9Y9bgEaIFKT4CLi0n18pXRM4cj?usp=sharing"><img src="images/colab.svg" alt="Open In Colab"></a> |

## 🧩 Cơ bản về LLM

![](images/roadmap_fundamentals.png)

### 1. Toán học cho học máy

Trước khi thành thạo machine learning, điều quan trọng là phải hiểu các khái niệm toán học cơ bản hỗ trợ các thuật toán này.

- **Đại số tuyến tính (Linear Algebra)**: Môn này rất quan trọng để hiểu nhiều thuật toán, đặc biệt là những thuật toán được sử dụng trong học sâu. Các khái niệm chính bao gồm vectơ, ma trận, định thức, giá trị riêng và vectơ riêng, không gian vectơ và các phép biến đổi tuyến tính.
- **Giải tích (Calculus)**: Nhiều thuật toán học máy liên quan đến việc tối ưu hóa các hàm liên tục, đòi hỏi sự hiểu biết về đạo hàm, tích phân, giới hạn và chuỗi. Phép tính đa biến và khái niệm gradient cũng rất quan trọng.
- **Xác suất và Thống kê**: Đây là những thông tin quan trọng để hiểu cách các mô hình học hỏi từ dữ liệu và đưa ra dự đoán. Các khái niệm chính bao gồm lý thuyết xác suất, biến ngẫu nhiên, phân bố xác suất, kỳ vọng, phương sai, hiệp phương sai, tương quan, kiểm tra giả thuyết, khoảng tin cậy, ước tính khả năng tối đa và suy luận Bayes.

📚 Tài nguyên học tập:

- [3Blue1Brown - The Essence of Linear Algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab): Chuỗi video cung cấp trực quan hình học cho các khái niệm này.
- [StatQuest with Josh Starmer - Statistics Fundamentals](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9): Cung cấp những giải thích đơn giản và rõ ràng cho nhiều khái niệm thống kê.
- [Immersive Linear Algebra](https://immersivemath.com/ila/learnmore.html): Một cách giải thích trực quan khác về đại số tuyến tính.
- [Khan Academy - Linear Algebra](https://www.khanacademy.org/math/linear-algebra): Tốt cho người mới bắt đầu vì nó giải thích các khái niệm một cách rất trực quan.
- [Khan Academy - Calculus](https://www.khanacademy.org/math/calculus-1): Khóa học tương tác bao gồm tất cả những kiến thức cơ bản về tính toán.
- [Khan Academy - Probability and Statistics](https://www.khanacademy.org/math/statistics-probability): Tài liệu học tập dễ hiểu. 
---

### 2. Python cho học máy (Machine Learning)

Python là ngôn ngữ lập trình mạnh mẽ và linh hoạt, đặc biệt tốt cho machine learning nhờ tính dễ đọc, tính nhất quán và hệ sinh thái mạnh mẽ của các thư viện khoa học dữ liệu.

- **Khái niệm cơ bản về Python**: Việc hiểu biết về cú pháp cơ bản, kiểu dữ liệu, xử lý lỗi và lập trình hướng đối tượng của Python là rất quan trọng.
- **Thư viện khoa học dữ liệu**: Bắt buộc phải làm quen với NumPy cho các phép toán số, Pandas để thao tác và phân tích dữ liệu, Matplotlib và Seaborn để trực quan hóa dữ liệu.
- **Tiền xử lý dữ liệu**: Quá trình này bao gồm việc chia tỷ lệ và chuẩn hóa thuộc tính, xử lý dữ liệu bị thiếu, phát hiện ngoại lệ, mã hóa dữ liệu theo phân loại và chia dữ liệu thành các tập huấn luyện, xác thực và kiểm tra.
- **Thư viện cho học máy**: Thành thạo Scikit-learn - thư viện cung cấp nhiều lựa chọn thuật toán học có giám sát và không giám sát -là rất quan trọng. Hiểu cách triển khai các thuật toán như hồi quy tuyến tính, hồi quy logistic, cây quyết định, rừng ngẫu nhiên, k-láng giềng gần nhất (K-NN) và phân cụm K-mean . Các kỹ thuật giảm kích thước như PCA và t-SNE cũng rất hữu ích để hiển thị dữ liệu nhiều chiều.

📚 Tài nguyên:

- [Real Python](https://realpython.com/): Một nguồn tài nguyên toàn diện với các bài viết và hướng dẫn cho cả khái niệm Python dành cho người mới bắt đầu và nâng cao.
- [freeCodeCamp - Learn Python](https://www.youtube.com/watch?v=rfscVS0vtbw): Video dài cung cấp phần giới thiệu đầy đủ về tất cả các khái niệm cốt lõi trong Python.
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/): Cuốn sách kỹ thuật số miễn phí là nguồn tài nguyên tuyệt vời để học về panda, NumPy, matplotlib và Seaborn.
- [freeCodeCamp - Machine Learning for Everybody](https://youtu.be/i_LwzRVP7bg): Giới thiệu thực tế về các thuật toán học máy khác nhau cho người mới bắt đầu.
- [Udacity - Intro to Machine Learning](https://www.udacity.com/course/intro-to-machine-learning--ud120): Khóa học miễn phí bao gồm PCA và một số khái niệm học máy khác.

---

### 3. Mạng nơ ron (Neural Networks)

Mạng lưới thần kinh là một phần cơ bản của nhiều mô hình học máy, đặc biệt là trong lĩnh vực học sâu. Để sử dụng chúng một cách hiệu quả, cần phải có sự hiểu biết toàn diện về thiết kế và cơ chế của chúng.

- **Cơ bản**: Gồm việc hiểu cấu trúc của mạng lưới thần kinh như các lớp, trọng số, độ lệch, hàm kích hoạt (sigmoid, tanh, ReLU, v.v.)
- **Huấn luyện và tối ưu**: Làm quen với lan truyền ngược và các loại hàm mất mát khác nhau, như Lỗi bình phương trung bình (MSE) và Entropy chéo. Hiểu các thuật toán tối ưu hóa khác nhau như Giảm dần độ dốc, Giảm dần độ dốc ngẫu nhiên, RMSprop và Adam.
- **Overfitting**: Hiểu khái niệm về overfitting (trong đó một mô hình hoạt động tốt trên dữ liệu huấn luyện nhưng kém trên dữ liệu không nhìn thấy) và các kỹ thuật chính quy hóa khác nhau để ngăn chặn overfitting. Các kỹ thuật bao gồm dropout, chuẩn hóa L1/L2, dừng sớm và tăng cường dữ liệu.
- **Triển khai Perceptron đa lớp (MLP)**: Xây dựng MLP, còn được gọi là mạng được kết nối đầy đủ, sử dụng PyTorch.

📚 Tài nguyên:

- [3Blue1Brown - But what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk): Video này đưa ra lời giải thích trực quan về mạng lưới thần kinh và hoạt động bên trong của chúng.
- [freeCodeCamp - Deep Learning Crash Course](https://www.youtube.com/watch?v=VyWAvY2CF9c): Video này giới thiệu một cách hiệu quả tất cả các khái niệm quan trọng nhất trong học sâu.
- [Fast.ai - Practical Deep Learning](https://course.fast.ai/): Khóa học miễn phí được thiết kế dành cho những người có kinh nghiệm viết mã muốn tìm hiểu về deep learning.
- [Patrick Loeber - PyTorch Tutorials](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4): loạt video dành cho người mới bắt đầu tìm hiểu về PyTorch.

---

### 4. Xử lý ngôn ngữ tự nhiên (NLP)

NLP là một nhánh của trí tuệ nhân tạo giúp thu hẹp khoảng cách giữa ngôn ngữ con người và sự hiểu biết của máy móc. Từ xử lý văn bản đơn giản đến hiểu các sắc thái ngôn ngữ, NLP đóng một vai trò quan trọng trong nhiều ứng dụng như dịch thuật, phân tích tình cảm, chatbot, v.v.

- **Tiền xử lý văn bản**: Tìm hiểu các bước tiền xử lý văn bản khác nhau như mã hóa (chia văn bản thành các từ hoặc câu), rút gọn từ gốc (rút gọn các từ về dạng gốc), từ vựng hóa (tương tự như tách gốc nhưng có tính đến ngữ cảnh), loại bỏ từ, v.v. .
- **Kỹ thuật trích xuất đặc trưng**: Làm quen với các kỹ thuật chuyển đổi dữ liệu văn bản sang định dạng mà thuật toán máy học có thể hiểu được. Các phương pháp chính bao gồm Túi từ (BoW), Tần số tài liệu nghịch đảo tần số thuật ngữ (TF-IDF) và n-gram.
- **Nhúng từ**: Nhúng từ là một kiểu trình bày từ cho phép các từ có nghĩa tương tự có cách trình bày tương tự. Các phương thức chính bao gồm Word2Vec, GloVe và FastText.
- **Mạng thần kinh hồi quy (RNN)**: Hiểu hoạt động của RNN, một loại mạng thần kinh được thiết kế để hoạt động với dữ liệu chuỗi. Khám phá LSTM và GRU, hai biến thể RNN có khả năng học các phần phụ thuộc lâu dài.
  
📚 Tài nguyên:

- [RealPython - NLP with spaCy in Python](https://realpython.com/natural-language-processing-spacy-python/): Hướng dẫn đầy đủ về thư viện spaCy cho các tác vụ NLP trong Python.
- [Kaggle - NLP Guide](https://www.kaggle.com/learn-guide/natural-language-processing): Một notebooks và tài nguyên để giải thích thực tế về NLP trong Python.
- [Jay Alammar - The Illustration Word2Vec](https://jalammar.github.io/illustrated-word2vec/): Tài liệu tham khảo tốt để hiểu kiến trúc nổi tiếng Word2Vec .
- [Jake Tae - PyTorch RNN from Scratch](https://jaketae.github.io/study/pytorch-rnn/): Triển khai thực tế và đơn giản các mô hình RNN, LSTM và GRU trong PyTorch.
- [colah's blog - Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/): Bài viết mang tính lý thuyết nhiều hơn về mạng LSTM.

## 🧑‍🔬 **Nhà nghiên cứu LLM**

![](images/roadmap_scientist.png)

### 1. Kiến trúc của LLM

While an in-depth knowledge about the Transformer architecture is not required, it is important to have a good understanding of its inputs (tokens) and outputs (logits). The vanilla attention mechanism is another crucial component to master, as improved versions of it are introduced later on.

* **High-level view**: Revisit the encoder-decoder Transformer architecture, and more specifically the decoder-only GPT architecture, which is used in every modern LLM.
* **Tokenization**: Understand how to convert raw text data into a format that the model can understand, which involves splitting the text into tokens (usually words or subwords).
* **Attention mechanisms**: Grasp the theory behind attention mechanisms, including self-attention and scaled dot-product attention, which allows the model to focus on different parts of the input when producing an output.
* **Text generation**: Learn about the different ways the model can generate output sequences. Common strategies include greedy decoding, beam search, top-k sampling, and nucleus sampling.

📚 **References**:
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar: A visual and intuitive explanation of the Transformer model.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) by Jay Alammar: Even more important than the previous article, it is focused on the GPT architecture, which is very similar to Llama's.
* [nanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy: A 2h-long YouTube video to reimplement GPT from scratch (for programmers).
* [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) by Lilian Weng: Introduce the need for attention in a more formal way.
* [Decoding Strategies in LLMs](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html): Provide code and a visual introduction to the different decoding strategies to generate text.

---
### 2. Building an instruction dataset

While it's easy to find raw data from Wikipedia and other websites, it's difficult to collect pairs of instructions and answers in the wild. Like in traditional machine learning, the quality of the dataset will directly influence the quality of the model, which is why it might be the most important component in the fine-tuning process.

* **[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)-like dataset**: Generate synthetic data from scratch with the OpenAI API (GPT). You can specify seeds and system prompts to create a diverse dataset.
* **Advanced techniques**: Learn how to improve existing datasets with [Evol-Instruct](https://arxiv.org/abs/2304.12244), how to generate high-quality synthetic data like in the [Orca](https://arxiv.org/abs/2306.02707) and [phi-1](https://arxiv.org/abs/2306.11644) papers.
* **Filtering data**: Traditional techniques involving regex, removing near-duplicates, focusing on answers with a high number of tokens, etc.
* **Prompt templates**: There's no true standard way of formatting instructions and answers, which is why it's important to know about the different chat templates, such as [ChatML](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt?tabs=python&pivots=programming-language-chat-ml), [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), etc.

📚 **References**:
* [Preparing a Dataset for Instruction tuning](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2) by Thomas Capelle: Exploration of the Alpaca and Alpaca-GPT4 datasets and how to format them.
* [Generating a Clinical Instruction Dataset](https://medium.com/mlearning-ai/generating-a-clinical-instruction-dataset-in-portuguese-with-langchain-and-gpt-4-6ee9abfa41ae) by Solano Todeschini: Tutorial on how to create a synthetic instruction dataset using GPT-4. 
* [GPT 3.5 for news classification](https://medium.com/@kshitiz.sahay26/how-i-created-an-instruction-dataset-using-gpt-3-5-to-fine-tune-llama-2-for-news-classification-ed02fe41c81f) by Kshitiz Sahay: Use GPT 3.5 to create an instruction dataset to fine-tune Llama 2 for news classification.
* [Dataset creation for fine-tuning LLM](https://colab.research.google.com/drive/1GH8PW9-zAe4cXEZyOIE-T9uHXblIldAg?usp=sharing): Notebook that contains a few techniques to filter a dataset and upload the result.
* [Chat Template](https://huggingface.co/blog/chat-templates) by Matthew Carrigan: Hugging Face's page about prompt templates

---
### 3. Pre-training models

Pre-training is a very long and costly process, which is why this is not the focus of this course. It's good to have some level of understanding of what happens during pre-training, but hands-on experience is not required.

* **Data pipeline**: Pre-training requires huge datasets (e.g., [Llama 2](https://arxiv.org/abs/2307.09288) was trained on 2 trillion tokens) that need to be filtered, tokenized, and collated with a pre-defined vocabulary.
* **Causal language modeling**: Learn the difference between causal and masked language modeling, as well as the loss function used in this case. For efficient pre-training, learn more about [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
* **Scaling laws**: The [scaling laws](https://arxiv.org/pdf/2001.08361.pdf) describe the expected model performance based on the model size, dataset size, and the amount of compute used for training.
* **High-Performance Computing**: Out of scope here, but more knowledge about HPC is fundamental if you're planning to create your own LLM from scratch (hardware, distributed workload, etc.).

📚 **References**:
* [LLMDataHub](https://github.com/Zjh-819/LLMDataHub) by Junhao Zhao: Curated list of datasets for pre-training, fine-tuning, and RLHF.
* [Training a causal language model from scratch](https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt) by Hugging Face: Pre-train a GPT-2 model from scratch using the transformers library.
* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM): State-of-the-art library to efficiently pre-train models.
* [TinyLlama](https://github.com/jzhang38/TinyLlama) by Zhang et al.: Check this project to get a good understanding of how a Llama model is trained from scratch.
* [Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) by Hugging Face: Explain the difference between causal and masked language modeling and how to quickly fine-tune a DistilGPT-2 model.
* [Chinchilla's wild implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications) by nostalgebraist: Discuss the scaling laws and explain what they mean to LLMs in general.
* [BLOOM](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4) by BigScience: Notion pages that describes how the BLOOM model was built, with a lot of useful information about the engineering part and the problems that were encountered.
* [OPT-175 Logbook](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf) by Meta: Research logs showing what went wrong and what went right. Useful if you're planning to pre-train a very large language model (in this case, 175B parameters).

---
### 4. Supervised Fine-Tuning

Pre-trained models are only trained on a next-token prediction task, which is why they're not helpful assistants. SFT allows you to tweak them into responding to instructions. Moreover, it allows you to fine-tune your model on any data (private, not seen by GPT-4, etc.) and use it without having to pay for an API like OpenAI's.

* **Full fine-tuning**: Full fine-tuning refers to training all the parameters in the model. It is not an efficient technique, but it produces slightly better results.
* [**LoRA**](https://arxiv.org/abs/2106.09685): A parameter-efficient technique (PEFT) based on low-rank adapters. Instead of training all the parameters, we only train these adapters.
* [**QLoRA**](https://arxiv.org/abs/2305.14314): Another PEFT based on LoRA, which also quantizes the weights of the model in 4 bits and introduce paged optimizers to manage memory spikes.
* **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**: A user-friendly and powerful fine-tuning tool that is used in a lot of state-of-the-art open-source models.
* [**DeepSpeed**](https://www.deepspeed.ai/): Efficient pre-training and fine-tuning of LLMs for multi-GPU and multi-node settings (implemented in Axolotl).

📚 **References**:
* [The Novice's LLM Training Guide](https://rentry.org/llm-training) by Alpin: Overview of the main concepts and parameters to consider when fine-tuning LLMs.
* [LoRA insights](https://lightning.ai/pages/community/lora-insights/) by Sebastian Raschka: Practical insights about LoRA and how to select the best parameters.
* [Fine-Tune Your Own Llama 2 Model](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html): Hands-on tutorial on how to fine-tune a Llama 2 model using Hugging Face libraries.
* [Padding Large Language Models](https://towardsdatascience.com/padding-large-language-models-examples-with-llama-2-199fb10df8ff) by Benjamin Marie: Best practices to pad training examples for causal LLMs
* [A Beginner's Guide to LLM Fine-Tuning](https://mlabonne.github.io/blog/posts/A_Beginners_Guide_to_LLM_Finetuning.html): Tutorial on how to fine-tune a CodeLlama model using Axolotl.

---
### 5. Reinforcement Learning from Human Feedback

After supervised fine-tuning, RLHF is a step used to align the LLM's answers with human expectations. The idea is to learn preferences from human (or artificial) feedback, which can be used to reduce biases, censor models, or make them act in a more useful way. It is more complex than SFT and often seen as optional.

* **Preference datasets**: These datasets typically contain several answers with some kind of ranking, which makes them more difficult to produce than instruction datasets.
* [**Proximal Policy Optimization**](https://arxiv.org/abs/1707.06347): This algorithm leverages a reward model that predicts whether a given text is highly ranked by humans. This prediction is then used to optimize the SFT model with a penalty based on KL divergence.
* **[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)**: DPO simplifies the process by reframing it as a classification problem. It uses a reference model instead of a reward model (no training needed) and only requires one hyperparameter, making it more stable and efficient.

📚 **References**:
* [An Introduction to Training LLMs using RLHF](https://wandb.ai/ayush-thakur/Intro-RLAIF/reports/An-Introduction-to-Training-LLMs-Using-Reinforcement-Learning-From-Human-Feedback-RLHF---VmlldzozMzYyNjcy) by Ayush Thakur: Explain why RLHF is desirable to reduce bias and increase performance in LLMs.
* [Illustration RLHF](https://huggingface.co/blog/rlhf) by Hugging Face: Introduction to RLHF with reward model training and fine-tuning with reinforcement learning.
* [StackLLaMA](https://huggingface.co/blog/stackllama) by Hugging Face: Tutorial to efficiently align a LLaMA model with RLHF using the transformers library.
* [Fine-tune Llama 2 with DPO](https://huggingface.co/blog/dpo-trl) by Hugging Face: Tutorial to fine-tune a Llama 2 model with DPO.
* [LLM Training: RLHF and Its Alternatives](https://substack.com/profile/27393275-sebastian-raschka-phd) by Sebastian Rashcka: Overview of the RLHF process and alternatives like RLAIF.

---
### 6. Evaluation

Evaluating LLMs is an undervalued part of the pipeline, which is time-consuming and moderately reliable. Your downstream task should dictate what you want to evaluate, but always remember the Goodhart's law: "when a measure becomes a target, it ceases to be a good measure."

* **Traditional metrics**: Metrics like perplexity and BLEU score are not popular as they were because they're flawed in most contexts. It is still important to understand them and when they can be applied.
* **General benchmarks**: Based on the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) is the main benchmark for general-purpose LLMs (like ChatGPT). There are other popular benchmarks like [BigBench](https://github.com/google/BIG-bench), [MT-Bench](https://arxiv.org/abs/2306.05685), etc.
* **Task-specific benchmarks**: Tasks like summarization, translation, question answering have dedicated benchmarks, metrics, and even subdomains (medical, financial, etc.), such as [PubMedQA](https://pubmedqa.github.io/) for biomedical question answering.
* **Human evaluation**: The most reliable evaluation is the acceptance rate by users or comparisons made by humans. If you want to know if a model performs well, the simplest but surest way is to use it yourself.

📚 **References**:
* [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity) by Hugging Face: Overview of perplexity with code to implement it with the transformers library.
* [BLEU at your own risk](https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213) by Rachael Tatman: Overview of the BLEU score and its many issues with examples.
* [A Survey on Evaluation of LLMs](https://arxiv.org/abs/2307.03109) by Chang et al.: Comprehensive paper about what to evaluate, where to evaluate, and how to evaluate.
* [Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) by lmsys: Elo rating of general-purpose LLMs, based on comparisons made by humans.

---
### 7. Quantization

Quantization is the process of converting the weights (and activations) of a model using a lower precision. For example, weights stored using 16 bits can be converted into a 4-bit representation. This technique has become increasingly important to reduce the computational and memory costs associated to LLMs.

* **Base techniques**: Learn the different levels of precision (FP32, FP16, INT8, etc.) and how to perform naïve quantization with absmax and zero-point techniques.
* **GGUF and llama.cpp**: Originally designed to run on CPUs, [llama.cpp](https://github.com/ggerganov/llama.cpp) and the GGUF format have become the most popular tools to run LLMs on consumer-grade hardware.
* **GPTQ and EXL2**: [GPTQ](https://arxiv.org/abs/2210.17323) and, more specifically, the [EXL2](https://github.com/turboderp/exllamav2) format offer an incredible speed but can only run on GPUs. Models also take a long time to be quantized.
* **AWQ**: This new format is more accurate than GPTQ (lower perplexity) but uses a lot more VRAM and is not necessarily faster.

📚 **References**:
* [Introduction to quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html): Overview of quantization, absmax and zero-point quantization, and LLM.int8() with code.
* [Quantize Llama models with llama.cpp](https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html): Tutorial on how to quantize a Llama 2 model using llama.cpp and the GGUF format.
* [4-bit LLM Quantization with GPTQ](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html): Tutorial on how to quantize an LLM using the GPTQ algorithm with AutoGPTQ.
* [ExLlamaV2: The Fastest Library to Run LLMs](https://mlabonne.github.io/blog/posts/ExLlamaV2_The_Fastest_Library_to_Run%C2%A0LLMs.html): Guide on how to quantize a Mistral model using the EXL2 format and run it with the ExLlamaV2 library.
* [Understanding Activation-Aware Weight Quantization](https://medium.com/friendliai/understanding-activation-aware-weight-quantization-awq-boosting-inference-serving-efficiency-in-10bb0faf63a8) by FriendliAI: Overview of the AWQ technique and its benefits.

---
### 8. Inference optimization

* **Flash Attention**: Optimization of the attention mechanism to transform its complexity from quadratic to linear, speeding up both training and inference.
* **Key-value cache**: Understand the key-value cache and the improvements introduced in [Multi-Query Attention](https://arxiv.org/abs/1911.02150) (MQA) and [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) (GQA).
* **Speculative decoding**: Use a small model to produce drafts that are then reviewed by a larger model to speed up text generation.
* **Positional encoding**: Understand positional encodings in transformers, particularly relative schemes like [RoPE](https://arxiv.org/abs/2104.09864), [ALiBi](https://arxiv.org/abs/2108.12409), and [YaRN](https://arxiv.org/abs/2309.00071). (Not directly connected to inference optimization but to longer context windows.)

📚 **References**:
* [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one) by Hugging Face: Explain how to optimize inference on GPUs.
* [Optimizing LLMs for Speed and Memory](https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization) by Hugging Face: Explain three main techniques to optimize speed and memory, namely quantization, Flash Attention, and architectural innovations.
* [Assisted Generation](https://huggingface.co/blog/assisted-generation) by Hugging Face: HF's version of speculative decoding, it's an interesting blog post about how it works with code to implement it.
* [Extending the RoPE](https://blog.eleuther.ai/yarn/) by EleutherAI: Article that summarizes the different position-encoding techniques.
* [Extending Context is Hard... but not Impossible](https://kaiokendev.github.io/context) by kaiokendev: This blog post introduces the SuperHOT technique and provides an excellent survey of related work.

## 👷 The LLM Engineer

W.I.P.

---
### Contributions

Feel free to contact me if you think other topics should be mentioned or if the current architecture can be improved.

### Acknowledgements

This roadmap was inspired by the excellent [DevOps Roadmap](https://github.com/milanm/DevOps-Roadmap) from Milan Milanović and Romano Roth.

Special thanks to Thomas Thelen for motivating me to create a roadmap, and André Frade for his input and review of the first draft.

*Disclaimer: I am not affiliated with any sources listed here.*

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mlabonne/llm-course&type=Date)](https://star-history.com/#mlabonne/llm-course&Date)
