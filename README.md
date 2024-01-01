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

Mặc dù không cần phải có kiến thức chuyên sâu về kiến trúc Transformer nhưng điều quan trọng là phải hiểu rõ về đầu vào (mã thông báo) và đầu ra (logits) của nó. Cơ chế chú ý cơ bản là một thành phần quan trọng khác cần phải thành thạo, vì các phiên bản cải tiến của nó sẽ được giới thiệu sau này.

* **Góc nhìn tổng quan**: Xem lại kiến trúc Transformer bộ mã hóa-giải mã và cụ thể hơn là kiến trúc GPT chỉ dành cho bộ giải mã, được sử dụng trong mọi LLM hiện đại.
* **Mã thông báo (Tokenization)**: Hiểu cách chuyển đổi dữ liệu văn bản thô sang định dạng mà mô hình có thể hiểu được, bao gồm việc chia văn bản thành các mã thông báo (thường là từ hoặc từ phụ).
* **Cơ chế chú ý (Attention mechanisms)**: Nắm bắt lý thuyết đằng sau các cơ chế chú ý, bao gồm cả sự tự chú ý và sự chú ý của tích vô hướng theo tỷ lệ (scaled dot product), cho phép mô hình tập trung vào các phần khác nhau của đầu vào khi tạo đầu ra.
* **Tạo văn bản**: Tìm hiểu về các cách khác nhau mà mô hình có thể tạo ra chuỗi đầu ra. Các chiến lược phổ biến bao gồm giải mã tham lam, tìm kiếm chùm tia, lấy mẫu top-k và lấy mẫu hạt nhân.

📚 **Tài liệu tham khảo**:
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) của Jay Alammar: Giải thích trực quan về mô hình Transformer.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) của Jay Alammar: Thậm chí còn quan trọng hơn bài viết trước, nó tập trung vào kiến trúc GPT, rất giống với kiến trúc của Llama.
* [nanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) của Andrej Karpathy: Một video YouTube dài 2 giờ để triển khai lại GPT từ đầu (dành cho lập trình viên).
* [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) của Lilian Weng: Giới thiệu nhu cầu được chú ý theo cách trang trọng hơn.
* [Decoding Strategies in LLMs](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html): Cung cấp mã và phần giới thiệu trực quan về các chiến lược giải mã khác nhau để tạo văn bản.

---
### 2. Xây dựng tập dữ liệu hướng dẫn

Mặc dù có thể dễ dàng để tìm thấy dữ liệu thô từ Wikipedia và các trang web khác, nhưng việc thu thập các cặp hướng dẫn và câu trả lời một cách tự nhiên lại rất khó khăn. Giống như trong học máy truyền thống, chất lượng của tập dữ liệu sẽ ảnh hưởng trực tiếp đến chất lượng của mô hình, đó là lý do tại sao nó có thể là thành phần quan trọng nhất trong quá trình tinh chỉnh.

* **[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)-like dataset**: Tạo dữ liệu tổng hợp từ đầu bằng API OpenAI (GPT). Bạn có thể chỉ định seed và lời nhắc hệ thống để tạo tập dữ liệu đa dạng.
* **Kỹ thuật nâng cao**: Tìm hiểu cách cải thiện các tập dữ liệu hiện có với [Evol-Instruct](https://arxiv.org/abs/2304.12244), cách tạo dữ liệu tổng hợp chất lượng cao như trong [Orca](https ://arxiv.org/abs/2306.02707) và các bài viết [phi-1](https://arxiv.org/abs/2306.11644).
* **Lọc dữ liệu**: Các kỹ thuật truyền thống liên quan đến biểu thức chính quy, loại bỏ các nội dung gần trùng lặp, tập trung vào các câu trả lời có số lượng mã thông báo cao, v.v.
* **Mẫu lời nhắc**: Không có cách tiêu chuẩn thực sự nào để định dạng hướng dẫn và câu trả lời, đó là lý do tại sao điều quan trọng là phải biết về các mẫu trò chuyện khác nhau, chẳng hạn như [ChatML](https://learn.microsoft.com/en- us/azure/ai-services/openai/how-to/chatgpt?tabs=python&pivots=programming-lingu-chat-ml), [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca .html), v.v.

📚 **Tài liệu tham khảo**:
* [Preparing a Dataset for Instruction tuning](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2) của Thomas Capelle: Khám phá bộ dữ liệu Alpaca và Alpaca-GPT4 và cách định dạng chúng.
* [Generating a Clinical Instruction Dataset](https://medium.com/mlearning-ai/generating-a-clinical-instruction-dataset-in-portuguese-with-langchain-and-gpt-4-6ee9abfa41ae) của Solano Todeschini: Hướng dẫn cách tạo tập dữ liệu hướng dẫn tổng hợp bằng GPT-4.
* [GPT 3.5 for news classification](https://medium.com/@kshitiz.sahay26/how-i-created-an-instruction-dataset-using-gpt-3-5-to-fine-tune-llama-2-for-news-classification-ed02fe41c81f) của Kshitiz Sahay: Sử dụng GPT 3.5 để tạo tập dữ liệu hướng dẫn nhằm tinh chỉnh Llama 2 để phân loại tin tức.
* [Dataset creation for fine-tuning LLM](https://colab.research.google.com/drive/1GH8PW9-zAe4cXEZyOIE-T9uHXblIldAg?usp=sharing): Notebooks chứa một số kỹ thuật để lọc tập dữ liệu và tải kết quả lên.
* [Chat Template](https://huggingface.co/blog/chat-templates) của Matthew Carrigan: Trang của Hugging Face về các mẫu lời nhắc

---
### 3. Mô hình huấn luyện trước (Pre-training models)

Huấn luyện trước là một quá trình rất dài và tốn kém, đó là lý do tại sao đây không phải là trọng tâm của khóa học này. Sẽ rất tốt nếu bạn hiểu biết ở mức độ nào đó về những gì xảy ra trong quá trình huấn luyện trước, nhưng không cần phải có kinh nghiệm thực hành.

* **Đường ống dữ liệu (Data pipeline)**: Quá trình huấn luyện trước yêu cầu các bộ dữ liệu khổng lồ (ví dụ: [Llama 2](https://arxiv.org/abs/2307.09288) đã được huấn luyện trên 2 nghìn tỷ mã thông báo) cần được lọc, mã hóa và đối chiếu với một từ vựng được xác định trước.
* **Mô hình hóa ngôn ngữ nhân quả**: Tìm hiểu sự khác biệt giữa mô hình hóa ngôn ngữ nhân quả và ngôn ngữ mặt nạ, cũng như hàm mất mát được sử dụng trong trường hợp này. Để huấn luyện trước hiệu quả, hãy tìm hiểu thêm về [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
* **Luật chia tỷ lệ**: [Luật chia tỷ lệ](https://arxiv.org/pdf/2001.08361.pdf) mô tả hiệu suất dự kiến của mô hình dựa trên kích thước mô hình, kích thước tập dữ liệu và lượng điện toán được sử dụng để đào tạo .


📚 **Tài liệu tham khảo**:
* [LLMDataHub](https://github.com/Zjh-819/LLMDataHub) của Junhao Zhao: Danh sách các bộ dữ liệu được tuyển chọn để đào tạo trước, tinh chỉnh và RLHF.
* [Training a causal language model from scratch](https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt) của Hugging Face:: Huấn luyện trước mô hình GPT-2 từ đầu bằng thư viện transformers.
* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM): Thư viện hiện đại để huấn luyện trước các mô hình một cách hiệu quả.
* [TinyLlama](https://github.com/jzhang38/TinyLlama) của Zhang et al.: Hãy xem dự án này để hiểu rõ về cách đào tạo mô hình Llama ngay từ đầu.
* [Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) của Hugging Face: Giải thích sự khác biệt giữa mô hình hóa ngôn ngữ nhân quả và ngôn ngữ mặt nạ cũng như cách tinh chỉnh nhanh chóng mô hình DistilGPT-2.
* [Chinchilla's wild implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications) của nostalgebraist: Thảo luận về các quy luật chia tỷ lệ và giải thích ý nghĩa của chúng đối với LLM nói chung.
* [BLOOM](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4) của BigScience: Các trang khái niệm mô tả cách xây dựng mô hình BLOOM, với nhiều thông tin hữu ích về phần kỹ thuật và các vấn đề gặp phải.
* [OPT-175 Logbook](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf) của Meta: Nhật ký nghiên cứu cho thấy điều gì sai và điều gì đúng. Hữu ích nếu bạn dự định đào tạo trước một mô hình ngôn ngữ rất lớn (trong trường hợp này là tham số 175B).

---
### 4. Tinh chỉnh được giám sát (Supervised Fine-Tuning)

Các mô hình được đào tạo trước chỉ được đào tạo về nhiệm vụ dự đoán mã thông báo tiếp theo, đó là lý do tại sao chúng không phải là trợ lý hữu ích. SFT cho phép bạn điều chỉnh chúng để đáp ứng các chỉ dẫn dẫn. Hơn nữa, nó cho phép bạn tinh chỉnh mô hình của mình trên bất kỳ dữ liệu nào (riêng tư, không bị GPT-4 nhìn thấy, v.v.) và sử dụng nó mà không phải trả tiền cho API như OpenAI.

* **Tinh chỉnh toàn bộ (Full fine-tuning)**: Tinh chỉnh toàn bộ đề cập đến việc huấn luyện tất cả các tham số trong mô hình. Đây không phải là một kỹ thuật hiệu quả nhưng nó mang lại kết quả tốt hơn một chút.
* [**LoRA**](https://arxiv.org/abs/2106.09685): Một kỹ thuật hiệu quả về tham số (PEFT) dựa trên các bộ điều hợp cấp thấp. Thay vì đào tạo tất cả các tham số, chúng tôi chỉ đào tạo những bộ điều hợp này.
* [**QLoRA**](https://arxiv.org/abs/2305.14314): Một PEFT khác dựa trên LoRA, cũng lượng tử hóa trọng số của mô hình thành 4 bit và giới thiệu các trình tối ưu hóa phân trang để quản lý mức tăng đột biến của bộ nhớ.
* **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**: Một công cụ tinh chỉnh mạnh mẽ và thân thiện với người dùng, được sử dụng trong nhiều mô hình nguồn mở tiên tiến nhất.
* [**DeepSpeed**](https://www.deepspeed.ai/): Đào tạo trước và tinh chỉnh LLM hiệu quả cho cài đặt nhiều GPU và nhiều nút (được triển khai trong Axolotl).


📚 **Tài liệu tham khảo**:
* [The Novice's LLM Training Guide](https://rentry.org/llm-training) bởi Alpin: Tổng quan về các khái niệm và tham số chính cần xem xét khi tinh chỉnh LLM.
* [LoRA insights](https://lightning.ai/pages/community/lora-insights/) bởi Sebastian Raschka: Những hiểu biết thực tế về LoRA và cách chọn thông số tốt nhất.
* [Fine-Tune Your Own Llama 2 Model](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html): Hướng dẫn thực hành về cách tinh chỉnh mô hình Llama 2 bằng thư viện Hugging Face.
* [Padding Large Language Models](https://towardsdatascience.com/padding-large-language-models-examples-with-llama-2-199fb10df8ff) của Benjamin Marie: Các phương pháp hay nhất để bổ sung các ví dụ đào tạo cho LLM nhân quả
* [A Beginner's Guide to LLM Fine-Tuning](https://mlabonne.github.io/blog/posts/A_Beginners_Guide_to_LLM_Finetuning.html): Hướng dẫn cách tinh chỉnh mô hình CodeLlama bằng Axolotl.

---
### 5. Học tăng cường từ phản hồi của con người

Sau khi tinh chỉnh có giám sát, RLHF là bước được sử dụng để điều chỉnh các câu trả lời của LLM phù hợp với mong đợi của con người. Ý tưởng là tìm hiểu các sở thích từ phản hồi của con người (hoặc nhân tạo), phản hồi này có thể được sử dụng để giảm bớt thành kiến, kiểm duyệt mô hình hoặc khiến họ hành động theo cách hữu ích hơn. Nó phức tạp hơn SFT và thường được xem là tùy chọn.

* **Bộ dữ liệu ưu tiên**: Những bộ dữ liệu này thường chứa một số câu trả lời với một số loại xếp hạng, điều này khiến chúng khó tạo ra hơn các bộ dữ liệu hướng dẫn.
* [**Tối ưu hóa chính sách gần nhất**](https://arxiv.org/abs/1707.06347): Thuật toán này tận dụng mô hình phần thưởng để dự đoán liệu một văn bản nhất định có được con người xếp hạng cao hay không. Dự đoán này sau đó được sử dụng để tối ưu hóa mô hình SFT với mức phạt dựa trên sự phân kỳ KL.
* **[Tối ưu hóa tùy chọn trực tiếp](https://arxiv.org/abs/2305.18290)**: DPO đơn giản hóa quy trình bằng cách sắp xếp lại quy trình thành một vấn đề phân loại. Nó sử dụng mô hình tham chiếu thay vì mô hình phần thưởng (không cần đào tạo) và chỉ yêu cầu một siêu tham số, giúp nó ổn định và hiệu quả hơn.

📚 **Tài liệu tham khảo**:
* [An Introduction to Training LLMs using RLHF](https://wandb.ai/ayush-thakur/Intro-RLAIF/reports/An-Introduction-to-Training-LLMs-Using-Reinforcement-Learning-From-Human-Feedback-RLHF---VmlldzozMzYyNjcy) của Ayush Thakur: Giải thích tại sao RLHF lại được mong muốn để giảm sai lệch và tăng hiệu suất trong LLM.
* [Illustration RLHF](https://huggingface.co/blog/rlhf) của Hugging Face: Giới thiệu về RLHF với đào tạo mô hình khen thưởng và tinh chỉnh bằng học tập tăng cường.
* [StackLLaMA](https://huggingface.co/blog/stackllama) của Hugging Face: Hướng dẫn căn chỉnh hiệu quả mô hình LLaMA với RLHF bằng thư viện transformers.
* [Fine-tune Llama 2 with DPO](https://huggingface.co/blog/dpo-trl) của Hugging Face: Hướng dẫn tinh chỉnh mô hình Llama 2 bằng DPO.
* [LLM Training: RLHF and Its Alternatives](https://substack.com/profile/27393275-sebastian-raschka-phd) của Sebastian Rashcka: Tổng quan về quy trình RLHF và các lựa chọn thay thế như RLAIF.

---
### 6. Đánh giá (Evaluation)

Đánh giá LLM là một phần được đánh giá thấp trong quy trình, tốn thời gian và có độ tin cậy vừa phải. Nhiệm vụ tiếp theo của bạn phải chỉ ra những gì bạn muốn đánh giá, nhưng hãy luôn nhớ định luật Goodhart: "khi một thước đo trở thành mục tiêu, nó không còn là thước đo tốt nữa".
* **Số liệu truyền thống**: Các số liệu như mức độ bối rối và điểm BLEU không còn phổ biến vì chúng có sai sót trong hầu hết các bối cảnh. Điều quan trọng vẫn là phải hiểu chúng và khi nào chúng có thể được áp dụng.
* **Điểm chuẩn chung**: Dựa trên [Khai thác đánh giá mô hình ngôn ngữ](https://github.com/EleutherAI/lm-evaluation-harness), [Bảng xếp hạng LLM mở](https://huggingface.co/ Spaces/HuggingFaceH4/open_llm_leaderboard) là điểm chuẩn chính cho các LLM có mục đích chung (như ChatGPT). Có các điểm chuẩn phổ biến khác như [BigBench](https://github.com/google/BIG-bench), [MT-Bench](https://arxiv.org/abs/2306.05685), v.v.
* **Điểm chuẩn dành riêng cho từng nhiệm vụ**: Các nhiệm vụ như tóm tắt, dịch thuật, trả lời câu hỏi có điểm chuẩn, số liệu chuyên dụng và thậm chí cả các miền phụ (y tế, tài chính, v.v.), chẳng hạn như [PubMedQA](https://pubmedqa.github. io/) để trả lời câu hỏi y sinh.
* **Đánh giá của con người**: Đánh giá đáng tin cậy nhất là tỷ lệ chấp nhận của người dùng hoặc so sánh do con người thực hiện. Nếu bạn muốn biết một mô hình có hoạt động tốt hay không, cách đơn giản nhất nhưng chắc chắn nhất là bạn hãy tự mình sử dụng nó.

📚 **Tài liệu tham khảo**:
* [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity) của Hugging Face: Tổng quan về sự phức tạp của mã để triển khai nó với thư viện transformers.
* [BLEU at your own risk](https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213) của Rachael Tatman: Tổng quan về điểm BLEU và nhiều vấn đề kèm theo ví dụ.
* [A Survey on Evaluation of LLMs](https://arxiv.org/abs/2307.03109) của Chang et al.: Bài viết đầy đủ về những gì cần đánh giá, đánh giá ở đâu và đánh giá như thế nào.
* [Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) của lmsys: Xếp hạng Elo của LLM có mục đích chung, dựa trên sự so sánh của con người.

---
### 7. Lượng tử hóa (Quantization)

Lượng tử hóa là quá trình chuyển đổi trọng số (và kích hoạt) của mô hình với độ chính xác thấp hơn. Ví dụ: trọng số được lưu trữ bằng 16 bit có thể được chuyển đổi thành biểu diễn 4 bit. Kỹ thuật này ngày càng trở nên quan trọng để giảm chi phí tính toán và bộ nhớ liên quan đến LLM.

* **Kỹ thuật cơ bản**: Tìm hiểu các mức độ chính xác khác nhau (FP32, FP16, INT8, v.v.) và cách thực hiện lượng tử hóa đơn giản bằng kỹ thuật absmax và điểm 0.
* **GGUF và llama.cpp**: Ban đầu được thiết kế để chạy trên CPU, [llama.cpp](https://github.com/ggerganov/llama.cpp) và định dạng GGUF đã trở thành công cụ phổ biến nhất để chạy LLM trên phần cứng cấp độ người tiêu dùng.
* **GPTQ và EXL2**: [GPTQ](https://arxiv.org/abs/2210.17323) và cụ thể hơn là định dạng [EXL2](https://github.com/turboderp/exllamav2) cung cấp tốc độ đáng kinh ngạc nhưng chỉ có thể chạy trên GPU. Các mô hình cũng mất nhiều thời gian để lượng tử hóa.
* **AWQ**: Định dạng mới này chính xác hơn GPTQ (độ phức tạp thấp hơn) nhưng sử dụng nhiều VRAM hơn và không nhất thiết phải nhanh hơn.

📚 **Tài liệu tham khảo**:
* [Introduction to quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html): Tổng quan về lượng tử hóa, lượng tử hóa absmax và điểm 0 cũng như LLM.int8() bằng mã.
* [Quantize Llama models with llama.cpp](https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html): Hướng dẫn cách lượng tử hóa mô hình Llama 2 bằng cách sử dụng llama.cpp và định dạng GGUF.
* [4-bit LLM Quantization with GPTQ](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html): Hướng dẫn cách định lượng LLM bằng thuật toán GPTQ với AutoGPTQ.
* [ExLlamaV2: The Fastest Library to Run LLMs](https://mlabonne.github.io/blog/posts/ExLlamaV2_The_Fastest_Library_to_Run%C2%A0LLMs.html): Hướng dẫn cách lượng tử hóa mô hình Mistral bằng định dạng EXL2 và chạy nó với thư viện ExLlamaV2.

---
### 8. Tối ưu hóa suy luận (Inference optimization)

* **Chú ý nhanh (flash attention)**: Tối ưu hóa cơ chế chú ý để chuyển đổi độ phức tạp của nó từ bậc hai sang tuyến tính, tăng tốc cả quá trình huấn luyện và suy luận.
* **Bộ nhớ đệm khóa-giá trị**: Hiểu bộ nhớ đệm khóa-giá trị và các cải tiến được giới thiệu trong [Chú ý nhiều truy vấn](https://arxiv.org/abs/1911.02150) (MQA) và [Chú ý truy vấn theo nhóm] (https://arxiv.org/abs/2305.13245) (GQA).
* **Giải mã suy đoán**: Sử dụng mô hình nhỏ để tạo bản nháp, sau đó được mô hình lớn hơn xem xét để tăng tốc độ tạo văn bản.
* **Mã hóa vị trí**: Hiểu mã hóa vị trí trong transfomers, đặc biệt là các sơ đồ tương đối như [RoPE](https://arxiv.org/abs/2104.09864), [ALiBi](https://arxiv.org/abs/2108.12409 ) và [YaRN](https://arxiv.org/abs/2309.00071). (Không được kết nối trực tiếp với tối ưu hóa suy luận mà với các cửa sổ ngữ cảnh dài hơn.)

📚 **Tài liệu tham khảo**:
* [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one) của Hugging Face: Giải thích cách tối ưu hóa suy luận trên GPU.
* [Optimizing LLMs for Speed and Memory](https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization) của Hugging Face: Giải thích ba kỹ thuật chính để tối ưu hóa tốc độ và bộ nhớ, đó là lượng tử hóa, Chú ý nhanh và đổi mới kiến trúc.
* [Assisted Generation](https://huggingface.co/blog/assisted-generation) của Hugging Face: Phiên bản giải mã suy đoán của HF, đây là một bài đăng blog thú vị về cách nó hoạt động với mã để triển khai nó.
* [Extending the RoPE](https://blog.eleuther.ai/yarn/) của EleutherAI: Bài viết tóm tắt các kỹ thuật mã hóa vị trí khác nhau.
* [Extending Context is Hard... but not Impossible](https://kaiokendev.github.io/context) của kaiokendev: Bài đăng trên blog này giới thiệu kỹ thuật SuperHOT và cung cấp một bản khảo sát tuyệt vời về công việc liên quan.

### Lời cảm ơn

Khóa học này được dịch từ [Maxime Labonne](https://github.com/mlabonne/llm-course)

Special thanks to Thomas Thelen for motivating me to create a roadmap, and André Frade for his input and review of the first draft.

*Disclaimer: I am not affiliated with any sources listed here.*

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mlabonne/llm-course&type=Date)](https://star-history.com/#mlabonne/llm-course&Date)
