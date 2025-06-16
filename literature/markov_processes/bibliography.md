# Bibliography: Autoregressive Models and Markov Processes

This bibliography compiles academic papers that explore the connections between autoregressive models and Markov processes, particularly in the context of neural networks, transformers, and language models.

## 1. Direct Connections Between Autoregressive Models and Markov Processes

### Cascaded Text Generation with Markov Transformers
**Authors:** Yuntian Deng, Alexander M. Rush  
**Published:** June 1, 2020  
**Link:** [https://hf.co/papers/2006.01112](https://hf.co/papers/2006.01112)  
**Summary:** Introduces a Markov transformer that combines autoregressive models with bounded-context conditional random fields, enabling parallel decoding while maintaining specific autoregressive context cutoffs. This work explicitly bridges Markov processes and transformer architectures.

### Autoregressive Hidden Markov Models with partial knowledge on latent space applied to aero-engines prognostics
**Authors:** Pablo Juesas, Emmanuel Ramasso, Sébastien Drujont, Vincent Placet  
**Published:** May 1, 2021  
**Link:** [https://hf.co/papers/2105.00211](https://hf.co/papers/2105.00211)  
**Summary:** Describes an Autoregressive Partially-hidden Markov model (ARPHMM) that combines Hidden Markov Models with autoregressive processes, demonstrating how these two paradigms can be integrated for fault detection and prognostics.

### TRACE Back from the Future: A Probabilistic Reasoning Approach to Controllable Language Generation
**Authors:** Gwen Yidou Weng, Benjie Wang, Guy Van den Broeck  
**Published:** April 25, 2025  
**Link:** [https://hf.co/papers/2504.18535](https://hf.co/papers/2504.18535)  
**Summary:** Introduces TRACE, which distills a Hidden Markov Model from a language model to enable tractable probabilistic reasoning for controllable generation, explicitly connecting autoregressive LMs with Markov models.

## 2. Markov Property in Language Models and Tokenization

### Toward a Theory of Tokenization in LLMs
**Authors:** Nived Rajaraman, Jiantao Jiao, Kannan Ramchandran  
**Published:** April 12, 2024  
**Link:** [https://hf.co/papers/2404.08335](https://hf.co/papers/2404.08335)  
**Summary:** Studies transformers trained on k-th order Markov processes, showing that without tokenization, transformers fail to learn the right distribution and predict according to a unigram model. With tokenization, they can model probabilities near-optimally, providing theoretical justification for tokenization through Markovian analysis.

### Understanding and Mitigating Tokenization Bias in Language Models
**Authors:** Buu Phan, Marton Havasi, Matthew Muckley, Karen Ullrich  
**Published:** June 24, 2024  
**Link:** [https://hf.co/papers/2406.16829](https://hf.co/papers/2406.16829)  
**Summary:** Shows how tokenization schemes introduce sampling bias in autoregressive models and proposes methods to obtain unbiased estimates. Uses Markov-chain setups to verify the correctness of their approach.

## 3. Neural Networks and Markov Jump Processes

### Foundation Inference Models for Markov Jump Processes
**Authors:** David Berghaus, Kostadin Cvejoski, Patrick Seifner, Cesar Ojeda, Ramses J. Sanchez  
**Published:** June 10, 2024  
**Link:** [https://hf.co/papers/2406.06419](https://hf.co/papers/2406.06419)  
**Summary:** Introduces neural network models for zero-shot inference of Markov jump processes, demonstrating how neural architectures can learn to model complex Markovian dynamics in various domains.

### Neural Markov Jump Processes
**Authors:** Patrick Seifner, Ramses J. Sanchez  
**Published:** May 31, 2023  
**Link:** [https://hf.co/papers/2305.19744](https://hf.co/papers/2305.19744)  
**Summary:** Introduces variational inference algorithms for Markov jump processes using neural ordinary differential equations, showing how neural continuous-time representations can approximate Markovian dynamics.

## 4. Memory and Markov Properties in Deep Learning

### On the Markov Property of Neural Algorithmic Reasoning: Analyses and Methods
**Authors:** Montgomery Bohde, Meng Liu, Alexandra Saxton, Shuiwang Ji  
**Published:** March 7, 2024  
**Link:** [https://hf.co/papers/2403.04929](https://hf.co/papers/2403.04929)  
**Summary:** Analyzes how historical embeddings in neural algorithmic reasoning contradict the Markov nature of algorithmic tasks. Proposes ForgetNet, which respects the Markov property by not using historical embeddings.

### Reinforcement Learning with Fast and Forgetful Memory
**Authors:** Steven Morad, Ryan Kortvelesy, Stephan Liwicki, Amanda Prorok  
**Published:** October 6, 2023  
**Link:** [https://hf.co/papers/2310.04128](https://hf.co/papers/2310.04128)  
**Summary:** Introduces a memory model for RL that summarizes trajectories into latent Markov states, designed specifically for reinforcement learning rather than supervised learning paradigms.

### Extending Conformal Prediction to Hidden Markov Models with Exact Validity via de Finetti's Theorem for Markov Chains
**Authors:** Buddhika Nettasinghe, Samrat Chatterjee, Ramakrishna Tipireddy, Mahantesh Halappanavar  
**Published:** October 5, 2022  
**Link:** [https://hf.co/papers/2210.02271](https://hf.co/papers/2210.02271)  
**Summary:** Extends conformal prediction to HMM frameworks, addressing the non-exchangeability of Markovian data using de Finetti's Theorem for Markov Chains.

## 5. Autoregressive Models Beyond Simple Markov Assumptions

### Autoregressive Diffusion Models
**Authors:** Emiel Hoogeboom, Alexey A. Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, Tim Salimans  
**Published:** October 5, 2021  
**Link:** [https://hf.co/papers/2110.02037](https://hf.co/papers/2110.02037)  
**Summary:** Introduces Autoregressive Diffusion Models (ARDMs) that generalize order-agnostic autoregressive models and absorbing discrete diffusion, showing these are special cases under mild assumptions.

### AR-Net: A simple Auto-Regressive Neural Network for time-series
**Authors:** Oskar Triebe, Nikolay Laptev, Ram Rajagopal  
**Published:** November 27, 2019  
**Link:** [https://hf.co/papers/1911.12436](https://hf.co/papers/1911.12436)  
**Summary:** Proposes modeling AR-process dynamics using feed-forward neural networks, bridging statistical AR models with deep learning while maintaining interpretability.

### Neural Autoregressive Distribution Estimation
**Authors:** Benigno Uria, Marc-Alexandre Côté, Karol Gregor, Iain Murray, Hugo Larochelle  
**Published:** May 7, 2016  
**Link:** [https://hf.co/papers/1605.02226](https://hf.co/papers/1605.02226)  
**Summary:** Presents NADE models that leverage the probability product rule for distribution estimation, showing how to make models agnostic to input dimension ordering.

## 6. Memory Limitations and State Space Models

### StableSSM: Alleviating the Curse of Memory in State-space Models through Stable Reparameterization
**Authors:** Shida Wang, Qianxiao Li  
**Published:** November 24, 2023  
**Link:** [https://hf.co/papers/2311.14495](https://hf.co/papers/2311.14495)  
**Summary:** Proves that state-space models exhibit memory limitations similar to RNNs with exponentially decaying memory for target relationships, and proposes reparameterization techniques to lift these limitations.

### Latent State Models of Training Dynamics
**Authors:** Michael Y. Hu, Angelica Chen, Naomi Saphra, Kyunghyun Cho  
**Published:** August 18, 2023  
**Link:** [https://hf.co/papers/2308.09543](https://hf.co/papers/2308.09543)  
**Summary:** Uses Hidden Markov Models to analyze neural network training dynamics, representing training as a stochastic process of transitions between latent states.

## 7. Scale and Efficiency in Autoregressive Models

### MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers
**Authors:** Lili Yu, Dániel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, Mike Lewis  
**Published:** May 12, 2023  
**Link:** [https://hf.co/papers/2305.07185](https://hf.co/papers/2305.07185)  
**Summary:** Proposes a multi-scale decoder architecture that segments sequences into patches with local and global models, enabling tokenization-free autoregressive modeling at scale.

### Beyond Autoregression: Fast LLMs via Self-Distillation Through Time
**Authors:** Justin Deschenaux, Caglar Gulcehre  
**Published:** October 28, 2024  
**Link:** [https://hf.co/papers/2410.21035](https://hf.co/papers/2410.21035)  
**Summary:** Shows that diffusion language models can generate multiple tokens simultaneously, challenging the traditional one-token-at-a-time Markovian assumption of autoregressive models.

## 8. Continuous and Non-Discrete Autoregressive Models

### Continuous Autoregressive Models with Noise Augmentation Avoid Error Accumulation
**Authors:** Marco Pasini, Javier Nistal, Stefan Lattner, George Fazekas  
**Published:** November 27, 2024  
**Link:** [https://hf.co/papers/2411.18447](https://hf.co/papers/2411.18447)  
**Summary:** Addresses error accumulation in continuous autoregressive models by injecting noise during training, making models robust against varying error levels at inference.

### Autoregressive Image Generation without Vector Quantization
**Authors:** Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, Kaiming He  
**Published:** June 17, 2024  
**Link:** [https://hf.co/papers/2406.11838](https://hf.co/papers/2406.11838)  
**Summary:** Proposes modeling per-token probability distributions using diffusion procedures in continuous-valued spaces, eliminating the need for discrete tokenization in autoregressive image generation.

## Key Themes and Insights

1. **Markov Property Preservation vs. Violation**: Several papers explore when neural models preserve or violate the Markov property, particularly in the context of using historical embeddings or memory mechanisms.

2. **Tokenization and Markov Processes**: Tokenization emerges as a crucial factor in enabling transformers to model higher-order Markov processes effectively.

3. **Hybrid Approaches**: Many successful models combine autoregressive mechanisms with Markov models (HMMs, Markov transformers) to leverage the strengths of both paradigms.

4. **Memory and State**: The tension between maintaining sufficient state information and respecting Markovian assumptions is a recurring theme, with various solutions proposed.

5. **Continuous vs. Discrete**: Recent work explores extending autoregressive models beyond discrete tokens to continuous spaces while maintaining or adapting Markovian properties.

## Future Directions

The literature suggests several promising directions:
- Theoretical analysis of when and why transformers exhibit non-Markovian behavior
- Development of architectures that can adaptively switch between Markovian and non-Markovian regimes
- Better understanding of the role of attention mechanisms in creating implicit memory beyond the Markov property
- Integration of classical Markov theory with modern deep learning architectures