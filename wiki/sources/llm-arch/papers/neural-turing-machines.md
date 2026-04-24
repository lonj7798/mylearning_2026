<!-- scope: differentiable external memory for neural networks
     deps: [[seq2seq]]
     see-also: [[mamba]], [[pointer-networks]], [[paged-attention]]
-->

# Neural Turing Machines
- **Core Insight:** Augmenting neural networks with differentiable external memory (readable and writable via attention) enables learning of algorithmic tasks that require working memory, like copying, sorting, and recall.
- **Guideline:** When a task requires persistent storage, random access, or algorithmic manipulation beyond what hidden states provide, consider explicit memory mechanisms rather than relying solely on implicit memory in weights or hidden states.
- **Authors:** Alex Graves, Greg Wayne, Ivo Danihelka
- **Year:** 2014
- **URL:** https://arxiv.org/abs/1410.5401
- **Relevant chapters:** Memory mechanisms, attention as memory access, KV cache as external memory, long-context architectures

## Abstract
We extend the capabilities of neural networks by coupling them to external memory resources, which they can interact with by attentional processes. The combined system is analogous to a Turing Machine or Von Neumann architecture but is differentiable end-to-end, allowing it to be efficiently trained with gradient descent. Preliminary results demonstrate that Neural Turing Machines can infer simple algorithms such as copying, sorting, and associative recall from input and output examples.

## Key Contributions
- Introduced the concept of differentiable external memory: a matrix M of memory slots that a neural controller reads from and writes to using soft attention weights -- fully differentiable and trainable end-to-end
- Defined two addressing mechanisms: content-based addressing (query-key similarity, essentially attention) and location-based addressing (shifts for sequential access) -- combining associative and positional retrieval
- Demonstrated that NTMs can learn simple algorithms (copy, repeat copy, associative recall, priority sort) from examples, generalizing to longer sequences than seen during training
- Established the foundational paradigm of "controller + external memory" that influenced all subsequent memory-augmented architectures, including Differentiable Neural Computers and modern retrieval-augmented systems
- Showed that attention mechanisms can serve as a general-purpose memory access primitive, not just sequence alignment

## Key Figures/Tables to Study
- **Figure 1** (NTM architecture): Shows the controller (LSTM or feedforward) connected to a memory matrix via read and write heads. The read/write operations use attention-weighted access. This is the foundational diagram.
- **Figure 2** (Content-based vs. location-based addressing): Illustrates how the NTM combines key-similarity lookup (like attention) with positional shifting (like a tape head). Modern KV caches use only the content-based part.
- **Figure 3** (Copy task learning curves): NTM learns to copy sequences perfectly and generalizes to longer sequences than seen in training, while vanilla LSTM fails.
- **Figure 5** (Memory access patterns during copy): Visualizes the read/write head attention weights over time -- shows the NTM learning to write sequentially then read sequentially, effectively discovering the copy algorithm.

## Architecture Details
- **Controller:** Neural network (LSTM or feedforward) that produces read/write head parameters at each time step
- **Memory:** M_t is an N x W matrix (N memory slots, each of dimension W). Fully differentiable.
- **Read operation:** r_t = sum_i w_t(i) * M_t(i) -- weighted sum over memory rows using attention weights w_t. This is identical to the attention mechanism used in Transformers.
- **Write operation:** Two stages: (1) Erase: M_t(i) = M_{t-1}(i) * (1 - w_t(i) * e_t) where e_t is an erase vector; (2) Add: M_t(i) += w_t(i) * a_t where a_t is an add vector. This decomposition allows selective modification.
- **Content-based addressing:** w_t^c(i) = softmax(beta_t * cosine(k_t, M_t(i))) where k_t is a query key and beta_t is a sharpness parameter. This is essentially scaled dot-product attention with cosine similarity.
- **Location-based addressing:** After content addressing, a gate g_t interpolates between content weights and previous weights, then a shift kernel s_t convolves the weights for positional movement, and a sharpening parameter gamma_t focuses the distribution.
- **Connection to Transformers:** The KV cache in autoregressive Transformers is a form of external memory accessed by content-based attention (QK similarity). NTMs made this connection explicit a decade before modern LLMs.
- **Connection to modern retrieval:** RAG systems externalize memory further (retrieval from a database), extending the NTM concept from differentiable memory to discrete retrieval.
- **Legacy:** NTMs led to Differentiable Neural Computers (Graves et al. 2016), Memory Networks (Weston et al. 2015), and influenced the conceptual framing of Transformers' attention as memory lookup.
- **Limitation:** NTMs are hard to train at scale due to the complexity of learning addressing patterns. Transformers solved this by using content-based attention only (no location-based addressing) and fixing the memory to be the sequence itself.
- **Publication venue:** arXiv preprint (2014), widely cited foundational work
