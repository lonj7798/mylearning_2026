<!-- scope: attention mechanism as output pointer for variable-length dictionaries
     deps: [[bahdanau-attention]], [[seq2seq]]
     see-also: [[attention-is-all-you-need]]
-->

# Pointer Networks
- **Core Insight:** Using attention scores directly as output probabilities (pointing to input positions) solves problems where the output dictionary varies with the input, bridging attention from an internal alignment mechanism to a full output layer.
- **Guideline:** When the output must select from the input (copy, reorder, select), use pointer/copy mechanisms rather than fixed-vocabulary softmax.
- **Authors:** Oriol Vinyals, Meire Fortunato, Navdeep Jaitly
- **Year:** 2015
- **URL:** https://arxiv.org/abs/1506.03134
- **Relevant chapters:** Attention mechanism history, copy mechanisms, attention as computation

## Abstract
We introduce a new neural architecture to learn the conditional probability of an output sequence with elements that are discrete tokens corresponding to positions in an input sequence. Such problems cannot be trivially addressed by existing methods such as sequence-to-sequence and Neural Turing Machines, because the number of target classes in each step of the output depends on the length of the input, which is variable. Problems such as sorting variable sized sequences, and various combinatorial optimization problems belong to this class. Our model solves the problem of variable size output dictionaries using a recently proposed mechanism of neural attention. It differs from the previous attention attempts in that, instead of using attention to blend hidden units of an encoder to a context vector at each decoder step, it uses attention as a pointer to select a member of the input sequence as the output. We call this architecture a Pointer Network (Ptr-Net). We show Ptr-Nets can be used to learn approximate solutions to three challenging geometric problems -- finding planar convex hulls, computing Delaunay triangulations, and the planar Travelling Salesman Problem -- using training examples alone. Ptr-Nets not only improve over sequence-to-sequence with input attention, but also allow us to generalize to variable size output dictionaries.

## Key Contributions
- Repurposed the attention mechanism from an internal alignment/context tool into the actual output distribution -- attention weights become output probabilities
- Solved the variable-size output dictionary problem: when the set of valid outputs depends on the input (e.g., selecting a subset or permutation of input elements), a fixed softmax layer cannot work
- Demonstrated that Ptr-Nets can learn approximate solutions to combinatorial optimization problems (convex hull, Delaunay triangulation, TSP) from supervised examples alone
- Directly influenced the development of copy mechanisms (CopyNet, Pointer-Generator Networks) which became essential for summarization and code generation
- Conceptual precursor to the way self-attention in Transformers uses attention to "point" at other positions for information routing

## Key Figures/Tables to Study
- **Figure 1** (Pointer Network architecture): Shows the encoder-decoder with attention weights used directly as output probabilities pointing to input positions. Compare with standard seq2seq attention where attention produces a context vector blended into the decoder.
- **Figure 2** (Convex hull example): Visual demonstration of Ptr-Net selecting the correct subset and ordering of input points to form a convex hull.
- **Table 1** (Results on combinatorial problems): Ptr-Net significantly outperforms seq2seq with attention on all three problems, and the gap widens with problem size -- demonstrating the importance of variable-size output.

## Architecture Details
- **Encoder:** Bidirectional LSTM processes input sequence (x_1, ..., x_n) to produce encoder hidden states (e_1, ..., e_n)
- **Decoder:** LSTM generates decoder hidden states (d_1, ..., d_m) autoregressively
- **Standard attention (Bahdanau):** u_j = v^T tanh(W_1 e_j + W_2 d_i), then a = softmax(u) produces a context vector c = sum(a_j * e_j) which is fed into the decoder. Output comes from a separate softmax over a fixed vocabulary.
- **Pointer mechanism:** u_j = v^T tanh(W_1 e_j + W_2 d_i), then P(output_i = j) = softmax(u_j). The attention distribution IS the output distribution. No fixed vocabulary needed.
- **Key difference:** In standard attention, the softmax output is a blending weight for computing context. In Ptr-Net, the softmax output IS the prediction -- it "points" to an input position.
- **Variable dictionary:** Output vocabulary at each step is {1, ..., n} where n is the input length. This naturally handles variable-length inputs without retraining.
- **Training:** Supervised learning with known optimal solutions (e.g., convex hull computed by classical algorithms). Loss is cross-entropy on the pointer positions.
- **Limitation:** Requires output tokens to come from the input. Later work (Pointer-Generator, Gu et al. 2016) combines pointing with generation from a fixed vocabulary.
- **Legacy for Transformers:** While Transformers don't use pointer mechanisms per se, the conceptual move of treating attention as computation (not just alignment) was influential. Self-attention can be viewed as every position "pointing" to every other position to aggregate information.
- **Publication venue:** NeurIPS 2015
