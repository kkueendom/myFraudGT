# Causal Dual-View Transaction Transformer for Financial Fraud Detection

## Draft Status and Material Passport

- Draft scope: Introduction, Related Work, Method, and Phase 2 Results
- Research question: Can an explicit temporally causal transaction-event graph
  complement FraudGT's account graph and improve fraud detection across AML
  graphs of different scales?
- Paper type: application-oriented empirical graph-learning paper
- Target level: Q3 journal; target venue not yet selected
- Body language: English
- Citation style: IEEE-style placeholders
- Primary source: the local full text of FraudGT and the verified CDVT code
- Model freeze: `dual_view`, `lambda_cons=0`, commit `9038f85`
- Phase 2 source commit: `df12ea6`, a documentation-only descendant of the
  frozen model commit
- Phase 2 portable execution commit: `2fb3333`, with a Git tree identical to
  source commit `df12ea6`
- Follow-up source commit: `34456ab`; portable execution commit: `f7209f2`
- Result status: six-dataset Phase 2 passed its preregistered gate; paired
  representative multi-seed, ablation, sensitivity, and runtime experiments
  are active

## Claim Boundary

In this manuscript, *causal* means that every event-graph edge respects
temporal precedence and that a target transaction cannot receive information
from a later transaction. It does not mean causal-effect identification,
counterfactual treatment estimation, or a statistical causal guarantee.

The final paper must not claim broad state-of-the-art performance. The
six-dataset seed-42 screen has passed, but stability claims remain conditional
on the paired three-seed experiments. All six AML datasets come from one
simulator family, so conclusions are limited to within-family evidence unless
an external dataset is later added.

## 1. Introduction

Financial institutions must inspect large streams of transactions while
fraudulent behavior remains rare, adaptive, and structurally distributed.
Rule-based systems and manual review are difficult to maintain at this scale,
and a transaction that appears ordinary in isolation may become suspicious
when it participates in a short transfer chain, a repeated account interaction,
or a temporally concentrated flow pattern [1], [2]. These properties motivate
models that use both transaction attributes and relational context.

Financial transaction data are naturally represented as directed multigraphs:
accounts are nodes, transactions are directed edges, and multiple transactions
may occur between the same ordered pair of accounts. This representation
preserves fund-flow direction and repeated interactions, but it also creates a
modeling challenge. Most information is attached to transaction edges, whereas
many graph neural networks are organized around node aggregation. FraudGT
addresses this mismatch with an edge-based message-passing gate, an
edge-attribute attention bias, and directed-multigraph enhancements [2], [3].
Its strong accuracy-efficiency trade-off makes it a suitable account-graph
backbone for large AML networks.

The account-graph formulation does not, however, make transaction-to-
transaction transitions explicit. Two transactions are related only
indirectly through their shared account and the account representation produced
by neighborhood aggregation. Consequently, the model must compress the order,
role change, elapsed time, and amount change between successive transactions
into account states before classifying a target edge. This may obscure patterns
whose meaning depends on a specific sequence of events, such as an incoming
transfer followed by an outgoing transfer within a short interval.

Temporal fraud research has examined evolving transaction networks,
time-dependent laundering patterns, and temporal graph models [4]-[6],
[14], [15]. Recent transaction-graph methods also incorporate recency into
edge message passing [18] or add a line-graph view to an account graph [19].
A remaining practical question in the FraudGT setting is how to construct a
leakage-controlled transaction-event graph with typed role transitions and
fuse its event states with a strong account-graph Transformer. Treating history
as a single pooled vector loses explicit event transitions. Replacing the
account graph entirely with an event graph may lose complementary
account-level structure. Appending a scalar correction to the final logit,
meanwhile, does not change transaction representation learning and provides
only a weak test of whether the two graph views are complementary.

This study proposes the Causal Dual-View Transaction Transformer (CDVT). CDVT
retains FraudGT as an account-view encoder and constructs a second graph in
which each transaction is an event node. For a target transaction, directed
event edges connect earlier events to later events only when they share an
account. Each edge records elapsed time, amount dynamics, categorical changes,
and the role transition of the shared account. A relation-aware temporal
Transformer propagates information along this event directed acyclic graph.
The resulting event states are not reduced to one fixed history vector.
Instead, the account-view representation queries the event states through
multi-head cross-attention before classification.

The design separates two complementary questions. The account view asks how a
transaction is situated in the sampled financial network. The event view asks
which admissible historical transactions lead to the target and how account
roles and transaction attributes change along those transitions. Cross-view
attention then decides which event states are relevant to the current
account-view representation. Both encoders and the fusion classifier are
trained jointly, so the event branch is part of representation learning rather
than a post-hoc decoder correction.

The study makes two methodological contributions:

1. It introduces a temporally causal transaction-event graph that represents
   transaction order, shared-account roles, time gaps, amount changes,
   currency changes, and payment-format changes without exposing a target to
   future transactions.
2. It introduces representation-level dual-view fusion in which a FraudGT
   account representation queries the full local event context through
   cross-attention before transaction classification.

The empirical study is designed around six AML graph settings from the
realistic synthetic financial-transaction family [13] and the public FraudGT
dynamic-sampling protocol. The final evaluation will report
validation-selected test F1 as the primary metric, raw-best test F1 only as a
supplementary diagnostic, three real random seeds on representative Small,
Medium, and Large settings, normal/shuffled/off event interventions, component
ablations, history-size sensitivity, and computational cost. This design tests
both predictive performance and whether the learned event context is aligned
with the target transaction.

## 2. Related Work

### 2.1 Graph-Based Financial Fraud Detection

Financial fraud detection includes anti-money laundering, illicit-account
detection, credit-card fraud, and suspicious-transaction classification [1].
Event-level methods classify transactions from their attributes, while
sequence methods search for suspicious behavior over time. Graph methods add
the relational structure among accounts and transactions. Message-passing
neural networks provide a general framework for learning from such structure
[7], and neighborhood sampling makes inductive graph learning feasible on
large networks [8]. Attention-based aggregation can further assign different
importance to different neighbors [9].

Transaction graphs differ from many standard graph-learning benchmarks.
They are directed multigraphs with rich edge attributes, repeated edges, severe
class imbalance, and temporal order. Directed-multigraph enhancements such as
reverse message passing, port numbering, and ego identifiers improve the
ability of GNNs to distinguish directions and parallel transactions [3].
FraudGT combines these enhancements with a local graph Transformer. Its
edge-based message-passing gate controls which feature channels pass between
neighboring accounts, while its edge-based attention bias lets transaction
attributes influence neighbor weighting [2].

CDVT builds on rather than replaces this account-view representation. The
FraudGT encoder is retained with the same hidden size, neighborhood sampling,
and optimization protocol used by the matched account-only model. The new
question is whether transaction transitions contain information that remains
implicit after account-level aggregation.

Other AML graph constructions operate at different units of analysis.
LaundroGraph represents customers and transactions as a bipartite graph and
learns self-supervised link-prediction representations from real banking data
[16]. Elliptic2 frames blockchain laundering as subgraph classification and
provides labeled illicit shapes within a much larger background graph [17].
LineMVGNN combines payment and receipt message passing with a line-graph view
of transactions [19]. These studies show that transaction nodes, subgraphs,
and alternative graph views are useful AML abstractions, but they do not test
the specific combination of a FraudGT account encoder, a temporally directed
event DAG, typed shared-account role transitions, and query-based
representation fusion used here.

### 2.2 Temporal and Event-Centric Fraud Modeling

Fraud patterns often depend on both topology and timing. Prior studies have
examined topology-and-spike behavior, smurf-based laundering in time-evolving
networks, real-time fraud detection on evolving graphs, and causal-temporal
GNNs for credit-card fraud [4]-[6], [10]. These studies support the broader
premise that time is not merely an edge attribute: temporal order can define
which interactions form a meaningful behavioral pattern.

General temporal graph learning provides two relevant design families. TGAT
uses attention with functional time encoding to aggregate temporal-topological
neighborhoods [14], while TGN treats dynamic graphs as sequences of timed
events and combines memory with graph operators [15]. UTG later formalizes a
common view of snapshot-based and event-based temporal models [20]. CDVT does
not maintain a global recurrent memory or convert the benchmark into graph
snapshots. It constructs a bounded event DAG for each target so that temporal
precedence can be audited directly under mini-batch edge classification.

An account graph and an event graph encode different adjacency relations. In
an account graph, two accounts are adjacent when a transaction connects them.
In an event graph, two transactions are adjacent when they share an account
and satisfy a temporal ordering constraint. The second representation exposes
incoming-to-outgoing, outgoing-to-incoming, shared-source, and
shared-destination transitions directly. This distinction is important for
AML because a rapid role change at an intermediary account may be more
informative than either transaction considered independently.

CDVT uses a target-conditioned local event graph rather than a global dense
temporal graph. For each target, it retrieves only the most recent admissible
events per encountered account, limits expansion depth, and caps graph size.
This keeps the temporal view compatible with mini-batch training on large
transaction graphs. Temporal precedence is enforced during construction, so
the event encoder cannot aggregate from a transaction later than the target.

TeMP-TraG addresses time in financial multigraphs by weighting transaction
edges according to temporal proximity during message passing [18]. CDVT
instead makes transactions the event nodes and models pairwise transitions
with elapsed time, amount dynamics, shared-account role, currency change, and
payment-format change. The methods therefore share a motivation but encode
different temporal objects.

### 2.3 Graph Transformers and Cross-View Fusion

Transformers use content-dependent attention to select relevant context [11].
Graph Transformers adapt this mechanism by adding graph structure, positional
information, or relation bias and by restricting attention when full global
attention is too expensive [2], [12]. FraudGT applies attention to direct
neighbors in a sampled account graph. CDVT applies a second, relation-aware
attention mechanism to the transaction-event graph, where relation embeddings
and transition features modify both keys and values.

Multi-view learning is useful when no single graph construction preserves all
task-relevant relationships. A simple concatenation or addition assumes that
the same event summary is equally suitable for every account representation.
CDVT instead uses the projected account representation as a query and the
event states as keys and values. This produces a target-specific context while
retaining a residual path from the account view. Fusion occurs before the
classifier, allowing gradients from the fraud objective to shape the account
encoder, event encoder, attention weights, and classifier jointly.

LineMVGNN is the closest multi-view AML comparison because it augments a
transaction account graph with a line graph [19]. Its multi-view design
propagates payment and receipt information and combines view-level
representations. CDVT differs in four respects: event edges must follow target
time, transition relations encode the shared account's source/destination role,
the event graph is target-conditioned and bounded, and the FraudGT account
state queries individual event states through cross-attention. These
differences define CDVT's contribution more precisely than a broad claim to be
the first multi-view transaction model.

### 2.4 Research Gap and Positioning

FraudGT demonstrates that directed account topology and transaction-edge
attributes can be modeled accurately and efficiently [2]. Temporal graph and
fraud studies demonstrate that evolving behavior is relevant [4]-[6],
[14], [15], [18], while LineMVGNN demonstrates a line-graph-assisted
multi-view AML model [19]. The narrower gap addressed here is an explicit,
leakage-controlled transaction-to-transaction graph used jointly with the
FraudGT account graph, with typed role transitions and target-specific
cross-attention. CDVT addresses this gap with two deliberately limited
additions: a causal event encoder and representation-level cross-view
attention.

The method is not presented as a new causal-inference estimator, a generic
temporal-graph foundation model, or a collection of decoder heuristics. Its
scope is an application-oriented test of whether explicit event transitions
provide complementary information to a strong graph-fraud backbone. Sampling
consistency was evaluated during screening but reduced validation performance;
it is therefore excluded from the frozen model and is not claimed as a
contribution.

## 3. Method

### 3.1 Problem Formulation

Let the financial account graph be

\[
\mathcal{G}_{A}=(\mathcal{V},\mathcal{E},\mathbf{X},\mathbf{Z}),
\]

where \(\mathcal{V}\) is the set of accounts and every transaction
\(e_i=(u_i,v_i,t_i,\mathbf{x}_i)\in\mathcal{E}\) is a directed edge from
source account \(u_i\) to destination account \(v_i\). The timestamp is
\(t_i\), and \(\mathbf{x}_i\) contains the amount, currency, payment format,
and other encoded transaction attributes. Parallel edges are retained.
The binary label \(y_i\in\{0,1\}\) indicates whether transaction \(e_i\) is
illicit.

For a target transaction \(e_i\), the model estimates

\[
\hat p_i=P(y_i=1\mid \mathcal{G}_{A},e_i,\mathcal{H}_i),
\]

where \(\mathcal{H}_i=\{e_j:e_j\prec e_i\}\) denotes admissible transaction
history. Equal timestamps are ordered by immutable global edge ID, and only an
edge ID smaller than the target ID is admissible. The latter rule makes the
ordering deterministic while preventing same-time future leakage.

The data are split chronologically using the original FraudGT split. In the
transductive online interpretation, a validation or test transaction may use
transactions from earlier split prefixes and earlier transactions within its
own prefix, but never a later transaction. Training, validation, and test
loaders retain random dynamic neighborhood sampling.

### 3.2 CDVT Overview

CDVT contains an account view and an event view. The account view applies the
FraudGT encoder to a dynamically sampled neighborhood around the target batch:

\[
\mathbf{H}^{A},\mathbf{R}^{A}
=f_{A}(\mathcal{G}_{A}^{B};\theta_A).
\]

For target \(e_i=(u_i,v_i)\), its account-view representation is

\[
\mathbf{a}_i=
\mathbf{h}^{A}_{u_i}\Vert
\mathbf{h}^{A}_{v_i}\Vert
\mathbf{r}^{A}_{i}.
\]

The event view constructs a target-conditioned transaction graph
\(\mathcal{G}^{E}_i\) and encodes all of its event nodes:

\[
\mathbf{S}_i=f_E(\mathcal{G}^{E}_i;\theta_E).
\]

The projected account state queries these event states through cross-attention.
The fused state is passed to an MLP classifier. All trainable components are
optimized jointly under the fraud-classification objective.

Figure 1 summarizes the complete CDVT pipeline, from leakage-controlled event
retrieval to representation-level dual-view fusion.

![Figure 1. Architecture of the Causal Dual-View Transaction Transformer.](CDVT_Figure/cdvt_architecture_render.png)

### 3.3 Causal Transaction-Event Graph

#### 3.3.1 Target-Conditioned Event Retrieval

Each transaction becomes a node in the event view. Starting from the source
and destination accounts of target \(e_i\), CDVT retrieves the \(K\) most
recent incident transactions that precede \(e_i\) for each encountered
account. The accounts touched by those events form the next expansion
frontier. Expansion continues for \(H\) hops or until at most \(M\) event nodes
have been selected. The target node is appended after the chronologically
ordered context nodes.

Define the recent-history operator

\[
\mathcal{R}_{K}(a,i)=\operatorname{TopK}_{e_j\prec e_i}
\{e_j:a\in\{u_j,v_j\}\},
\]

where \(\prec\) is the timestamp-edge-ID order. With
\(\mathcal{A}^{(0)}_i=\{u_i,v_i\}\), one expansion step retrieves

\[
\mathcal{C}^{(h+1)}_i=
\mathcal{C}^{(h)}_i\cup
\bigcup_{a\in\mathcal{A}^{(h)}_i}\mathcal{R}_{K}(a,i),
\]

and the next frontier contains accounts incident to the newly retrieved
events. Retrieval is label-free. The implementation uses \(K=4\), \(H=2\),
and \(M=48\) in the frozen model.

#### 3.3.2 Directed Event Transitions

Within the selected event set, CDVT adds a directed edge \(e_j\rightarrow e_k\)
when (i) the two transactions share at least one account, (ii)
\(e_j\prec e_k\), and (iii) \(e_j\) is among the \(K\) most recent selected
predecessors of \(e_k\) for that shared account. The resulting graph is a
directed acyclic graph because every edge follows the strict event order.
When two events share both endpoint accounts, parallel transitions may be
retained because each shared-account role instance carries distinct semantics.

For a shared account \(a\), its role in a transaction is outgoing (O) when it
is the source and incoming (I) when it is the destination. Each transition is
assigned one of four relation types:

\[
r_{jk}\in\{\mathrm{O\!\rightarrow O},
\mathrm{I\!\rightarrow I},
\mathrm{O\!\rightarrow I},
\mathrm{I\!\rightarrow O}\}.
\]

These respectively represent shared-source, shared-destination,
outgoing-to-incoming, and incoming-to-outgoing behavior. Relation type is
embedded separately in the key and value pathways of the temporal attention
layer.

#### 3.3.3 Leakage Control

The event index is built once from immutable transaction fields, but every
query is filtered relative to its target. For any node \(e_j\) in
\(\mathcal{G}^{E}_i\), \(e_j\preceq e_i\). For any event edge
\(e_j\rightarrow e_k\), \(e_j\prec e_k\preceq e_i\). Therefore no directed
path ending at the target can contain an event later than the target. Labels
are not used to retrieve nodes, create event edges, or construct transition
features.

### 3.4 Event Features

The immutable raw event fields are

\[
\boldsymbol{\rho}_j=(t_j,\widetilde m_j,c_j,p_j),
\]

where \(\widetilde m_j\) is the amount standardized using training-prefix
statistics, \(c_j\) is the currency ID, and \(p_j\) is the payment-format ID.
The event-node numeric input relative to target \(e_i\) is

\[
\boldsymbol{\nu}_{j|i}=\left[
\frac{\log(1+t_j)}{20},
\frac{\operatorname{clip}(\widetilde m_j,-10,10)}{5},
\frac{\log(1+t_i-t_j)}{20}
\right].
\]

Currency and payment format use learned embeddings. A learned binary embedding
distinguishes the target event from context events, and a sinusoidal encoding
represents the lag \(t_i-t_j\). The initial event state is

\[
\mathbf{s}^{(0)}_{j|i}=\operatorname{Dropout}\!\left(
\operatorname{LN}\!\left(
\operatorname{GELU}\!\left(
W_n\boldsymbol{\nu}_{j|i}
+W_c[\operatorname{Emb}_c(c_j)\Vert\operatorname{Emb}_p(p_j)]
+\operatorname{Emb}_{tar}(\mathbb{I}[j=i])
+\operatorname{PE}(t_i-t_j)
\right)\right)\right).
\]

For transition \(e_j\rightarrow e_k\), CDVT constructs a ten-dimensional
feature vector containing:

1. \(\log(1+t_k-t_j)\);
2. the log ratio of the absolute normalized amounts;
3. normalized signed amount change;
4. a four-dimensional one-hot role-transition vector;
5. indicators for currency change and payment-format change; and
6. an indicator that the shared account changes between incoming and outgoing
   roles.

More explicitly, the amount terms are

\[
q_{jk}=\log\frac{|\widetilde m_k|+1}{|\widetilde m_j|+1},
\qquad
d_{jk}=\frac{\widetilde m_k-\widetilde m_j}
{|\widetilde m_j|+1}.
\]

These transition features describe how two connected events differ; they are
not fraud scores or hand-set prediction rules.

### 3.5 Relation-Aware Temporal Event Encoder

CDVT applies \(L_E\) relation-aware temporal attention layers to the event
graph. For transition \(e_j\rightarrow e_k\) at layer \(\ell\), head \(h\)
uses the destination event as query and the preceding event as key and value:

\[
\mathbf{q}^{\ell,h}_{jk}=W_Q^{\ell,h}\mathbf{s}^{\ell-1}_{k},
\]

\[
\mathbf{k}^{\ell,h}_{jk}=
W_K^{\ell,h}\mathbf{s}^{\ell-1}_{j}
+U_K^{\ell,h}[r_{jk}]
+A_K^{\ell,h}\boldsymbol{\xi}_{jk},
\]

\[
\mathbf{v}^{\ell,h}_{jk}=
W_V^{\ell,h}\mathbf{s}^{\ell-1}_{j}
+U_V^{\ell,h}[r_{jk}]
+A_V^{\ell,h}\boldsymbol{\xi}_{jk},
\]

where \(\boldsymbol{\xi}_{jk}\) is the transition-feature vector and
\(U_K,U_V\) are learned relation embeddings. Attention is normalized over
incoming event edges of \(e_k\):

\[
\alpha^{\ell,h}_{jk}=
\operatorname{softmax}_{j\in\mathcal{N}^{-}(k)}
\left(
\frac{(\mathbf{q}^{\ell,h}_{jk})^\top
\mathbf{k}^{\ell,h}_{jk}}{\sqrt{d_h}}
\right).
\]

The aggregated message and residual updates are

\[
\widetilde{\mathbf{s}}^{\ell}_{k}=
W_O^{\ell}\mathop{\Vert}_{h=1}^{H_E}
\sum_{j\in\mathcal{N}^{-}(k)}
\alpha^{\ell,h}_{jk}\mathbf{v}^{\ell,h}_{jk},
\]

\[
\bar{\mathbf{s}}^{\ell}_{k}=
\operatorname{LN}\left(
\mathbf{s}^{\ell-1}_{k}
+\operatorname{Dropout}(\widetilde{\mathbf{s}}^{\ell}_{k})
\right),
\]

\[
\mathbf{s}^{\ell}_{k}=
\operatorname{LN}\left(
\bar{\mathbf{s}}^{\ell}_{k}
+\operatorname{Dropout}(\operatorname{FFN}^{\ell}
(\bar{\mathbf{s}}^{\ell}_{k}))
\right).
\]

The frozen model uses \(L_E=2\), \(H_E=4\), and hidden dimension \(d=64\).
The output \(\mathbf{S}_i\) contains every context-event state and the target-
event state. Keeping the full set allows the account view to attend to
different events for different targets.

### 3.6 Representation-Level Dual-View Fusion

The account representation is first projected into the event hidden space:

\[
\mathbf{q}_i=
\operatorname{LN}\left(
\operatorname{GELU}(W_A\mathbf{a}_i+\mathbf{b}_A)
\right).
\]

CDVT then applies multi-head cross-attention with one account query and all
event states as keys and values:

\[
\mathbf{c}_i=W_C\operatorname{MHA}
(\mathbf{q}_i,\mathbf{S}_i,\mathbf{S}_i).
\]

Padding masks ensure that event states from different targets do not interact
after batching. The residual fused representation is

\[
\mathbf{f}_i=\operatorname{LN}(\mathbf{q}_i+\mathbf{c}_i).
\]

A two-layer classifier produces the final logit and probability:

\[
z_i=\mathbf{w}_2^\top
\operatorname{Dropout}\left(
\operatorname{GELU}(W_1\mathbf{f}_i+\mathbf{b}_1)
\right)+b_2,
\qquad
\hat p_i=\sigma(z_i).
\]

This fusion differs from a decoder-side scalar residual. The event context
modifies a hidden transaction representation before classification, and the
attention distribution can select different historical events for each
account-view query. The FraudGT encoder, event encoder, cross-attention, and
classifier are trained end to end.

### 3.7 Learning Objective

The frozen model uses weighted binary cross-entropy to address class imbalance:

\[
\mathcal{L}_{fraud}=-\frac{1}{|B|}
\sum_{i\in B}w_{y_i}
\left[y_i\log\hat p_i+(1-y_i)\log(1-\hat p_i)\right],
\]

with class weights \(w_0=1\) and \(w_1=6\). Sampling consistency was screened
through an additional Jensen-Shannon term between two sampled views of aligned
targets, but it reduced validation F1 at both screening scales. The final CDVT
therefore optimizes only \(\mathcal{L}_{fraud}\).

### 3.8 Training and Evaluation Protocol

We use the six temporally split AML transaction graphs released with the
FraudGT evaluation [2]. Each scale has a high-illicit-rate (HI) and
low-illicit-rate (LI) setting:

| Dataset | Accounts | Transactions | Illicit rate | Time span | Train/val/test |
|---|---:|---:|---:|---:|---:|
| AML Small-HI | 515,088 | 5,078,345 | 0.102% | 10 days | 64/19/17 |
| AML Small-LI | 705,907 | 6,924,049 | 0.051% | 10 days | 64/19/17 |
| AML Medium-HI | 2,077,023 | 31,898,238 | 0.110% | 16 days | 61/17/22 |
| AML Medium-LI | 2,032,095 | 31,251,483 | 0.051% | 16 days | 61/17/22 |
| AML Large-HI | 2,116,168 | 179,702,229 | 0.124% | 97 days | 60/20/20 |
| AML Large-LI | 2,070,980 | 176,066,557 | 0.057% | 97 days | 60/20/20 |

Transactions are ordered by timestamp before splitting. CDVT does not change
these splits. Although the six settings cover three graph scales and two
illicit rates, they originate from one synthetic AML simulator family; this
limits claims about transfer to independently collected financial networks.

The evaluation follows the public FraudGT dynamic-random sampling behavior.
Training, validation, and test use `LinkNeighborLoader` with `shuffle=True`.
No fixed validation or test target panel is introduced, no dedicated
evaluation generator is used, and sampler RNG state is not restored before an
evaluation. Each epoch contains 256 training iterations; validation and test
each contain 256 iterations. The batch size is 2048 and the maximum budget is
500 epochs.

At each evaluation event, a decision threshold is selected to maximize
validation F1 and is applied unchanged to the corresponding test predictions.
The formal checkpoint is the epoch with the highest validation F1, and the
primary reported value is the test F1 from that epoch. The maximum test F1
observed over epochs is recorded only as a supplementary raw-best diagnostic
and is compared only with the corresponding raw-best baseline column.

PE-FraudGT is the primary published baseline because CDVT retains the same
Ports + Ego ID account-view architecture without reverse message passing.
Multi-FraudGT, the strongest overall variant in FraudGT Table 2, is reported as
a stricter published reference. The previously developed A2 decoder is
reported as an internal strong comparator. Published FraudGT results contain
only validation-selected five-run means, so raw-best CDVT results are compared
only with the A2 raw-best column and never across selection rules.

Because dynamic sampling introduces evaluation variation, an absolute F1
change smaller than 0.005 is labeled as potentially within sampling variation.
The final paper will emphasize real three-seed mean and standard deviation on
representative datasets rather than duplicating a best seed.

Before predictive evaluation, 42 CDVT engineering tests verified causal
ordering, latest-\(K\) history, relation features, target-ID alignment,
future/label leakage, intervention semantics, and gradient propagation. A real
Small-LI GPU smoke then confirmed nonzero gradients in the account encoder,
event encoder, and fusion module. On a dynamically sampled batch containing
both classes, 24 fixed-batch optimization steps reduced evaluation loss from
0.67856 to 0.04245. This is reported only as a trainability check, not as
generalization evidence.

### 3.9 Variants and Mechanism Tests

The core ablation variants isolate the two graph views and their fusion:

| Variant | Account encoder | Event encoder | Fusion |
|---|---|---|---|
| Account-only | FraudGT | No | Original FraudGT head |
| Event-only | No | Causal event Transformer | Target-event MLP |
| CDVT | FraudGT | Causal event Transformer | Cross-attention |
| CDVT + consistency | FraudGT | Causal event Transformer | Cross-attention plus JS loss |

The consistency variant is retained only as a negative screening result and is
not part of the final architecture. An additive diagnostic replaces
cross-attention with the sum of the projected account state and target-event
state; it tests whether access to event features alone explains the effect.

Three event conditions test alignment:

- **Normal** uses the observed causal event graph.
- **Shuffled** preserves graph sizes and target nodes but permutes context
  event features and transition attributes, breaking their alignment with the
  target.
- **Off** removes all historical context nodes and event transitions while
  retaining the target event itself.

Normal performance above both shuffled and off supports the interpretation
that aligned historical transitions matter, rather than merely adding
parameters or target-event attributes. The final experiment package also
removes relation types and evaluates one alternative value of \(K\).

### 3.10 Computational Scope

For target \(e_i\), the event graph contains at most \(M\) nodes. Each event
retains at most \(K\) recent predecessors per shared endpoint, so event-edge
construction is bounded by \(O(MK)\) up to duplicated shared-account cases.
With \(m_i\) event edges, one temporal layer costs approximately
\(O(m_i d + M d^2)\). Cross-attention uses one account query over at most
\(M\) event states and costs \(O(Md+d^2)\) per target. The account-view
FraudGT computation is unchanged.

The immutable incident index is built once, and target-conditioned event graphs
are cached by target ID and graph-construction settings. The final paper will
report measured parameter count, peak GPU memory, training time, and inference
time from experiment manifests rather than relying only on asymptotic cost.

## 4. Results

### 4.1 Six-Dataset Main Results

Table 2 reports the frozen CDVT model with seed 42 under the public
dynamic-random sampling protocol. PE-FraudGT is the primary comparison because
it is the direct account-view parent of CDVT. Multi-FraudGT is included as the
stronger published FraudGT reference. The published FraudGT entries are
five-run means, whereas the CDVT entries in this table are seed-42 results;
paired multi-seed comparisons are reported separately once complete.

| Dataset | PE-FraudGT | Multi-FraudGT | CDVT | Delta vs PE | Delta vs Multi |
|---|---:|---:|---:|---:|---:|
| AML Small-LI | 0.45810 | 0.47010 | 0.43798 | -0.02012 | -0.03212 |
| AML Small-HI | 0.76410 | 0.76130 | 0.76940 | +0.00530 | +0.00810 |
| AML Medium-LI | 0.43530 | 0.44060 | 0.44711 | +0.01181 | +0.00651 |
| AML Medium-HI | 0.74220 | 0.75930 | 0.77301 | +0.03081 | +0.01371 |
| AML Large-LI | 0.30440 | 0.37430 | 0.33803 | +0.03363 | -0.03627 |
| AML Large-HI | 0.68640 | 0.73340 | 0.78793 | +0.10153 | +0.05453 |
| Mean | 0.56508 | 0.58983 | 0.59224 | +0.02716 | +0.00241 |

CDVT exceeds PE-FraudGT on five of six datasets and improves mean
validation-selected test F1 by 0.02716, satisfying the preregistered Phase 2
gate of at least four wins and a positive mean delta. The largest gain occurs
on Large-HI (+0.10153), followed by Large-LI (+0.03363) and Medium-HI
(+0.03081). Small-LI is the only loss (-0.02012). These results support a
scale-dependent rather than universal claim: explicit event transitions are
most promising on the larger or higher-illicit-rate settings in this screen.

Against Multi-FraudGT, CDVT wins four of six settings and has a small positive
mean delta of 0.00241. This aggregate difference is below the predefined 0.005
sampling-variation threshold and is not treated as evidence of stable overall
superiority. CDVT nevertheless shows clear gains on Large-HI and both
medium-scale settings, while remaining below Multi-FraudGT on Small-LI and
Large-LI.

The internal A2 comparator has mean validation-selected F1 0.59329, compared
with 0.59224 for CDVT, a difference of -0.00104. CDVT is therefore not claimed
to dominate A2 overall. It does, however, exceed A2 by 0.03695 on Large-LI and
0.05896 on Large-HI. The remaining paired experiments test whether this
large-scale advantage persists across independent seeds.

### 4.2 Event-Context Intervention

The normal, shuffled, and off conditions reuse the same trained model and
validation-selected threshold. Shuffling breaks alignment between a target and
its historical event context, while the off condition removes historical
context and retains only the target event.

| Event condition | Small-LI | Medium-LI | Large-LI | Mean |
|---|---:|---:|---:|---:|
| Normal causal event graph | 0.43798 | 0.44711 | 0.33803 | 0.40771 |
| Shuffled event graph | 0.09350 | 0.12058 | 0.01596 | 0.07668 |
| Event graph off | 0.27294 | 0.03012 | 0.00000 | 0.10102 |

Normal event context outperforms both interventions on all three
representative scales. The consistent degradation after shuffling indicates
that the model uses target-aligned historical information rather than merely
benefiting from additional event-branch parameters. The off result further
shows that the target-event attributes alone do not explain the full CDVT
prediction. These are mechanism diagnostics, not causal-effect estimates.

### 4.3 Pending Robustness and Efficiency Evidence

The paired follow-up uses independent seeds 42, 43, and 44 for both
FraudGT/account-only and CDVT on Small-LI, Medium-LI, and Large-LI. It also
completes account-only and event-only controls, removes relation types,
evaluates \(K=2\) against the frozen \(K=4\), and benchmarks normal-only
inference. The follow-up is running from portable commit `f7209f2`. Mean,
sample standard deviation, same-seed paired deltas, and computational overhead
will be inserted only from final manifests.

## 5. Discussion

### 5.1 Complementarity of Account and Event Views

The Phase 2 results support the central premise that an explicit event graph
can complement FraudGT's account graph. CDVT changes the representation
available to the classifier rather than appending a correction to an existing
FraudGT logit. The account encoder retains directed account topology, while
the event encoder exposes transaction order, role transitions, time gaps, and
attribute changes. The five wins against PE-FraudGT indicate that this second
adjacency relation can add useful information across several AML settings.

The intervention results strengthen this interpretation. If improvement came
only from a larger classifier or from target-transaction attributes, breaking
the correspondence between the target and its history would not be expected
to reduce F1 consistently. Instead, shuffled and off context perform below
normal context on Small-LI, Medium-LI, and Large-LI. This pattern indicates
that CDVT uses aligned historical events. It does not establish that every
encoded transition is necessary, which is why the no-relation and history-size
controls remain important.

### 5.2 Scale-Dependent Performance

CDVT's gains are not uniform. The strongest improvement occurs on Large-HI,
and both large settings improve over the internal A2 comparator, whereas
Small-LI falls below both published FraudGT references. One plausible
explanation is that explicit event transitions become more useful when a graph
covers a longer time span or contains enough transaction history to expose
repeated role changes. The large datasets cover 97 days, compared with 10 days
for the small datasets. This explanation remains a hypothesis because scale,
time span, illicit rate, and generated transaction patterns vary together in
the available benchmark family.

The Small-LI loss also shows why CDVT should not be described as a universal
replacement for FraudGT. Event retrieval can introduce irrelevant history
when target-aligned transitions are sparse or weak. Cross-attention can learn
to down-weight context, but the current model does not include an explicit
per-target abstention mechanism. Adding another scalar decoder gate would make
the architecture resemble the unsuccessful post-hoc routes examined earlier
and would weaken the representation-learning contribution. A more appropriate
future direction is adaptive event retrieval within the event encoder, tested
prospectively rather than tuned to Small-LI test outcomes.

### 5.3 Relation to the Internal A2 Comparator

The seed-42 mean of CDVT is within 0.00104 of the initial A2 mean. The current
evidence therefore supports different strengths rather than overall dominance.
A2 remains stronger on Medium-LI and the two small settings, while CDVT is
stronger on both large settings. CDVT nevertheless provides a clearer
methodological contribution: it defines a second transaction-to-transaction
graph, learns relation-aware temporal states jointly with FraudGT, and fuses
the two views before classification. The contribution is consequently not
that a more complicated decoder raises a score, but that a distinct graph
representation can recover temporal transition information that is implicit
in the account graph.

This distinction matters for an application-oriented paper, but architectural
clarity does not compensate for unstable performance. The paired three-seed
experiment is the deciding robustness test. If the large-scale gains persist
while the overall paired mean remains non-negative, the evidence will support
a targeted claim about large AML graphs. If they do not persist, the paper
must report Phase 2 as a stochastic screen rather than as stable improvement.

### 5.4 Practical Considerations

The frozen CDVT model contains 301,147 trainable parameters. Its observed peak
GPU memory ranges from approximately 1.94 GB on the small settings to 7.19 GB
on Large-HI under the formal training configuration. These values show that
the event view fits on the available 22 GB GPUs, but they do not establish an
efficiency advantage over FraudGT. CDVT additionally constructs and encodes a
target-conditioned event graph, so higher latency is expected. The dedicated
normal-only benchmark will quantify parameter, memory, and inference-time
ratios under matched checkpoints and batches.

## 6. Limitations and Future Work

The first limitation is dataset scope. All six AML settings originate from
the same realistic synthetic transaction generator [13]. They vary in scale,
illicit rate, and time span, but they do not establish transfer to a different
simulator, financial institution, country, or operational labeling process.
External validation on independently collected data would materially
strengthen the paper.

Second, the six-dataset table currently contains one CDVT seed compared with
published FraudGT five-run means. Dynamic neighborhood sampling changes the
evaluated target set and sampled context, so individual results can fluctuate.
The matched three-seed experiment reduces this limitation on three
representative settings, but it does not turn all six seed-42 comparisons into
paired statistical tests.

Third, *causal* in CDVT describes temporal information flow, not causal
inference. The construction prevents a target from receiving later events,
but it does not identify treatment effects, remove all confounding, or support
counterfactual claims about fraud. A different name or an explicit qualifier
may be preferable if a target venue reserves causal terminology for
identification-based methods.

Fourth, the event graph depends on fixed retrieval choices: \(K=4\),
two history-expansion hops, and at most 48 events. These settings make training
tractable but may not be optimal for every graph scale or transaction regime.
The preregistered \(K=2\) control tests one sensitivity axis; broader adaptive
retrieval remains future work.

Finally, target-conditioned construction and caching add preprocessing,
memory, and latency costs. Future implementations could maintain streaming
per-account event indices, share event subgraphs across nearby targets, or
distill the event encoder after training. Such optimizations should preserve
the temporal-precedence invariant and be evaluated under the same dynamic
sampling protocol.

## 7. Conclusion

This study introduces CDVT, a dual-view extension of FraudGT that combines an
account graph with a temporally ordered transaction-event graph. A
relation-aware event Transformer encodes shared-account transitions, and the
FraudGT account representation queries those event states through
representation-level cross-attention. The design adds transaction-level
temporal structure without replacing the original account topology or
returning to post-hoc decoder corrections.

Under the frozen seed-42 protocol, CDVT exceeds PE-FraudGT on five of six AML
settings and improves mean validation-selected test F1 by 0.02716. Normal
event context also outperforms shuffled and off interventions on all three
representative scales. The strongest gains occur on the large datasets, while
Small-LI remains a clear failure case. These findings support a scoped claim
that explicit event transitions can complement FraudGT, particularly on
larger AML graphs. Final claims about stability, component contribution, and
computational trade-offs remain conditional on the active paired multi-seed,
ablation, sensitivity, and runtime experiments.

## Drafting Notes for the Remaining Paper

The abstract, final discussion, and conclusion must be finalized only after
the paired follow-up is complete. The remaining evidence package is:

1. real seeds 42, 43, and 44 on Small-LI, Medium-LI, and Large-LI;
   FraudGT/account-only and CDVT are both run at each seed so that the
   robustness table reports same-seed paired deltas;
2. account-only, event-only, and CDVT ablations on representative scales;
3. a no-relation test and one alternative \(K\);
4. parameter, memory, training-time, and inference-time measurements; and
5. an explicit limitation that all six AML settings share one simulator
   family.

## Provisional References

The FraudGT full text [2] was inspected locally. Bibliographic metadata for all
entries was independently checked against Crossref, arXiv, DBLP, PMLR,
OpenReview indexing, or the official NeurIPS proceedings. This confirms that
the cited works and metadata exist; a final claim-to-source alignment audit is
still required before submission.

[1] W. Hilal, S. A. Gadsden, and J. Yawney, "Financial Fraud: A Review of
Anomaly Detection Techniques and Recent Advances," *Expert Systems with
Applications*, vol. 193, art. 116429, 2022, doi:
10.1016/j.eswa.2021.116429.

[2] J. Lin, X. Guo, Y. Zhu, S. Mitchell, E. Altman, and J. Shun, "FraudGT: A
Simple, Effective, and Efficient Graph Transformer for Financial Fraud
Detection," in *Proc. 5th ACM International Conference on AI in Finance
(ICAIF)*, 2024, doi: 10.1145/3677052.3698648.

[3] B. Egressy, L. von Niederhausern, J. Blanusa, E. Altman, R. Wattenhofer,
and K. Atasu, "Provably Powerful Graph Neural Networks for Directed
Multigraphs," in *Proc. AAAI Conference on Artificial Intelligence*, vol. 38,
pp. 11838-11846, 2024, doi: 10.1609/aaai.v38i10.29069.

[4] M. Starnini et al., "Smurf-Based Anti-Money Laundering in Time-Evolving
Transaction Networks," in *ECML PKDD*, pp. 171-186, 2021, doi:
10.1007/978-3-030-86514-6_11.

[5] J. Jiang et al., "Spade: A Real-Time Fraud Detection Framework on Evolving
Graphs," *Proceedings of the VLDB Endowment*, vol. 16, no. 3, pp. 461-469,
2022, doi: 10.14778/3570690.3570696.

[6] Y. Duan et al., "CaT-GNN: Enhancing Credit Card Fraud Detection via Causal
Temporal Graph Neural Networks," arXiv:2402.14708, 2024.

[7] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl,
"Neural Message Passing for Quantum Chemistry," in *Proc. International
Conference on Machine Learning*, pp. 1263-1272, 2017.

[8] W. Hamilton, Z. Ying, and J. Leskovec, "Inductive Representation Learning
on Large Graphs," in *Advances in Neural Information Processing Systems*,
vol. 30, pp. 1024-1034, 2017.

[9] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, and
Y. Bengio, "Graph Attention Networks," in *International Conference on
Learning Representations*, 2018.

[10] S. Liu, B. Hooi, and C. Faloutsos, "HoloScope: Topology-and-Spike Aware
Fraud Detection," in *Proc. ACM International Conference on Information and
Knowledge Management*, pp. 1539-1548, 2017, doi:
10.1145/3132847.3133018.

[11] A. Vaswani et al., "Attention Is All You Need," in *Advances in Neural
Information Processing Systems*, vol. 30, pp. 5998-6008, 2017.

[12] L. Rampasek, M. Galkin, V. P. Dwivedi, A. T. Luu, G. Wolf, and
D. Beaini, "Recipe for a General, Powerful, Scalable Graph Transformer," in
*Advances in Neural Information Processing Systems*, vol. 35,
pp. 14501-14515, 2022.

[13] E. Altman, J. Blanusa, L. von Niederhausern, B. Egressy, A. Anghel, and
K. Atasu, "Realistic Synthetic Financial Transactions for Anti-Money
Laundering Models," in *Advances in Neural Information Processing Systems*,
vol. 36, pp. 29851-29874, 2023, doi: 10.52202/075280-1300.

[14] D. Xu, C. Ruan, E. Korpeoglu, S. Kumar, and K. Achan, "Inductive
Representation Learning on Temporal Graphs," in *International Conference on
Learning Representations*, 2020.

[15] E. Rossi, B. Chamberlain, F. Frasca, D. Eynard, F. Monti, and
M. Bronstein, "Temporal Graph Networks for Deep Learning on Dynamic Graphs,"
arXiv:2006.10637, 2020.

[16] M. Cardoso, P. Saleiro, and P. Bizarro, "LaundroGraph:
Self-Supervised Graph Representation Learning for Anti-Money Laundering," in
*Proc. 3rd ACM International Conference on AI in Finance*, pp. 130-138, 2022,
doi: 10.1145/3533271.3561727.

[17] C. Bellei, M. Xu, R. Phillips, T. Robinson, M. Weber, T. Kaler,
C. E. Leiserson, Arvind, and J. Chen, "The Shape of Money Laundering:
Subgraph Representation Learning on the Blockchain with the Elliptic2
Dataset," in *KDD Workshop on Machine Learning in Finance*, 2024,
arXiv:2404.19109.

[18] S. Gounoue, A. Sao, and S. Gottschalk, "TeMP-TraG: Edge-Based Temporal
Message Passing in Transaction Graphs," arXiv:2503.16901, 2025.

[19] C.-H. Poon, J. Kwok, C. Chow, and J.-H. Choi, "LineMVGNN:
Anti-Money Laundering with Line-Graph-Assisted Multi-View Graph Neural
Networks," *AI*, vol. 6, no. 4, art. 69, 2025, doi:
10.3390/ai6040069.

[20] S. Huang, F. Poursafaei, R. Rabbany, G. Rabusseau, and E. Rossi, "UTG:
Towards a Unified View of Snapshot and Event Based Models for Temporal
Graphs," in *Proc. 3rd Learning on Graphs Conference*, PMLR, vol. 269,
pp. 28:1-28:16, 2025.

## Evidence-to-Claim Audit Status

| Claim area | Current evidence | Status |
|---|---|---|
| FraudGT architecture and evaluation | Local full-text PDF plus source code | Verified locally |
| CDVT architecture and formulas | Commit `9038f85` source code | Verified against implementation |
| Phase 1 model selection | Recorded validation trajectories and freeze document | Verified for screening |
| Normal/shuffled/off mechanism | Final Small-LI and Large-LI manifests | Supported on two scales |
| Six-dataset comparison | Final Phase 1/2 manifests and protocol summary | 5/6 wins vs PE-FraudGT; mean delta +0.02716 |
| Multi-seed stability | Paired follow-up active at `f7209f2` | Must not claim yet |
| External generalization | No external result | Must not claim |
| Reference metadata [1]-[20] | DOI, arXiv, DBLP, PMLR, OpenReview indexing, and proceedings records | Independently verified |
| Citation-to-claim alignment | Partial local source inspection | Final full-text audit pending |
