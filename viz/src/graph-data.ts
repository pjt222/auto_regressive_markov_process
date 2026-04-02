import { Status, GraphNode, GraphEdge } from './types';

export const nodes: GraphNode[] = [
  // === Core ===
  {
    id: 'core-thesis',
    title: 'Core Thesis',
    summary:
      'Spinor SSM achieves better prediction than Diagonal when data has genuine rotational structure. Domain-specific: helps on rotation (+20-22%), hurts on language (-14-20%).',
    formulas: [
      'h_{t+1} = \\lambda_t \\cdot u_t h_t u_t^{-1} + \\bar{B}_t x_t',
      'u_t = \\exp(B_t/2) \\in \\text{Spin}(D)',
    ],
    status: Status.Support,
    position: { x: 0, y: 0 },
    cluster: 'core',
  },

  // === Theory ===
  {
    id: 'state-definition',
    title: 'State Definition (v0.2)',
    summary:
      'Hidden state is the full vector h_t in R^D, not the projected direction on S^{D-1}. Norm carries history information.',
    formulas: ['h_t \\in \\mathbb{R}^D', '\\mathcal{S} = \\mathbb{R}^D'],
    status: Status.Support,
    position: { x: -700, y: 500 },
    cluster: 'theory',
  },
  {
    id: 'k1-disproof',
    title: 'k=1 Disproof',
    summary:
      'The quotient to S^{D-1} is NOT Markov. The norm ||h_t|| acts as a hidden variable leaking history through the mixing weight.',
    formulas: [
      '\\|h_{t+1}\\|^2 = \\lambda_t^2 \\|h_t\\|^2 + 2\\lambda_t \\|h_t\\| \\langle v_t, \\bar{B}_t x_t \\rangle + \\|\\bar{B}_t x_t\\|^2',
    ],
    status: Status.Falsified,
    position: { x: -500, y: 500 },
    cluster: 'theory',
  },
  {
    id: 'norm-leakage',
    title: 'Norm Leakage',
    summary:
      'Norm carries geometrically-decayed history of all previous inputs. The pair (direction, norm) IS Markov, but direction alone is not.',
    formulas: ['r_{t+1} \\approx r_t + \\log\\lambda_t + O(e^{-r_t})'],
    status: Status.Inconclusive,
    position: { x: -500, y: 350 },
    cluster: 'theory',
  },
  {
    id: 'epsilon-bound',
    title: 'Epsilon Bound (#25)',
    summary:
      'Simple norm-variance bound yields R^2=0.19 (inadequate). Two-channel leakage model needed — history leaks through both norm AND direction.',
    formulas: [
      '\\epsilon(1,D) \\leq C \\cdot \\frac{\\text{Var}[\\log\\|h_t\\|]}{D}',
    ],
    status: Status.Inconclusive,
    position: { x: -350, y: 350 },
    cluster: 'theory',
  },
  {
    id: 'clifford-grades',
    title: 'Clifford Grade Hierarchy',
    summary:
      'Grade 0 (scalar): universal. Grade 1 (per-dim decay): universal. Grade 2 (rotation): domain-specific. This predicts when rotation helps or hurts.',
    formulas: [
      '\\text{Grade 1: } \\Lambda_t \\odot h_t',
      '\\text{Grade 2: } u_t h_t u_t^{-1}',
    ],
    status: Status.Support,
    position: { x: -700, y: 350 },
    cluster: 'theory',
  },
  {
    id: 'scalability',
    title: 'Scalability Options',
    summary:
      'Three parameterization schemes: grade truncation O(D^2), factored Cl(3,0) blocks O(D), and low-rank bivectors O(rD).',
    formulas: [
      '\\binom{D}{2} = \\frac{D(D-1)}{2}',
      '\\mathbb{R}^D = \\bigoplus_{i=1}^{D/3} \\mathbb{R}^3',
    ],
    status: Status.Secondary,
    position: { x: -700, y: 200 },
    cluster: 'theory',
  },
  {
    id: 'markov-score',
    title: 'Markov Score',
    summary:
      'Mutual information gap measuring non-Markovianity. Spinor has higher MI (0.18) but uses it productively for better prediction.',
    formulas: ['I(s_{t+1}; s_{t-j} \\mid s_t) = \\epsilon(j, D)'],
    status: Status.Support,
    position: { x: -350, y: 500 },
    cluster: 'theory',
  },

  // === Toy Experiments ===
  {
    id: 'exp-30',
    title: '#30: D=3 Training',
    summary:
      'Spinor+Decay beats Diagonal by 8.7% on test loss (0.5882 vs 0.6446) on 2nd-order Markov toy data. Spinor carries more history productively.',
    formulas: [
      '\\text{Spinor: } 0.5882 \\pm 0.004',
      '\\text{Diagonal: } 0.6446 \\pm 0.006',
    ],
    status: Status.Support,
    position: { x: -700, y: -250 },
    cluster: 'toy',
  },
  {
    id: 'exp-38',
    title: '#38: Dimension Scaling',
    summary:
      'Spinor advantage persists at D in {3, 8, 16} with Givens parameterization. Gap narrows slightly from -6.3% to -4.8%.',
    formulas: ['\\Delta = -6.3\\% \\text{ (D=3)}, -4.8\\% \\text{ (D=16)}'],
    status: Status.Support,
    position: { x: -550, y: -250 },
    cluster: 'toy',
  },
  {
    id: 'exp-45',
    title: '#45: Dense Ablation',
    summary:
      'CRITICAL: Spinor beats unconstrained Dense at D>=9 despite 6-10x fewer parameters. Geometry carries structural load beyond non-diagonality.',
    formulas: [
      '\\text{Spinor} > \\text{Dense}_{D \\times D} \\text{ at } D \\geq 9',
    ],
    status: Status.Support,
    position: { x: -400, y: -250 },
    cluster: 'toy',
  },
  {
    id: 'exp-44',
    title: '#44: Block Size',
    summary:
      'Sharp threshold: any rotation (block size >= 2) beats no rotation (block size = 1). Flat performance across block sizes 2-12.',
    formulas: ['b=1 \\text{ (diagonal)} \\ll b \\geq 2 \\text{ (rotation)}'],
    status: Status.Support,
    position: { x: -700, y: -400 },
    cluster: 'toy',
  },
  {
    id: 'exp-40',
    title: '#40: Discrete Spinor',
    summary:
      'Gumbel-softmax over binary icosahedral group 2I collapsed with ES optimizer — all probability mass on one element. Inconclusive.',
    formulas: ['u_t \\in 2I \\subset \\text{Spin}(3), \\; |2I| = 120'],
    status: Status.Inconclusive,
    position: { x: -550, y: -400 },
    cluster: 'toy',
  },
  {
    id: 'exp-47',
    title: '#47: Convergence',
    summary:
      'No spontaneous convergence of trained quaternions toward polyhedral group elements. Continuous rotation manifold preferred by optimization.',
    formulas: ['d(q, 2I) \\not\\to 0'],
    status: Status.Inconclusive,
    position: { x: -400, y: -400 },
    cluster: 'toy',
  },

  // === Real Data Experiments ===
  {
    id: 'exp-31',
    title: '#31: Language (WikiText-2)',
    summary:
      'Diagonal SSM WINS decisively: 469 PPL vs QuatBlock 562 (+20%) and Pascal 535 (+14%). Rotation is the wrong inductive bias for language.',
    formulas: ['\\text{Diagonal: 469 PPL}', '\\text{QuatBlock: 562 PPL}'],
    status: Status.Falsified,
    position: { x: 500, y: -300 },
    cluster: 'real-data',
  },
  {
    id: 'exp-50',
    title: '#50: SO(3) Rotation',
    summary:
      'Givens (2.24 deg) and QuatBlock (2.30 deg) beat Diagonal (2.88 deg) by 20-22% on synthetic SO(3) random walks. Rotation IS the right bias here.',
    formulas: [
      '\\text{Givens: } 2.24°',
      '\\text{QuatBlock: } 2.30°',
      '\\text{Diagonal: } 2.88°',
    ],
    status: Status.Support,
    position: { x: 700, y: -300 },
    cluster: 'real-data',
  },
  {
    id: 'exp-51',
    title: '#51: Gated Rotation',
    summary:
      'Learned gate blends rotation and diagonal paths. Gate should open on rotation data (near 1) and close on language (near 0). Init fix in progress.',
    formulas: [
      'h_{t+1} = g_t \\odot h_{\\text{rot}} + (1-g_t) \\odot h_{\\text{diag}} + \\bar{B}_t x_t',
    ],
    status: Status.InProgress,
    position: { x: 600, y: -450 },
    cluster: 'real-data',
  },

  // === Literature ===
  {
    id: 'adversarial-review',
    title: 'Adversarial Literature Review',
    summary:
      'Seven papers challenged the thesis. All addressed: Cirone resolved by dense ablation, Movahedi confirmed by language results, Olsson confirmed.',
    formulas: [],
    status: Status.Support,
    position: { x: 500, y: 400 },
    cluster: 'literature',
  },
  {
    id: 'polyhedral-spinors',
    title: 'Polyhedral Spinor Connections',
    summary:
      "Binary icosahedral 2I is a subgroup of Spin(3). ADE classification connects to E8 via Dechant's spinor induction theorem.",
    formulas: [
      '2I \\subset \\text{Spin}(3) \\cong SU(2)',
      'H_3 \\xrightarrow{\\text{spinor}} H_4 \\to E_8',
    ],
    status: Status.Secondary,
    position: { x: 700, y: 400 },
    cluster: 'literature',
  },
  {
    id: 'barbour',
    title: "Barbour's Complexity",
    summary:
      "Julian Barbour's relational physics inspired the embedding space intuition. Analogy is apt but no formal homomorphism exists.",
    formulas: [],
    status: Status.Secondary,
    position: { x: 500, y: 250 },
    cluster: 'literature',
  },
  {
    id: 'geometric-dl',
    title: 'Geometric Deep Learning',
    summary:
      'Bronstein et al. framework provides mathematical foundation. Clifford algebras (Ruhe et al.) supersede division algebra hierarchy.',
    formulas: ['Cl(p,q), \\; \\text{Spin}(p,q)'],
    status: Status.Secondary,
    position: { x: 700, y: 250 },
    cluster: 'literature',
  },

  // === Infrastructure ===
  {
    id: 'cuda-kernels',
    title: 'CUDA Kernels (#53)',
    summary:
      'Four kernels restructured to multi-thread blocks (up to 256 threads). 732x speedup on language task. All using stride loops.',
    formulas: ['\\text{<<<B, min(N,256)>>>}'],
    status: Status.Support,
    position: { x: 950, y: 0 },
    cluster: 'infrastructure',
  },

  // === Open Questions ===
  {
    id: 'open-questions',
    title: 'Open Questions',
    summary:
      'Theoretical: sufficient statistic, stationary distribution, mixing time, norm concentration. Empirical: D=3/D=48 scaling, N-body physics, re-run #40.',
    formulas: [
      '\\text{Var}[\\log\\|h_t\\|] \\to 0 \\text{ as } D \\to \\infty \\text{ ?}',
    ],
    status: Status.InProgress,
    position: { x: 0, y: -650 },
    cluster: 'open',
  },
];

export const edges: GraphEdge[] = [
  {
    id: 'e-core-state',
    source: 'core-thesis',
    target: 'state-definition',
    label: 'formalizes h_t in R^D',
  },
  {
    id: 'e-core-k1',
    source: 'core-thesis',
    target: 'k1-disproof',
    label: 'disproved original quotient claim',
  },
  {
    id: 'e-k1-norm',
    source: 'k1-disproof',
    target: 'norm-leakage',
    label: 'norm is the hidden variable',
  },
  {
    id: 'e-norm-epsilon',
    source: 'norm-leakage',
    target: 'epsilon-bound',
    label: 'bounds non-Markovianity',
  },
  {
    id: 'e-core-clifford',
    source: 'core-thesis',
    target: 'clifford-grades',
    label: 'predicts grade-2 domain-specific',
  },
  {
    id: 'e-clifford-lang',
    source: 'clifford-grades',
    target: 'exp-31',
    label: 'grade-2 hurts on language (confirmed)',
  },
  {
    id: 'e-clifford-rot',
    source: 'clifford-grades',
    target: 'exp-50',
    label: 'grade-2 helps on rotation (confirmed)',
  },
  {
    id: 'e-30-38',
    source: 'exp-30',
    target: 'exp-38',
    label: 'scales to higher dimensions',
  },
  {
    id: 'e-38-45',
    source: 'exp-38',
    target: 'exp-45',
    label: 'rules out non-diagonality explanation',
  },
  {
    id: 'e-45-adversarial',
    source: 'exp-45',
    target: 'adversarial-review',
    label: 'resolves Cirone et al. challenge',
  },
  {
    id: 'e-31-51',
    source: 'exp-31',
    target: 'exp-51',
    label: 'motivated learnable coupling',
  },
  {
    id: 'e-50-51',
    source: 'exp-50',
    target: 'exp-51',
    label: 'test if gate opens on rotation',
  },
  {
    id: 'e-core-31',
    source: 'core-thesis',
    target: 'exp-31',
    label: 'falsified universal version',
  },
  {
    id: 'e-core-50',
    source: 'core-thesis',
    target: 'exp-50',
    label: 'confirmed domain-specific version',
  },
  {
    id: 'e-poly-40',
    source: 'polyhedral-spinors',
    target: 'exp-40',
    label: 'motivates discrete spinor test',
  },
  {
    id: 'e-poly-47',
    source: 'polyhedral-spinors',
    target: 'exp-47',
    label: 'tests convergence toward 2I',
  },
  {
    id: 'e-cuda-31',
    source: 'cuda-kernels',
    target: 'exp-31',
    label: 'enables language scale (732x)',
  },
  {
    id: 'e-gdl-core',
    source: 'geometric-dl',
    target: 'core-thesis',
    label: 'Clifford algebra framework',
  },
  {
    id: 'e-barbour-core',
    source: 'barbour',
    target: 'core-thesis',
    label: 'inspirational, not formal',
  },
  {
    id: 'e-adversarial-core',
    source: 'adversarial-review',
    target: 'core-thesis',
    label: 'all 7 challenges addressed',
  },
  {
    id: 'e-state-markov',
    source: 'state-definition',
    target: 'markov-score',
    label: 'defines MI measurement',
  },
  {
    id: 'e-markov-30',
    source: 'markov-score',
    target: 'exp-30',
    label: 'Spinor MI=0.18, Diag MI=0.009',
  },
  {
    id: 'e-44-scale',
    source: 'exp-44',
    target: 'scalability',
    label: 'any rotation beats none',
  },
  {
    id: 'e-open-core',
    source: 'open-questions',
    target: 'core-thesis',
    label: '4 theoretical + 6 empirical open',
  },
  {
    id: 'e-core-30',
    source: 'core-thesis',
    target: 'exp-30',
    label: 'initial D=3 validation',
  },
];
