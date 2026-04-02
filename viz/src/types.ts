export enum Status {
  Support = 'support',
  Falsified = 'falsified',
  Inconclusive = 'inconclusive',
  InProgress = 'in_progress',
  Secondary = 'secondary',
}

export interface GraphNode {
  id: string;
  title: string;
  summary: string;
  formulas: string[];       // LaTeX strings
  status: Status;
  position: { x: number; y: number };
  cluster: string;
}

export interface GraphEdge {
  id: string;
  source: string;           // node id
  target: string;           // node id
  label: string;
}

export const STATUS_COLORS: Record<Status, { fill: string; border: string; text: string }> = {
  [Status.Support]:      { fill: '#d4edda', border: '#2d8a4e', text: '#155724' },
  [Status.Falsified]:    { fill: '#f8d7da', border: '#c0392b', text: '#721c24' },
  [Status.Inconclusive]: { fill: '#fff3cd', border: '#d4a017', text: '#856404' },
  [Status.InProgress]:   { fill: '#cce5ff', border: '#2980b9', text: '#004085' },
  [Status.Secondary]:    { fill: '#e2e3e5', border: '#6c757d', text: '#383d41' },
};
