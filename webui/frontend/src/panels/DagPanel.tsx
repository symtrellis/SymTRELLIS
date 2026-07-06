import { useEffect, useState } from 'react';
import type { DagEdge, DagNode, DagStatus, NodeId } from '../types';

type DagPanelProps = {
  currentNodeId: NodeId;
  edges: DagEdge[];
  nodes: DagNode[];
  statusByNode: Record<NodeId, DagStatus>;
};

type Lane = 'left' | 'main';
type Side = 'top' | 'right' | 'bottom' | 'left';

type Point = {
  x: number;
  y: number;
};

type NodeBox = {
  current: boolean;
  height: number;
  id: NodeId;
  label: string;
  status: DagStatus;
  width: number;
  x: number;
  y: number;
};

type EdgeRoute = {
  id: string;
  path: string;
  status: DagStatus;
};

type DagLayout = {
  edges: EdgeRoute[];
  height: number;
  nodes: NodeBox[];
  width: number;
};

const nodeLayout: Record<NodeId, { lane: Lane; rank: number }> = {
  img_cond: { lane: 'main', rank: 0 },
  nat_ss: { lane: 'main', rank: 1 },
  nat_shape: { lane: 'main', rank: 2 },
  manual_sym: { lane: 'left', rank: 3 },
  detect_sym: { lane: 'main', rank: 3 },
  sym_ss: { lane: 'main', rank: 4 },
  sym_shape: { lane: 'main', rank: 5 },
  texture: { lane: 'main', rank: 6 },
};

function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);

  useEffect(() => {
    const media = window.matchMedia(query);
    const updateMatches = () => setMatches(media.matches);
    media.addEventListener('change', updateMatches);
    return () => media.removeEventListener('change', updateMatches);
  }, [query]);

  return matches;
}

function edgeStatus(edge: DagEdge, statusByNode: Record<NodeId, DagStatus>): DagStatus {
  const sourceStatus = statusByNode[edge.source];
  const targetStatus = statusByNode[edge.target];

  if (sourceStatus === 'completed' && targetStatus === 'completed') {
    return 'completed';
  }

  if (sourceStatus === 'completed' && targetStatus === 'current') {
    return 'current';
  }

  return 'inactive';
}

function nodeWidth(label: string, compact: boolean) {
  return compact ? 12 : Math.ceil(label.length * 7.4 + 18);
}

function nodeHeight(compact: boolean) {
  return compact ? 12 : 24;
}

function sidePoint(node: NodeBox, side: Side, overlap: number): Point {
  if (side === 'top') {
    return { x: node.x + node.width / 2, y: node.y + overlap };
  }

  if (side === 'right') {
    return { x: node.x + node.width - overlap, y: node.y + node.height / 2 };
  }

  if (side === 'bottom') {
    return { x: node.x + node.width / 2, y: node.y + node.height - overlap };
  }

  return { x: node.x + overlap, y: node.y + node.height / 2 };
}

function roundedPath(points: Point[], radius: number) {
  const parts = [`M ${points[0].x} ${points[0].y}`];

  for (let index = 1; index < points.length - 1; index += 1) {
    const previous = points[index - 1];
    const current = points[index];
    const next = points[index + 1];
    const incoming = { x: previous.x - current.x, y: previous.y - current.y };
    const outgoing = { x: next.x - current.x, y: next.y - current.y };
    const incomingLength = Math.hypot(incoming.x, incoming.y);
    const outgoingLength = Math.hypot(outgoing.x, outgoing.y);
    const cornerRadius = Math.min(radius, incomingLength / 2, outgoingLength / 2);
    const before = {
      x: current.x + (incoming.x / incomingLength) * cornerRadius,
      y: current.y + (incoming.y / incomingLength) * cornerRadius,
    };
    const after = {
      x: current.x + (outgoing.x / outgoingLength) * cornerRadius,
      y: current.y + (outgoing.y / outgoingLength) * cornerRadius,
    };

    parts.push(`L ${before.x} ${before.y}`);
    parts.push(`Q ${current.x} ${current.y} ${after.x} ${after.y}`);
  }

  const last = points[points.length - 1];
  parts.push(`L ${last.x} ${last.y}`);
  return parts.join(' ');
}

function edgePoints(edge: DagEdge, boxById: Map<NodeId, NodeBox>, width: number, compact: boolean) {
  const source = boxById.get(edge.source)!;
  const target = boxById.get(edge.target)!;
  const overlap = compact ? 4 : 6;

  if (edge.id === 'img_cond-manual_sym') {
    const start = sidePoint(source, 'left', overlap);
    const end = sidePoint(target, 'top', overlap);
    return [start, { x: end.x, y: start.y }, end];
  }

  if (edge.id === 'manual_sym-sym_ss') {
    const start = sidePoint(source, 'bottom', overlap);
    const end = sidePoint(target, 'left', overlap);
    return [start, { x: start.x, y: end.y }, end];
  }

  if (edge.id === 'nat_shape-texture') {
    const start = sidePoint(source, 'right', overlap);
    const end = sidePoint(target, 'right', overlap);
    const routeX = width - (compact ? 7 : 14);
    return [start, { x: routeX, y: start.y }, { x: routeX, y: end.y }, end];
  }

  return [sidePoint(source, 'bottom', overlap), sidePoint(target, 'top', overlap)];
}

function layoutDag(
  nodes: DagNode[],
  edges: DagEdge[],
  compact: boolean,
  currentNodeId: NodeId,
  statusByNode: Record<NodeId, DagStatus>,
): DagLayout {
  const nodeHeightValue = nodeHeight(compact);
  const rankGap = compact ? 10 : 22;
  const margin = compact ? 6 : 8;
  const laneGap = compact ? 10 : 7;
  const rightRouteGap = compact ? 12 : 28;
  const maxRank = Math.max(...nodes.map((node) => nodeLayout[node.id].rank));
  const measuredNodes = nodes.map((node) => ({
    ...node,
    height: nodeHeightValue,
    width: nodeWidth(node.shortLabel, compact),
  }));
  const leftWidth = Math.max(
    ...measuredNodes
      .filter((node) => nodeLayout[node.id].lane === 'left')
      .map((node) => node.width),
  );
  const mainWidth = Math.max(
    ...measuredNodes
      .filter((node) => nodeLayout[node.id].lane === 'main')
      .map((node) => node.width),
  );
  const leftCenterX = margin + leftWidth / 2;
  const mainCenterX = margin + leftWidth + laneGap + mainWidth / 2;
  const width = margin * 2 + leftWidth + laneGap + mainWidth + rightRouteGap;
  const height = margin * 2 + (maxRank + 1) * nodeHeightValue + maxRank * rankGap;
  const boxes = measuredNodes.map((node) => {
    const layout = nodeLayout[node.id];
    const centerX = layout.lane === 'left' ? leftCenterX : mainCenterX;
    return {
      current: node.id === currentNodeId,
      height: node.height,
      id: node.id,
      label: node.shortLabel,
      status: statusByNode[node.id],
      width: node.width,
      x: centerX - node.width / 2,
      y: margin + layout.rank * (nodeHeightValue + rankGap),
    };
  });
  const boxById = new Map(boxes.map((node) => [node.id, node]));
  const routes = edges.map((edge) => ({
    id: edge.id,
    path: roundedPath(edgePoints(edge, boxById, width, compact), compact ? 6 : 12),
    status: edgeStatus(edge, statusByNode),
  }));

  return { edges: routes, height, nodes: boxes, width };
}

export function DagPanel({ currentNodeId, edges, nodes, statusByNode }: DagPanelProps) {
  const compact = useMediaQuery('(max-width: 760px)');
  const layout = layoutDag(nodes, edges, compact, currentNodeId, statusByNode);

  return (
    <section
      className="dag-panel"
      style={{ height: layout.height, width: layout.width }}
      aria-label="Generation path"
    >
      <svg
        className="dag-edges"
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        aria-hidden="true"
      >
        {layout.edges.map((edge) => (
          <path key={edge.id} className="dag-edge dag-edge--base" d={edge.path} />
        ))}
      </svg>

      <div className="dag-nodes">
        {layout.nodes.map((node) => (
          <div
            key={node.id}
            className={`dag-node dag-node--${node.status}${node.current ? ' dag-node--active' : ''}`}
            style={{
              height: node.height,
              left: node.x,
              top: node.y,
              width: node.width,
            }}
          >
            <span className="dag-node-dot" aria-hidden="true" />
            <span className="dag-node-label">{node.label}</span>
          </div>
        ))}
      </div>

      <svg
        className="dag-progress-edges"
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        aria-hidden="true"
      >
        {layout.edges.map((edge) =>
          edge.status !== 'inactive' ? (
            <path
              key={edge.id}
              className={`dag-edge dag-edge--progress dag-edge--${edge.status}`}
              d={edge.path}
            />
          ) : null,
        )}
      </svg>
    </section>
  );
}
