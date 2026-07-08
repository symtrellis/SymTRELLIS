import { useEffect, useState } from 'react';
import type {
  DagEdgeRoute,
  DagStatusByNode,
  ModelDagEdge,
  ModelDagLayout,
  ModelDagNode,
  NodeInstanceId,
} from '../models/types';
import type { DagStatus } from '../types';

type DagPanelProps = {
  chosenEdgeIds: string[];
  currentNodeId: NodeInstanceId;
  edges: ModelDagEdge[];
  layout: ModelDagLayout;
  nodes: ModelDagNode[];
  statusByNode: DagStatusByNode;
};

type Side = 'top' | 'right' | 'bottom' | 'left';

type Point = {
  x: number;
  y: number;
};

type NodeBox = {
  current: boolean;
  height: number;
  id: NodeInstanceId;
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

type RenderedDag = {
  edges: EdgeRoute[];
  height: number;
  nodes: NodeBox[];
  width: number;
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

function edgeStatus(edge: ModelDagEdge, statusByNode: DagStatusByNode, chosenEdgeIds: string[]): DagStatus {
  const targetStatus = statusByNode[edge.target];

  if (!chosenEdgeIds.includes(edge.id)) {
    return 'inactive';
  }

  return targetStatus === 'current' ? 'current' : 'completed';
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

function edgePoints(
  edge: ModelDagEdge,
  route: DagEdgeRoute,
  boxById: Map<NodeInstanceId, NodeBox>,
  width: number,
  compact: boolean,
) {
  const source = boxById.get(edge.source)!;
  const target = boxById.get(edge.target)!;
  const overlap = compact ? 4 : 6;

  if (route === 'side_branch') {
    const start = sidePoint(source, 'left', overlap);
    const end = sidePoint(target, 'top', overlap);
    return [start, { x: end.x, y: start.y }, end];
  }

  if (route === 'side_merge') {
    const start = sidePoint(source, 'bottom', overlap);
    const end = sidePoint(target, 'left', overlap);
    return [start, { x: start.x, y: end.y }, end];
  }

  if (route === 'right_bypass') {
    const start = sidePoint(source, 'right', overlap);
    const end = sidePoint(target, 'right', overlap);
    const routeX = width - (compact ? 7 : 14);
    return [start, { x: routeX, y: start.y }, { x: routeX, y: end.y }, end];
  }

  return [sidePoint(source, 'bottom', overlap), sidePoint(target, 'top', overlap)];
}

function renderDag(
  nodes: ModelDagNode[],
  edges: ModelDagEdge[],
  dagLayout: ModelDagLayout,
  compact: boolean,
  currentNodeId: NodeInstanceId,
  statusByNode: DagStatusByNode,
  chosenEdgeIds: string[],
): RenderedDag {
  const nodeHeightValue = nodeHeight(compact);
  const rankGap = compact ? 10 : 22;
  const margin = compact ? 6 : 8;
  const laneGap = compact ? 10 : 7;
  const rightRouteGap = compact ? 12 : 28;
  const maxRank = Math.max(...nodes.map((node) => dagLayout.nodes[node.id].rank));
  const measuredNodes = nodes.map((node) => ({
    ...node,
    height: nodeHeightValue,
    width: nodeWidth(node.shortLabel, compact),
  }));
  const leftNodes = measuredNodes.filter((node) => dagLayout.nodes[node.id].lane === 'left');
  const mainNodes = measuredNodes.filter((node) => dagLayout.nodes[node.id].lane === 'main');
  const leftWidth = Math.max(0, ...leftNodes.map((node) => node.width));
  const mainWidth = Math.max(0, ...mainNodes.map((node) => node.width));
  const leftCenterX = margin + leftWidth / 2;
  const mainCenterX = margin + leftWidth + laneGap + mainWidth / 2;
  const width = margin * 2 + leftWidth + laneGap + mainWidth + rightRouteGap;
  const height = margin * 2 + (maxRank + 1) * nodeHeightValue + maxRank * rankGap;
  const boxes = measuredNodes.map((node) => {
    const layout = dagLayout.nodes[node.id];
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
  const edgeRoutes = edges.map((edge) => {
    const route = dagLayout.edges[edge.id].route;
    return {
      id: edge.id,
      path: roundedPath(edgePoints(edge, route, boxById, width, compact), compact ? 6 : 12),
      status: edgeStatus(edge, statusByNode, chosenEdgeIds),
    };
  });

  return { edges: edgeRoutes, height, nodes: boxes, width };
}

export function DagPanel({ chosenEdgeIds, currentNodeId, edges, layout, nodes, statusByNode }: DagPanelProps) {
  const compact = useMediaQuery('(max-width: 760px)');
  const rendered = renderDag(nodes, edges, layout, compact, currentNodeId, statusByNode, chosenEdgeIds);

  return (
    <section
      className="dag-panel"
      style={{ height: rendered.height, width: rendered.width }}
      aria-label="Generation path"
    >
      <svg
        className="dag-edges"
        viewBox={`0 0 ${rendered.width} ${rendered.height}`}
        aria-hidden="true"
      >
        {rendered.edges.map((edge) => (
          <path key={edge.id} className="dag-edge dag-edge--base" d={edge.path} />
        ))}
      </svg>

      <div className="dag-nodes">
        {rendered.nodes.map((node) => (
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
        viewBox={`0 0 ${rendered.width} ${rendered.height}`}
        aria-hidden="true"
      >
        {rendered.edges.map((edge) =>
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
