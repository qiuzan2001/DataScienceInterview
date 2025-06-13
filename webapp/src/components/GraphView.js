import React, { useEffect, useState, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph';
import { useNavigate } from 'react-router-dom';

export default function GraphView() {
  const [graph, setGraph] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetch('/graph.json').then(res => res.json()).then(setGraph);
  }, []);

  const handleClick = useCallback(node => {
    navigate(`/note/${encodeURIComponent(node.id)}`);
  }, [navigate]);

  if (!graph) return <div>Loading graph&hellip;</div>;

  return (
    <ForceGraph2D
      graphData={graph}
      onNodeClick={handleClick}
      nodeAutoColorBy="id"
    />
  );
}
